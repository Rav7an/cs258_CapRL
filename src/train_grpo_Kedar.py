import os
import json
import re
import logging
from typing import List, Dict, Any
from collections import deque
from datetime import datetime
import time  # <-- for ETA

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from PIL import Image
import matplotlib.pyplot as plt

from qwen_vl_utils import process_vision_info

# ===========================
# Logging
# ===========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================
DATASET_PATH = "caprl_mcq_dataset_final.jsonl"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "output/grpo_qwen2vl_3"

VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Hyperparameters
BATCH_SIZE = 2              # group size for GRPO
RL_STEPS = 100_000          # cap, you can stop earlier
LR = 5e-6                   # smaller than before

MAX_NEW_TOKENS_CAPTION = 64
MAX_NEW_TOKENS_ANSWER = 32

CAPTION_TEMPERATURE = 0.7   # exploration for captions
TOP_P = 0.9

ANSWER_TEMPERATURE = 0.0    # deterministic reward

REWARD_WINDOW = 500         # moving average window (for logging only)

CHECKPOINT_EVERY = 250   # periodic checkpoint

# Evaluation config
EVAL_EVERY = 500          # eval frequency
EVAL_SAMPLES = 26           # eval rollout size

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_BEST = os.path.join(OUTPUT_DIR, "best")
os.makedirs(CHECKPOINT_BEST, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.jsonl")
EVAL_LOG_PATH = os.path.join(OUTPUT_DIR, "eval_log.jsonl")

# ===========================
# Utils
# ===========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items

def normalize_output_to_label(text: str) -> str:
    t = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1) if m else ""

def load_image(img_path: str) -> Image.Image:
    if os.path.exists(img_path):
        full_path = img_path
    else:
        full_path = os.path.join(IMAGE_DIR, img_path)
        if not os.path.exists(full_path):
            filename = os.path.basename(img_path)
            full_path = os.path.join(IMAGE_DIR, filename)
    return Image.open(full_path).convert("RGB")

def build_answer_prompt(caption: str, question_text: str) -> str:
    return f"""You are answering multiple-choice questions about an image.
You are NOT given the image, only a caption that describes it.

Caption:
{caption}

Now answer the following multiple-choice question based ONLY on the caption.
Return ONLY the letter of the correct option (A, B, C, or D).

Question and options:
{question_text}

Answer:"""

# ===========================
# Dataset
# ===========================
class ImageMcqDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_rel = it.get("image_path") or it.get("image_id")
        question = it["question"]
        correct_label = it["correct_label"].strip().upper()
        return {
            "image_path": img_rel,
            "question": question,
            "correct_label": correct_label,
        }

# ===========================
# Models
# ===========================
def load_models():
    logger.info(f"Using device: {device}")
    logger.info(f"Loading VLM policy: {VL_MODEL_ID}...")
    vlm_policy = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    vlm_policy.train()
    vlm_processor = AutoProcessor.from_pretrained(VL_MODEL_ID)

    logger.info(f"Loading reward LLM: {LLM_MODEL_ID}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )

    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False


    return vlm_policy, vlm_processor, llm_model, llm_tokenizer

# ===========================
# Reward calculation
# ===========================
def compute_reward(llm_model, llm_tokenizer, caption: str, question: str, correct_label: str) -> float:
    prompt = build_answer_prompt(caption, question)
    device_llm = next(llm_model.parameters()).device
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device_llm)

    with torch.no_grad():
        gen_ids = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_ANSWER,
            do_sample=False,
            temperature=ANSWER_TEMPERATURE,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    gen_only = gen_ids[0, inputs["input_ids"].shape[1]:]
    out_text = llm_tokenizer.decode(gen_only, skip_special_tokens=True)
    pred_label = normalize_output_to_label(out_text)
    return 1.0 if pred_label == correct_label else 0.0

# ===========================
# Caption generation
# ===========================
def generate_caption_batch(model, processor, imgs, eval_mode: bool = False):
    """
    imgs: list of PIL Images of length B
    Returns:
      captions: List[str] length B
      gen_ids: LongTensor [B, T]
      inputs: dict from processor
    """
    caption_prompt = "Describe this image in a concise but detailed caption."

    messages_batch = []
    for img in imgs:
        messages_batch.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": caption_prompt},
                ],
            }
        )

    texts = [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS_CAPTION,
    }

    if eval_mode:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=CAPTION_TEMPERATURE, top_p=TOP_P))

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids_new = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids_new, skip_special_tokens=True)
    captions = [d.strip() for d in decoded]

    return captions, generated_ids, inputs

# ===========================
# Checkpoint & logging
# ===========================
def save_checkpoint(vlm_policy, vlm_processor, step: int, path: str, tag: str, extra_state: Dict[str, Any]):
    os.makedirs(path, exist_ok=True)
    logger.info(f"[{tag}] Saving checkpoint at step {step} to {path}")
    vlm_policy.save_pretrained(path)
    vlm_processor.save_pretrained(path)
    meta_path = os.path.join(path, "trainer_state.json")
    with open(meta_path, "w") as f:
        json.dump(extra_state, f, indent=2)

def append_log(record: Dict[str, Any], path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ===========================
# Evaluation
# ===========================
def evaluate_policy(vlm_policy, vlm_processor, llm_model, llm_tokenizer, dataset, num_samples: int) -> float:
    vlm_policy.eval()
    total_reward = 0.0
    n = min(num_samples, len(dataset))

    for i in range(n):
        sample = dataset[i]
        img_rel = sample["image_path"]
        question = sample["question"]
        correct_label = sample["correct_label"]

        try:
            img = load_image(img_rel)
        except Exception as e:
            logger.warning(f"[EVAL] Skipping {img_rel}: {e}")
            continue

        captions, _, _ = generate_caption_batch(vlm_policy, vlm_processor, [img], eval_mode=True)
        caption_text = captions[0]
        reward = compute_reward(llm_model, llm_tokenizer, caption_text, question, correct_label)
        total_reward += reward

    mean_reward = total_reward / float(n)
    return mean_reward

# ===========================
# Plotting
# ===========================
def plot_metrics():
    if not os.path.exists(LOG_PATH):
        logger.warning("No training log found, skipping plots.")
        return

    steps, train_rewards, train_mean_rewards, nll_losses, last_eval_list = [], [], [], [], []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps.append(r["step"])
            train_rewards.append(r["batch_mean_reward"])
            train_mean_rewards.append(r["mean_reward"])
            nll_losses.append(r["nll_loss"])
            last_eval_list.append(r.get("last_eval_mean_reward", None))

    eval_steps, eval_mean_rewards = [], []
    if os.path.exists(EVAL_LOG_PATH):
        with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eval_steps.append(r["step"])
                eval_mean_rewards.append(r["eval_mean_reward"])

    if steps:
        plt.figure()
        plt.plot(steps, train_mean_rewards, label="Train Mean Reward")
        # optionally overlay last eval reward where available
        if any(x is not None for x in last_eval_list):
            eval_overlay_steps = [s for s, v in zip(steps, last_eval_list) if v is not None]
            eval_overlay_vals = [v for v in last_eval_list if v is not None]
            if eval_overlay_steps:
                plt.plot(eval_overlay_steps, eval_overlay_vals, label="Last Eval Reward")
                plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Training Mean Reward (and Last Eval) vs Step")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "train_mean_reward.png"))
        plt.close()

    if eval_steps:
        plt.figure()
        plt.plot(eval_steps, eval_mean_rewards)
        plt.xlabel("Step")
        plt.ylabel("Eval Mean Reward")
        plt.title("Eval Mean Reward vs Step")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "eval_mean_reward.png"))
        plt.close()

    if steps:
        plt.figure()
        plt.plot(steps, nll_losses)
        plt.xlabel("Step")
        plt.ylabel("NLL Loss")
        plt.title("Training NLL Loss vs Step")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "train_nll_loss.png"))
        plt.close()

# ===========================
# Main GRPO loop
# ===========================
def main():
    logger.info("Loading dataset...")
    items = read_jsonl(DATASET_PATH)
    logger.info(f"Loaded {len(items)} samples.")
    dataset = ImageMcqDataset(items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vlm_policy, vlm_processor, llm_model, llm_tokenizer = load_models()
    optimizer = torch.optim.AdamW(vlm_policy.parameters(), lr=LR)

    step = 0
    reward_history = deque(maxlen=REWARD_WINDOW)
    best_mean_reward = -1.0
    last_eval_mean_reward = None

    data_iter = iter(dataloader)
    start_time = time.time()

    try:
        while step < RL_STEPS:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            img_rels = batch["image_path"]
            questions = batch["question"]
            correct_labels = batch["correct_label"]

            imgs = []
            valid_indices = []
            for i, img_rel in enumerate(img_rels):
                try:
                    imgs.append(load_image(img_rel))
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Skipping {img_rel}: {e}")

            if len(imgs) == 0:
                continue

            B = len(imgs)

            # 1) Sample captions (exploration)
            vlm_policy.eval()
            captions, gen_ids, inputs = generate_caption_batch(vlm_policy, vlm_processor, imgs, eval_mode=False)

            # 2) Compute rewards per sample
            rewards = []
            for idx_local, cap in enumerate(captions):
                idx_global = valid_indices[idx_local]
                q = questions[idx_global]
                c_label = correct_labels[idx_global]
                r = compute_reward(llm_model, llm_tokenizer, cap, q, c_label)
                rewards.append(r)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=vlm_policy.device)  # [B]

            # 3) Policy update (GRPO: per-sample NLL + group-normalized advantage)
            vlm_policy.train()

            labels = gen_ids.clone()
            prompt_len = inputs["input_ids"].shape[1]
            labels[:, :prompt_len] = -100  # ignore prompt in loss

            outputs = vlm_policy(
                input_ids=gen_ids,
                attention_mask=torch.ones_like(gen_ids),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                output_logits=True,
            )
            logits = outputs.logits  # [B, T, V]

            shift_logits = logits[:, :-1, :]        # [B, T-1, V]
            shift_labels = labels[:, 1:]           # [B, T-1]

            vocab_size = shift_logits.size(-1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_losses = loss_fct(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
            ).view(B, -1)  # [B, T-1]

            valid_mask = (shift_labels != -100).float()
            token_count = valid_mask.sum(dim=1) + 1e-8  # [B]
            nll_per_sample = (token_losses * valid_mask).sum(dim=1) / token_count  # [B]

            # GRPO: normalize rewards within group
            r_mean = rewards_t.mean()
            r_std = rewards_t.std(unbiased=False) + 1e-8
            advantages = (rewards_t - r_mean) / r_std  # [B]
            advantages_detached = advantages.detach()

            # Loss: mean over group of (advantage * NLL)
            reinforce_losses = advantages_detached * nll_per_sample  # [B]
            loss = reinforce_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm_policy.parameters(), 1.0)
            optimizer.step()

            step += 1

            # Logging stats
            mean_reward_batch = rewards_t.mean().item()
            reward_history.append(mean_reward_batch)
            mean_reward = sum(reward_history) / len(reward_history)
            mean_nll = nll_per_sample.mean().item()

            # ETA computation
            elapsed = time.time() - start_time
            steps_done = step
            if steps_done > 0:
                steps_per_sec = steps_done / max(elapsed, 1e-6)
                remaining_steps = RL_STEPS - steps_done
                eta_sec = remaining_steps / max(steps_per_sec, 1e-6)
            else:
                steps_per_sec = 0.0
                eta_sec = 0.0

            if step % 50 == 0:
                logger.info(
                    f"Step {step}/{RL_STEPS} | BatchMeanReward={mean_reward_batch:.3f} | "
                    f"TrainMeanReward={mean_reward:.3f} | NLL={mean_nll:.4f} "
                    f"| r_mean={r_mean.item():.3f}, r_std={r_std.item():.3f} "
                    f"| ETA={eta_sec/3600:.2f}h"
                )
                logger.info(f"  Example caption: {captions[0]}")

            # Best checkpoint by moving mean reward
            if len(reward_history) == REWARD_WINDOW and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                extra_state = {
                    "step": step,
                    "best_mean_reward": best_mean_reward,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(vlm_policy, vlm_processor, step, CHECKPOINT_BEST, tag="BEST", extra_state=extra_state)

            # Periodic checkpoint
            if step % CHECKPOINT_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-step-{step}")
                extra_state = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(vlm_policy, vlm_processor, step, ckpt_dir, tag="PERIODIC", extra_state=extra_state)

            # Log per-step stats (with last_eval_mean_reward)
            log_record = {
                "step": step,
                "batch_mean_reward": mean_reward_batch,
                "mean_reward": mean_reward,
                "nll_loss": float(mean_nll),
                "r_mean": float(r_mean.item()),
                "r_std": float(r_std.item()),
                "last_eval_mean_reward": float(last_eval_mean_reward) if last_eval_mean_reward is not None else None,
            }
            append_log(log_record, LOG_PATH)

            # Periodic evaluation
            if step % EVAL_EVERY == 0:
                logger.info(f"Running eval at step {step}...")
                eval_mean_reward = evaluate_policy(
                    vlm_policy, vlm_processor, llm_model, llm_tokenizer, dataset, EVAL_SAMPLES
                )
                last_eval_mean_reward = eval_mean_reward
                logger.info(f"[EVAL] Step {step} | EvalMeanReward={eval_mean_reward:.3f}")
                eval_record = {
                    "step": step,
                    "eval_mean_reward": eval_mean_reward,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                append_log(eval_record, EVAL_LOG_PATH)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected! Saving interrupt checkpoint...")
        interrupt_dir = os.path.join(OUTPUT_DIR, f"interrupt-step-{step}")
        extra_state = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "KeyboardInterrupt",
        }
        save_checkpoint(vlm_policy, vlm_processor, step, interrupt_dir, tag="INTERRUPT", extra_state=extra_state)

    # Final save
    logger.info("Saving final model...")
    final_state = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(),
        "best_mean_reward": best_mean_reward,
    }
    save_checkpoint(vlm_policy, vlm_processor, step, OUTPUT_DIR, tag="FINAL", extra_state=final_state)

    # Plot curves
    logger.info("Plotting training and eval curves...")
    plot_metrics()
    logger.info("Done.")

if __name__ == "__main__":
    main()
