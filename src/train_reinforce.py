import os
import json
import re
import logging
from typing import List, Dict, Any
from collections import deque
from datetime import datetime

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
OUTPUT_DIR = "output/reinforce_qwen2vl"

VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Hyperparameters
BATCH_SIZE = 1              # Online REINFORCE, 1 sample per update
RL_STEPS = 150_000          # You can set 200_000 if you want
LR = 1e-5

MAX_NEW_TOKENS_CAPTION = 64
MAX_NEW_TOKENS_ANSWER = 32

CAPTION_TEMPERATURE = 0.7   # Exploration for captions
TOP_P = 0.9

ANSWER_TEMPERATURE = 0.0    # Deterministic reward
BASELINE_MOMENTUM = 0.99    # Running baseline momentum
REWARD_WINDOW = 200         # Moving average window for training reward

CHECKPOINT_EVERY = 10_000   # Periodic checkpointing during training

# Evaluation config
EVAL_EVERY = 500         # Run eval every N steps
EVAL_SAMPLES = 26         # Number of samples per eval rollout

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
    # Try direct path
    if os.path.exists(img_path):
        full_path = img_path
    else:
        # Try with IMAGE_DIR + relative path
        full_path = os.path.join(IMAGE_DIR, img_path)
        if not os.path.exists(full_path):
            # Try IMAGE_DIR + basename
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
def generate_caption(model, processor, img: Image.Image, eval_mode: bool = False):
    """
    If eval_mode=True -> deterministic (do_sample=False, temp=0.0)
    If eval_mode=False -> sampling for RL (do_sample=True, temp=CAPTION_TEMPERATURE)
    """
    caption_prompt = "Describe this image in a concise but detailed caption."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": caption_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS_CAPTION,
    }

    if eval_mode:
        gen_kwargs.update(
            dict(
                do_sample=False,
                temperature=0.0,
            )
        )
    else:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=CAPTION_TEMPERATURE,
                top_p=TOP_P,
            )
        )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids_new = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]
    caption_text = decoded.strip()

    return caption_text, generated_ids, inputs

# ===========================
# Checkpoint and logging helpers
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

        # Deterministic caption for eval
        caption_text, _, _ = generate_caption(vlm_policy, vlm_processor, img, eval_mode=True)
        reward = compute_reward(llm_model, llm_tokenizer, caption_text, question, correct_label)
        total_reward += reward

    mean_reward = total_reward / float(n)
    return mean_reward

# ===========================
# Plotting metrics
# ===========================
def plot_metrics():
    # Training log
    if not os.path.exists(LOG_PATH):
        logger.warning("No training log found, skipping plots.")
        return

    steps = []
    train_rewards = []
    train_mean_rewards = []
    nll_losses = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps.append(r["step"])
            train_rewards.append(r["reward"])
            train_mean_rewards.append(r["mean_reward"])
            nll_losses.append(r["nll_loss"])

    # Eval log
    eval_steps = []
    eval_mean_rewards = []
    if os.path.exists(EVAL_LOG_PATH):
        with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eval_steps.append(r["step"])
                eval_mean_rewards.append(r["eval_mean_reward"])

    # Plot training mean reward
    if steps:
        plt.figure()
        plt.plot(steps, train_mean_rewards)
        plt.xlabel("Step")
        plt.ylabel("Training Mean Reward")
        plt.title("Training Mean Reward vs Step")
        plt.grid(True)
        train_plot_path = os.path.join(OUTPUT_DIR, "train_mean_reward.png")
        plt.savefig(train_plot_path)
        plt.close()
        logger.info(f"Saved training mean reward plot to {train_plot_path}")

    # Plot eval mean reward curve
    if eval_steps:
        plt.figure()
        plt.plot(eval_steps, eval_mean_rewards)
        plt.xlabel("Step")
        plt.ylabel("Eval Mean Reward")
        plt.title("Eval Mean Reward vs Step")
        plt.grid(True)
        eval_plot_path = os.path.join(OUTPUT_DIR, "eval_mean_reward.png")
        plt.savefig(eval_plot_path)
        plt.close()
        logger.info(f"Saved eval mean reward plot to {eval_plot_path}")

    # Plot NLL loss curve
    if steps:
        plt.figure()
        plt.plot(steps, nll_losses)
        plt.xlabel("Step")
        plt.ylabel("NLL Loss")
        plt.title("Training NLL Loss vs Step")
        plt.grid(True)
        nll_plot_path = os.path.join(OUTPUT_DIR, "train_nll_loss.png")
        plt.savefig(nll_plot_path)
        plt.close()
        logger.info(f"Saved NLL loss plot to {nll_plot_path}")

# ===========================
# Main RL loop
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
    baseline = 0.0
    reward_history = deque(maxlen=REWARD_WINDOW)
    best_mean_reward = -1.0

    data_iter = iter(dataloader)

    try:
        while step < RL_STEPS:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            img_rel = batch["image_path"][0]
            question = batch["question"][0]
            correct_label = batch["correct_label"][0]

            try:
                img = load_image(img_rel)
            except Exception as e:
                logger.warning(f"Skipping {img_rel}: {e}")
                continue

            # 1) Sample caption from policy (exploration mode)
            vlm_policy.eval()
            caption_text, gen_ids, inputs = generate_caption(vlm_policy, vlm_processor, img, eval_mode=False)

            # 2) Compute reward via LLM
            reward = compute_reward(llm_model, llm_tokenizer, caption_text, question, correct_label)

            # 3) Policy gradient update (REINFORCE with baseline)
            vlm_policy.train()

            labels = gen_ids.clone()
            prompt_len = inputs["input_ids"].shape[1]
            labels[:, :prompt_len] = -100  # ignore prompt tokens

            outputs = vlm_policy(
                input_ids=gen_ids,
                attention_mask=torch.ones_like(gen_ids),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                labels=labels,
            )
            nll_loss = outputs.loss  # scalar NLL

            # Update baseline and compute advantage
            baseline = BASELINE_MOMENTUM * baseline + (1.0 - BASELINE_MOMENTUM) * reward
            advantage = reward - baseline

            # REINFORCE loss: advantage * NLL (NLL = -log pi)
            reinforce_loss = advantage * nll_loss

            optimizer.zero_grad()
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm_policy.parameters(), 1.0)
            optimizer.step()

            step += 1

            reward_history.append(reward)
            mean_reward = sum(reward_history) / len(reward_history)

            if step % 50 == 0:
                logger.info(
                    f"Step {step} | Reward={reward:.2f} | MeanReward={mean_reward:.3f} | "
                    f"NLL={nll_loss.item():.4f} | Advantage={advantage:.3f}"
                )
                logger.info(f"  Q: {question[:80]}...")
                logger.info(f"  Caption: {caption_text}")

            # Best checkpoint by moving mean reward
            if len(reward_history) == REWARD_WINDOW and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                extra_state = {
                    "step": step,
                    "best_mean_reward": best_mean_reward,
                    "baseline": baseline,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(
                    vlm_policy,
                    vlm_processor,
                    step,
                    CHECKPOINT_BEST,
                    tag="BEST",
                    extra_state=extra_state,
                )

            # Periodic checkpoint
            if step % CHECKPOINT_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-step-{step}")
                extra_state = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "baseline": baseline,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(
                    vlm_policy,
                    vlm_processor,
                    step,
                    ckpt_dir,
                    tag="PERIODIC",
                    extra_state=extra_state,
                )

            # Log per-step stats
            log_record = {
                "step": step,
                "reward": reward,
                "mean_reward": mean_reward,
                "baseline": baseline,
                "nll_loss": float(nll_loss.item()),
                "advantage": float(advantage),
            }
            append_log(log_record, LOG_PATH)

            # Periodic evaluation
            if step % EVAL_EVERY == 0:
                logger.info(f"Running eval at step {step}...")
                eval_mean_reward = evaluate_policy(
                    vlm_policy, vlm_processor, llm_model, llm_tokenizer, dataset, EVAL_SAMPLES
                )
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
            "baseline": baseline,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "KeyboardInterrupt",
        }
        save_checkpoint(
            vlm_policy,
            vlm_processor,
            step,
            interrupt_dir,
            tag="INTERRUPT",
            extra_state=extra_state,
        )

    # Final save
    logger.info("Saving final model...")
    final_state = {
        "step": step,
        "baseline": baseline,
        "timestamp": datetime.utcnow().isoformat(),
        "best_mean_reward": best_mean_reward,
    }
    save_checkpoint(
        vlm_policy,
        vlm_processor,
        step,
        OUTPUT_DIR,
        tag="FINAL",
        extra_state=final_state,
    )

    # Plot curves at the end
    logger.info("Plotting training and eval curves...")
    plot_metrics()
    logger.info("Done.")

if __name__ == "__main__":
    main()
