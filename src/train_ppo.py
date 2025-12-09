import os
import json
import torch
import re
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
from collections import deque
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = "caprl_mcq_dataset_final.jsonl"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "output/ppo_qwen2vl_2_0"
VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16  # Effective batch size = 16
RL_STEPS = 150000          # You can set 200_000 if you want

CLIP_EPS = 0.2
LR = 1e-6
MAX_NEW_TOKENS_CAPTION = 256
MAX_NEW_TOKENS_ANSWER = 32
CAPTION_TEMPERATURE = 0.9 # Exploration
ANSWER_TEMPERATURE = 0.0 # Deterministic Reward
TOP_P = 0.9

REWARD_WINDOW = 200         # Moving average window
CHECKPOINT_EVERY = 200      # Periodic checkpointing
EVAL_EVERY = 500           # Run eval every N steps
EVAL_SAMPLES = 26           # Number of samples per eval rollout

device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
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
    with open(path, 'r', encoding='utf-8') as f:
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
    m = re.search(r'\b([ABCD])\b', t)
    return m.group(1) if m else ''

def load_image(img_path: str) -> Image.Image:
    # Handle both full paths and relative paths if necessary
    # The dataset usually has relative paths like 'coco/train2017/...'
    # We assume IMAGE_DIR is the root for these.
    
    # Check if it's already a full path or needs joining
    if os.path.exists(img_path):
        full_path = img_path
    else:
        # Try joining with IMAGE_DIR
        # Note: The dataset might have 'coco/train2017/xxx.jpg'
        # But on disk it might be 'data/images/xxx.jpg' (flat structure) or 'data/images/coco/train2017/...'
        # Based on previous context, it seems images are in data/images/
        
        # Try direct join
        full_path = os.path.join(IMAGE_DIR, img_path)
        if not os.path.exists(full_path):
            # Try flat structure (just filename)
            filename = os.path.basename(img_path)
            full_path = os.path.join(IMAGE_DIR, filename)
            
    return Image.open(full_path).convert('RGB')

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
        img_rel = it.get('image_path') or it.get('image_id')
        question = it['question']
        correct_label = it['correct_label'].strip().upper()
        return {
            'image_path': img_rel,
            'question': question,
            'correct_label': correct_label,
        }

# ===========================
# Models
# ===========================
def load_models():
    print('Loading VLM policy...')
    vlm_policy = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa"
    )
    vlm_policy.train()
    
    # Processor
    vlm_processor = AutoProcessor.from_pretrained(VL_MODEL_ID)
    
    print('Loading VLM ref (frozen)...')
    vlm_ref = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa"
    )
    vlm_ref.eval()
    for p in vlm_ref.parameters():
        p.requires_grad = False
        
    print('Loading Reward LLM...')
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa"
    )
    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False
        
    return vlm_policy, vlm_ref, vlm_processor, llm_model, llm_tokenizer

# ===========================
# Core Logic
# ===========================
def generate_caption_and_logprobs(model, processor, img: Image.Image, for_policy_training: bool):
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
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CAPTION,
            do_sample=True,
            temperature=CAPTION_TEMPERATURE,
            top_p=TOP_P,
        )
        
    # Decode
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids_new = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]
    caption_text = decoded.strip()
    
    # Compute Logprobs
    # We need to run forward pass on the generated sequence
    if for_policy_training:
        outputs = model(
            input_ids=generated_ids,
            attention_mask=torch.ones_like(generated_ids),
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs['image_grid_thw']
        )
    else:
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                pixel_values=inputs['pixel_values'],
                image_grid_thw=inputs['image_grid_thw']
            )
            
    logits = outputs.logits # [1, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Gather logprobs for generated tokens
    # generated_ids contains [prompt, generated]
    # logits[t] predicts generated_ids[t+1]
    
    # We want logprobs for tokens starting from prompt_len
    # prompt_len is defined above
    
    # Shift logits and labels
    # logits[:, :-1] predicts generated_ids[:, 1:]
    
    # We are interested in generated_ids[:, prompt_len:]
    # These correspond to logits indices [prompt_len-1, ..., -2]
    
    # Let's do it carefully
    # Full sequence logprobs
    log_probs_full = log_probs[:, :-1, :].gather(
        -1, generated_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # Slice for generated tokens
    # The token at index `prompt_len` in generated_ids corresponds to index `prompt_len-1` in log_probs_full
    log_probs_generated = log_probs_full[:, prompt_len-1:]
    
    return caption_text, generated_ids, log_probs_generated

def compute_reward(llm_model, llm_tokenizer, caption: str, question: str, correct_label: str) -> float:
    prompt = build_answer_prompt(caption, question)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    
    with torch.no_grad():
        gen_ids = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_ANSWER,
            do_sample=False,
            temperature=ANSWER_TEMPERATURE,
            pad_token_id=llm_tokenizer.eos_token_id
        )
        
    gen_only = gen_ids[0, inputs['input_ids'].shape[1]:]
    out_text = llm_tokenizer.decode(gen_only, skip_special_tokens=True)
    pred_label = normalize_output_to_label(out_text)
    
    return 1.0 if pred_label == correct_label else 0.0

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

        # Deterministic caption for eval (using generate_caption_and_logprobs with temp=0 implicitly or just greedy)
        # We'll reuse generate_caption_and_logprobs but we need to ensure it can do deterministic generation
        # The current implementation of generate_caption_and_logprobs uses global CAPTION_TEMPERATURE.
        # Let's just use a simplified generation here or modify the function.
        # For simplicity, I'll use the existing function but we should ideally control temp.
        # However, generate_caption_and_logprobs is hardcoded.
        # Let's add a small helper or just use the function and accept it's sampling (which is fine for PPO eval usually, but deterministic is better).
        
        # Let's just call the model directly for eval to be safe and deterministic
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
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm_policy.device)
        
        with torch.no_grad():
            generated_ids = vlm_policy.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_CAPTION,
                do_sample=False, # Deterministic
                temperature=0.0,
            )
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids_new = generated_ids[:, prompt_len:]
        decoded = vlm_processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]
        caption_text = decoded.strip()

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
    losses = []
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
            losses.append(r["loss"])
            if "nll_loss" in r:
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

    # Plot Loss curve
    if steps:
        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Step")
        plt.grid(True)
        loss_plot_path = os.path.join(OUTPUT_DIR, "train_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()
        logger.info(f"Saved loss plot to {loss_plot_path}")

    # Plot NLL Loss curve
    if nll_losses:
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

def main():
    # Load Data
    print("Loading Dataset...")
    items = read_jsonl(DATASET_PATH)
    dataset = ImageMcqDataset(items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load Models
    vlm_policy, vlm_ref, vlm_processor, llm_model, llm_tokenizer = load_models()
    
    # Optimizer
    optimizer = torch.optim.AdamW(vlm_policy.parameters(), lr=LR)
    
    print("Starting PPO Training...")
    step = 0
    reward_history = deque(maxlen=REWARD_WINDOW)
    best_mean_reward = -1.0
    
    data_iter = iter(dataloader)
    pbar = tqdm(total=RL_STEPS, desc="PPO Training")
    
    try:
        while step < RL_STEPS:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            img_rel = batch['image_path'][0]
            question = batch['question'][0]
            correct_label = batch['correct_label'][0]
            
            try:
                img = load_image(img_rel)
            except Exception as e:
                print(f"Skipping {img_rel}: {e}")
                continue
                
            # 1. Policy Rollout
            vlm_policy.train()
            caption_text, gen_ids, log_probs_policy = generate_caption_and_logprobs(
                vlm_policy, vlm_processor, img, for_policy_training=True
            )
            
            # 2. Reference Logprobs
            _, _, log_probs_ref = generate_caption_and_logprobs(
                vlm_ref, vlm_processor, img, for_policy_training=False
            )
            
            # 3. Reward
            reward = compute_reward(llm_model, llm_tokenizer, caption_text, question, correct_label)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            
            # 4. PPO Loss (Contextual Bandit style)
            # Average logprobs over the sequence (or sum?)
            # Usually we take the mean logprob of the trajectory for single-step PPO
            logp = log_probs_policy.mean()
            logp_old = log_probs_ref.mean()
            
            ratio = torch.exp(logp - logp_old)
            advantage = reward_t # No baseline
            
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
            policy_loss = -torch.min(unclipped, clipped)
            
            # KL Penalty
            kl = logp_old - logp
            kl_coef = 0.1
            loss = policy_loss + kl_coef * kl
            
            # 5. Update
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(vlm_policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            pbar.update(1)
            
            reward_history.append(reward)
            mean_reward = sum(reward_history) / len(reward_history)
            
            if step % 50 == 0:
                logger.info(f"Step {step}: Reward={reward}, MeanRew={mean_reward:.2f}, Loss={loss.item():.3f}, Caption='{caption_text}'")
                pbar.set_postfix({"MeanRew": f"{mean_reward:.2f}", "Loss": f"{loss.item():.3f}"})
            
            # Log per-step stats
            log_record = {
                "step": step,
                "reward": reward,
                "mean_reward": mean_reward,
                "loss": float(loss.item()),
                "nll_loss": float(-logp.item()),
                "advantage": float(advantage.item()),
            }
            append_log(log_record, LOG_PATH)

            # Best checkpoint by moving mean reward
            if len(reward_history) == REWARD_WINDOW and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                extra_state = {
                    "step": step,
                    "best_mean_reward": best_mean_reward,
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
        pbar.close()
        logger.warning("KeyboardInterrupt detected! Saving interrupt checkpoint...")
        interrupt_dir = os.path.join(OUTPUT_DIR, f"interrupt-step-{step}")
        extra_state = {
            "step": step,
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

    pbar.close()
    print("Saving final model...")
    final_state = {
        "step": step,
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

