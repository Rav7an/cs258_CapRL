import os
import json
import torch
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info

# ===========================
# Configuration
# ===========================
DATASET_PATH = "caprl_mcq_dataset_final.jsonl"
IMAGE_DIR = "data/images"  # Path to the images folder
OUTPUT_REPORT_JSON = "evaluation_report_grpo.json"
OUTPUT_REPORT_MD = "analysis_summary_grpo.md"

# Models to evaluate
MODELS_TO_EVALUATE = {
    "GRPO_BEST": "output/grpo_qwen2vl_3/best",
    "GRPO-STEP-7500": "output/grpo_qwen2vl_3/checkpoint-step-7500",
    "GRPO-STEP-14500": "output/grpo_qwen2vl_3/checkpoint-step-14500",
    "GRPO-STEP-15000": "output/grpo_qwen2vl_3/checkpoint-step-15000",
    # "Baseline": "Qwen/Qwen2-VL-2B-Instruct" # Optional: Add baseline if needed
}

LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS_CAPTION = 256
MAX_NEW_TOKENS_ANSWER = 32
EVAL_SAMPLES = 400  # Number of samples to evaluate per model (set to None for full dataset)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Handle both full paths and relative paths
    if os.path.exists(img_path):
        full_path = img_path
    else:
        # Try joining with IMAGE_DIR
        full_path = os.path.join(IMAGE_DIR, img_path)
        if not os.path.exists(full_path):
            # Try flat structure (just filename)
            filename = os.path.basename(img_path)
            full_path = os.path.join(IMAGE_DIR, filename)
            
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
        
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
# Model Loading & Inference
# ===========================
def load_vlm_model(model_path: str):
    logger.info(f"Loading VLM model from {model_path}...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(model_path) # Load processor from same path usually
    except Exception as e:
        logger.warning(f"Could not load from {model_path}, trying base model config for processor or fallback: {e}")
        # Fallback for processor if not saved in checkpoint, or if model_path is just weights
        # Assuming base model ID for processor if local load fails for it
        base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(base_model_id)
        if "model" not in locals(): # If model failed to load
             model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                attn_implementation="sdpa"
            )
            
    model.eval()
    return model, processor

def load_reward_model():
    logger.info(f"Loading Reward LLM: {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa"
    )
    model.eval()
    return model, tokenizer

def generate_caption(model, processor, img: Image.Image):
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
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CAPTION,
            do_sample=False, # Deterministic for evaluation
            temperature=0.0,
        )
        
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids_new = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]
    return decoded.strip()

def predict_answer(llm_model, llm_tokenizer, caption: str, question: str):
    prompt = build_answer_prompt(caption, question)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    
    with torch.no_grad():
        gen_ids = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_ANSWER,
            do_sample=False,
            temperature=0.0,
            pad_token_id=llm_tokenizer.eos_token_id
        )
        
    gen_only = gen_ids[0, inputs['input_ids'].shape[1]:]
    out_text = llm_tokenizer.decode(gen_only, skip_special_tokens=True)
    return normalize_output_to_label(out_text)

# ===========================
# Main Evaluation Loop
# ===========================
def main():
    # 1. Load Dataset
    logger.info("Loading dataset...")
    items = read_jsonl(DATASET_PATH)
    if EVAL_SAMPLES:
        items = items[:EVAL_SAMPLES]
    logger.info(f"Evaluating on {len(items)} samples.")
    
    # 2. Load Reward Model (Shared)
    llm_model, llm_tokenizer = load_reward_model()
    
    results = {}
    detailed_samples = [] # Store details for a subset of samples
    
    # Initialize detailed samples structure
    for i, item in enumerate(items[:20]): # Keep details for first 20
        detailed_samples.append({
            "id": i,
            "image_path": item.get('image_path') or item.get('image_id'),
            "question": item['question'],
            "correct_label": item['correct_label'].strip().upper(),
            "model_outputs": {}
        })

    # 3. Evaluate each VLM
    for model_name, model_path in MODELS_TO_EVALUATE.items():
        logger.info(f"--- Evaluating {model_name} ---")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist. Skipping.")
            results[model_name] = {"accuracy": 0.0, "error": "Path not found"}
            continue
            
        # Load VLM
        vlm_model, vlm_processor = load_vlm_model(model_path)
        
        correct_count = 0
        total_count = 0
        caption_lengths = []
        
        for i, item in tqdm(enumerate(items), total=len(items), desc=f"Eval {model_name}"):
            img_rel = item.get('image_path') or item.get('image_id')
            question = item['question']
            correct_label = item['correct_label'].strip().upper()
            
            try:
                img = load_image(img_rel)
            except Exception as e:
                logger.error(f"Error loading image {img_rel}: {e}")
                continue
                
            # Generate Caption
            caption = generate_caption(vlm_model, vlm_processor, img)
            
            # Calculate length (word count)
            cap_len = len(caption.split())
            caption_lengths.append(cap_len)
            
            # Predict Answer
            pred_label = predict_answer(llm_model, llm_tokenizer, caption, question)
            
            # Check Correctness
            is_correct = (pred_label == correct_label)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store details for the report subset
            if i < 20:
                detailed_samples[i]["model_outputs"][model_name] = {
                    "caption": caption,
                    "caption_length": cap_len,
                    "predicted_label": pred_label,
                    "is_correct": is_correct
                }
        
        # Calculate Metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_len = sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0.0
        
        results[model_name] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
            "avg_caption_length": avg_len
        }
        logger.info(f"{model_name} Accuracy: {accuracy:.2%}, Avg Len: {avg_len:.1f}")
        
        # Free up VRAM
        del vlm_model
        del vlm_processor
        torch.cuda.empty_cache()

    # 4. Save Report
    report = {
        "summary": results,
        "detailed_samples": detailed_samples
    }
    
    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved detailed JSON report to {OUTPUT_REPORT_JSON}")
    
    # 5. Generate Markdown Summary
    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# RL-VLM Evaluation Report\n\n")
        f.write("## Summary of Accuracy\n")
        f.write("| Model | Accuracy | Correct / Total | Avg Caption Length |\n")
        f.write("|-------|----------|-----------------|--------------------|\n")
        for name, metrics in results.items():
            if "error" in metrics:
                f.write(f"| {name} | N/A | {metrics['error']} | N/A |\n")
            else:
                f.write(f"| {name} | {metrics['accuracy']:.2%} | {metrics['correct']} / {metrics['total']} | {metrics['avg_caption_length']:.1f} words |\n")
        
        f.write("\n## Qualitative Analysis (First 5 Samples)\n")
        for sample in detailed_samples[:50]:
            f.write(f"### Sample {sample['id']}\n")
            f.write(f"**Image:** `{sample['image_path']}`\n\n")
            f.write(f"**Question:** {sample['question']}\n\n")
            f.write(f"**Correct Answer:** {sample['correct_label']}\n\n")
            
            for model_name in MODELS_TO_EVALUATE.keys():
                if model_name in sample["model_outputs"]:
                    out = sample["model_outputs"][model_name]
                    status = "✅" if out["is_correct"] else "❌"
                    f.write(f"**{model_name}** ({status} Pred: {out['predicted_label']}):\n")
                    f.write(f"> {out['caption']}\n\n")
            f.write("---\n")
            
    logger.info(f"Saved Markdown summary to {OUTPUT_REPORT_MD}")

if __name__ == "__main__":
    main()
