import os
import json
import torch
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from qwen_vl_utils import process_vision_info
import logging
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = "caprl_mcq_dataset_final.jsonl"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "output/grpo_qwen2vl"
VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Global Reward Model (loaded once)
reward_model = None
reward_tokenizer = None

def load_reward_model():
    global reward_model, reward_tokenizer
    if reward_model is None:
        logger.info("Loading Reward Model (LLM)...")
        reward_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda:0", # Explicitly put on GPU
            attn_implementation="sdpa"
        ).eval() # Set to eval mode
        reward_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        reward_tokenizer.padding_side = "left" # Important for batch generation

def reward_function(prompts, completions, **kwargs):
    """
    Computes the reward for a batch of completions (captions).
    kwargs contains the columns from the dataset (question, correct_label, etc.)
    """
    global reward_model, reward_tokenizer
    if reward_model is None:
        load_reward_model()
        
    questions = kwargs.get("question")
    correct_labels = kwargs.get("correct_label")
    
    rewards = []
    
    # Prepare batch for LLM
    batch_prompts = []
    
    for caption, question in zip(completions, questions):
        prompt = f"""Based on the following description of an image, answer the multiple-choice question.
        
Description: {caption}

Question: {question}

Answer with the option letter only (A, B, C, or D).
Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided descriptions."},
            {"role": "user", "content": prompt}
        ]
        text = reward_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_prompts.append(text)
        
    # Tokenize and generate
    inputs = reward_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(reward_model.device)
    
    with torch.no_grad():
        generated_ids = reward_model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1 # Low temp for deterministic evaluation
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = reward_tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    for response, correct_label in zip(responses, correct_labels):
        predicted_label = "Unknown"
        clean_response = response.strip().upper()
        
        # Robust parsing using Regex
        match = re.search(r'\b([A-D])\b', clean_response)
        if match:
            predicted_label = match.group(1)
        else:
            # Fallback heuristics
            if "A)" in clean_response or "A." in clean_response: predicted_label = "A"
            elif "B)" in clean_response or "B." in clean_response: predicted_label = "B"
            elif "C)" in clean_response or "C." in clean_response: predicted_label = "C"
            elif "D)" in clean_response or "D." in clean_response: predicted_label = "D"
             
        # Reward: 1.0 for correct, 0.0 for incorrect
        if predicted_label == correct_label:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def prepare_dataset():
    data = []
    print("Loading dataset metadata...")
    with open(DATASET_PATH, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Parsing items"):
        try:
            item = json.loads(line)
            # Add local image path
            filename = item['image_id'].split("train2017/")[-1]
            local_path = os.path.join(IMAGE_DIR, filename)
            if os.path.exists(local_path):
                item['image_path_local'] = local_path
                # We don't process images here to save RAM
                data.append(item)
        except json.JSONDecodeError:
            pass
    return Dataset.from_list(data)

def get_transform(processor):
    def transform(batch):
        # batch is a dict of lists: {'image_path_local': [...], 'question': [...], ...}
        
        texts = []
        images = []
        
        for image_path in batch['image_path_local']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            texts.append(text)
            images.append(image_inputs)
            
        inputs = processor(
            text=texts,
            images=images,
            padding="max_length",
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        
        # Return dict of lists/tensors
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'prompt': texts, # GRPOTrainer needs this
            'question': batch['question'],
            'correct_label': batch['correct_label']
        }
    return transform

def main():
    # Load Processor
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)

    # Load Dataset (Lightweight)
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Set Transform (Lazy Processing)
    dataset.set_transform(get_transform(processor))
    
    # Training Config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,
        per_device_train_batch_size=16, # H100 Optimized
        gradient_accumulation_steps=1,
        num_generations=8, # Group size
        max_completion_length=512, # Increased to avoid clipping
        num_train_epochs=1,
        dataloader_num_workers=4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="tensorboard",
        logging_dir="output/logs",
        remove_unused_columns=False, # Essential for passing 'question', 'correct_label'
        temperature=0.9, # Must be > 0 for GRPO to work
    )
    
    # Load Model (Actor)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto" # Let accelerate handle placement
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Trainer
    # We pass processing_class=processor so it uses the default collator which should handle the tensors
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    
    print("Starting GRPO Training...")
    trainer.train(resume_from_checkpoint=True)
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
