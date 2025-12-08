import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from qwen_vl_utils import process_vision_info
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable TF32 for speed
torch.set_float32_matmul_precision('high')

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
            temperature=0.1
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = reward_tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    for response, correct_label in zip(responses, correct_labels):
        predicted_label = "Unknown"
        clean_response = response.strip().upper()
        
        # Simple parsing
        for char in clean_response:
            if char in ['A', 'B', 'C', 'D']:
                predicted_label = char
                break
        
        # Fallback heuristics
        if predicted_label == "Unknown":
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
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Add local image path
                filename = item['image_id'].split("train2017/")[-1]
                local_path = os.path.join(IMAGE_DIR, filename)
                if os.path.exists(local_path):
                    item['image_path_local'] = local_path
                    
                    # Format prompt for Qwen2-VL
                    # We need to provide the messages structure that the model expects
                    # But GRPOTrainer expects a text prompt usually? 
                    # Wait, for VLMs in TRL, we usually pass the formatted inputs.
                    # Let's try passing the raw messages and let the collator handle it if possible.
                    # Or we pre-format using the processor.
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": local_path},
                                {"type": "text", "text": "Describe this image in detail."},
                            ],
                        }
                    ]
                    item['prompt'] = messages
                    data.append(item)
            except json.JSONDecodeError:
                pass
    return Dataset.from_list(data)

def main():
    # Load Dataset
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)
    
    # Monkey patch apply_chat_template to handle batching of messages with images
    original_apply_chat_template = processor.apply_chat_template
    
    def patched_apply_chat_template(*args, **kwargs):
        # Debugging
        # print(f"DEBUG: args={len(args)}, kwargs={kwargs.keys()}")
        
        if len(args) > 0:
            # Check if first arg is processor instance (self)
            if hasattr(args[0], 'tokenizer') and hasattr(args[0], 'apply_chat_template'):
                conversations = args[1] if len(args) > 1 else kwargs.get('conversations')
            else:
                conversations = args[0]
        else:
            conversations = kwargs.get('conversations')
            
        if conversations is None:
             # Fallback or error
             # Maybe it was passed as 'conversation'?
             conversations = kwargs.get('conversation')
             
        texts = []
        images = []
        videos = []
        
        for conv in conversations:
            # Clean conv: remove None values from content dicts (caused by Arrow/Dataset schema)
            cleaned_conv = []
            for msg in conv:
                new_content = []
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        new_item = {k: v for k, v in item.items() if v is not None}
                        new_content.append(new_item)
                else:
                    new_content = msg['content']
                
                cleaned_conv.append({'role': msg['role'], 'content': new_content})
            
            conv = cleaned_conv

            try:
                text = original_apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(conv)
                texts.append(text)
                images.append(image_inputs)
                videos.append(video_inputs)
            except Exception as e:
                print(f"Error processing conversation: {conv}")
                print(f"Error: {e}")
                raise e
        
        # Check if we have any images or videos
        has_images = any(img is not None for img in images)
        has_videos = any(vid is not None for vid in videos)
        
        inputs = processor(
            text=texts,
            images=images if has_images else None,
            videos=videos if has_videos else None,
            padding=True,
            return_tensors="pt",
        )
        return inputs
        
    # processor.apply_chat_template = types.MethodType(patched_apply_chat_template, processor)
    processor.apply_chat_template = patched_apply_chat_template
    
    # Training Config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,
        per_device_train_batch_size=32, # Reduced to 32 to avoid OOM
        gradient_accumulation_steps=2, # Accumulate to reach effective batch size
        num_generations=8, 
        max_completion_length=256,
        num_train_epochs=1,
        dataloader_num_workers=4,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Load Model (Actor)
    # We use Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting GRPO Training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
