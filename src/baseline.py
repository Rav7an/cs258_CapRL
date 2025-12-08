import json
import os
import torch
import gc
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Configuration
DATASET_PATH = "caprl_mcq_dataset_final.jsonl"
IMAGE_DIR = "data/images"
OUTPUT_FILE = "baseline_results.jsonl"
PLOT_DIR = "plots/baseline"
VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

os.makedirs(PLOT_DIR, exist_ok=True)

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def generate_captions(data):
    print(f"Loading VL Model: {VL_MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)
    
    captions = {}
    # We only need to caption each unique image once
    unique_images = {item['image_id']: item['image_path'] for item in data}
    
    print(f"Generating captions for {len(unique_images)} unique images...")
    
    for image_id, image_path_raw in tqdm(unique_images.items()):
        filename = image_path_raw.split("train2017/")[-1]
        local_image_path = os.path.join(IMAGE_DIR, filename)
        
        if not os.path.exists(local_image_path):
            captions[image_id] = "Image not found."
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": local_image_path},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        captions[image_id] = output_text[0]
        
    # Cleanup
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return captions

def evaluate_with_llm(data, captions):
    print(f"Loading LLM Model: {LLM_MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    
    results = []
    
    print("Evaluating questions...")
    for item in tqdm(data):
        image_id = item['image_id']
        caption = captions.get(image_id, "")
        question = item['question']
        correct_label = item['correct_label']
        
        prompt = f"""Based on the following description of an image, answer the multiple-choice question.
        
Description: {caption}

Question: {question}

Answer with the option letter only (A, B, C, or D).
Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided descriptions."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10, # We only need a short answer
                temperature=0.1 # Low temperature for deterministic answers
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        # Extract the first letter (A, B, C, D) from the response
        predicted_label = "Unknown"
        for char in response:
            if char.upper() in ['A', 'B', 'C', 'D']:
                predicted_label = char.upper()
                break
        
        # Fallback if the model outputs something like "The answer is A"
        if predicted_label == "Unknown":
             # Simple heuristic
             if "A)" in response or "A." in response: predicted_label = "A"
             elif "B)" in response or "B." in response: predicted_label = "B"
             elif "C)" in response or "C." in response: predicted_label = "C"
             elif "D)" in response or "D." in response: predicted_label = "D"

        results.append({
            "id": item['id'],
            "image_id": image_id,
            "caption": caption,
            "question": question,
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "raw_response": response,
            "is_correct": predicted_label == correct_label
        })

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def analyze_results(results):
    df = pd.DataFrame(results)
    
    # 1. Overall Accuracy
    accuracy = df['is_correct'].mean()
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # 2. Confusion Matrix
    labels = ['A', 'B', 'C', 'D']
    # Filter out 'Unknown' for confusion matrix or treat it as a separate class
    valid_preds = df[df['predicted_label'].isin(labels)]
    cm = confusion_matrix(valid_preds['correct_label'], valid_preds['predicted_label'], labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Accuracy by Question Type (Heuristic: First few words of question)
    # This is a bit rough, but gives some insight
    df['q_start'] = df['question'].apply(lambda x: " ".join(x.split()[:3]))
    acc_by_type = df.groupby('q_start')['is_correct'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    acc_by_type.plot(kind='bar')
    plt.title('Top 10 Question Types by Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'accuracy_by_type.png'))
    plt.close()

    # Save results to file
    with open(OUTPUT_FILE, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Results saved to {OUTPUT_FILE}")

def main():
    data = load_dataset(DATASET_PATH)
    print(f"Loaded {len(data)} questions.")
    
    # Step 1: Generate Captions (Actor)
    captions = generate_captions(data)
    
    # Step 2: Evaluate (Reward Model)
    results = evaluate_with_llm(data, captions)
    
    # Step 3: Analyze
    analyze_results(results)

if __name__ == "__main__":
    main()
