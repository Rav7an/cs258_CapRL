from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

model_id = "Qwen/Qwen2-VL-2B-Instruct"

try:
    # Try to load the model with the Value Head wrapper
    # Note: AutoModelForCausalLMWithValueHead usually expects a model that can be loaded via AutoModelForCausalLM
    # or it wraps a pre-loaded model.
    
    # Let's try wrapping the specific class
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
    print("Successfully loaded with AutoModelForCausalLMWithValueHead.from_pretrained")
except Exception as e:
    print(f"Failed to load with AutoModelForCausalLMWithValueHead.from_pretrained: {e}")

try:
    # Try wrapping the instance
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    print("Successfully wrapped instance")
except Exception as e:
    print(f"Failed to wrap instance: {e}")
