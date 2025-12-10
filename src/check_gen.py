import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

class Qwen2VLWithValueHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.v_head = nn.Linear(pretrained_model.config.hidden_size, 1, bias=False)
        
    def forward(self, **kwargs):
        output_hidden_states = kwargs.get("output_hidden_states", True)
        kwargs["output_hidden_states"] = True
        
        outputs = self.pretrained_model(**kwargs)
        
        hidden_states = outputs.hidden_states[-1]
        values = self.v_head(hidden_states).squeeze(-1)
        
        return outputs.logits, values
        
    def generate(self, **kwargs):
        return self.pretrained_model.generate(**kwargs)
        
    @property
    def device(self):
        return self.pretrained_model.device

model_id = "Qwen/Qwen2-VL-2B-Instruct"
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = Qwen2VLWithValueHead(base_model)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.new('RGB', (224, 224), color='black')
text = "Describe this image."

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ],
    }
]
text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Keys:", inputs.keys())

output = model.generate(**inputs, max_new_tokens=10)
print("Output:", output)
