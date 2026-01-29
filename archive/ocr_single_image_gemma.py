# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    # device_map="auto"
    # device_map=None,          # <-- avoid auto
    # low_cpu_mem_usage=True,
).eval()

# model.to("mps")

img_path = "/Users/liamflynn/tennis/test/t7140.00.png"
img_path = "/Users/liamflynn/tennis/test/t12900.00.png"

# ---- load + crop to bottom-left quarter ----
img = Image.open(img_path).convert("RGB")
W, H = img.size
img_crop = img.crop((0, H // 2, W // 2, H))  # (left, upper, right, lower)


processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

prompt_text = (
    "You are reading the tennis scoreboard in the bottom-left of the image.\n\n"
    "You should read the score, one column at a time, from left to right.\n\n"
    "Return the score EXACTLY in this format:\n"
    "SETS:a-b [more set tokens] | GAME:x-y\n\n"
    "Example : SETS:6-4 3-6 5-6 | GAME:40-30\n\n"
    "Rules:\n"
    "- Output exactly ONE line.\n"
    "- After 'SETS:' output 1 to 5 set tokens, each token is 'n-n' with n in 0–7.\n"
    "- Use ONLY the sets that are visible; do not output empty or placeholder commas.\n"
    "- After '| GAME:' output 'x-y' where x and y are 0–99 or AD.\n"
    "- If game score not visible, output GAME:0-0.\n"
    "- If no scoreboard is visible, return exactly: SETS:| GAME:\n"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_crop,
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.float16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
img_crop.show()