import json
import torch
from PIL import Image
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"

IMG1 = Image.open("/Users/liamflynn/tennis/debug/frame_0322500_t12900.00_crop.png").convert("RGB")
IMG2 = Image.open("/Users/liamflynn/tennis/debug/frame_0090000_t3600.00_crop.png").convert("RGB")
IMG3 = Image.open("/Users/liamflynn/tennis/debug/frame_0060000_t2400.00_crop.png").convert("RGB")
IMG4 = Image.open("/Users/liamflynn/tennis/debug/frame_0135000_t5400.00_crop.png").convert("RGB")

Y1 = {"player_1": {"sets": [4, 6, 6], "games": 5, "points": 40}, "player_2": {"sets": [6, 4, 3], "games": 4, "points": 30}}
Y2 = {"player_1": {"sets": [4], "games": 1, "points": 30}, "player_2": {"sets": [6], "games": 0, "points": 40}}
Y3 = {"player_1": {"sets": [], "games": 4, "points": 15}, "player_2": {"sets": [], "games": 4, "points": 30}}

PROMPT_TEXT = """
Read the tennis scoreboard and output ONLY valid JSON following the schema in the examples.
""".strip()

device = "mps" if torch.backends.mps.is_available() else "cpu"
# Gemma 3 is known to go blank in float16; use float32 on MPS.
dtype = torch.float32 if device == "mps" else torch.bfloat16

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},   # keep it simple
).eval()

# IMPORTANT: Use structured "content" with {"type":"image"} so Gemma inserts image tokens.
import json

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a vision OCR system for tennis scoreboards. Output JSON only."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT_TEXT + "\n\n---\n\nHere the labeled examples. Learn the format exactly.\n\n"},

            {"type": "text", "text": "Example 1 image:\n"},
            {"type": "image"},
            {"type": "text", "text": "Example 1 correct JSON:\n" + json.dumps(Y1, indent=2) + "\n\n---\n\n"},

            {"type": "text", "text": "Example 2 image:\n"},
            {"type": "image"},
            {"type": "text", "text": "Example 2 correct JSON:\n" + json.dumps(Y2, indent=2) + "\n\n---\n\n"},

            {"type": "text", "text": "Example 3 image:\n"},
            {"type": "image"},
            {"type": "text", "text": "Example 3 correct JSON:\n" + json.dumps(Y3, indent=2) + "\n\n---\n\n"},

            {"type": "text", "text": "Now read the next image and output ONLY JSON for it.\n"},
            {"type": "image"},
            {"type": "text", "text": "Example 4 correct JSON:\n"},
        ],
    },
]
# This will now include 3 image tokens.
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print(prompt)

inputs = processor(
    text=prompt,
    images=[IMG1, IMG2, IMG3, IMG4],   # must match the 3 {"type":"image"} items above
    return_tensors="pt",
).to(device)

# IMPORTANT on MPS: keep compute in float32 (no autocast to fp16)
for k, v in list(inputs.items()):
    if hasattr(v, "dtype") and v.dtype.is_floating_point:
        inputs[k] = v.to(dtype)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        min_new_tokens=20,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

prompt_len = inputs["input_ids"].shape[-1]
text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
print(text)