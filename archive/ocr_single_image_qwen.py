from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
# model_id = "Qwen/Qwen3-VL-32B-Instruct"

img_path = "/Users/liamflynn/tennis/test/t12900.00.png"

# ---- load + crop to bottom-left quarter ----
img = Image.open(img_path).convert("RGB")
W, H = img.size
img_crop = img.crop((0, H // 2, W // 2, H))  # (left, upper, right, lower)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    # device_map=None,          # <-- avoid auto
    # low_cpu_mem_usage=True,
)

# model.to("mps")



processor = AutoProcessor.from_pretrained(model_id)

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
                "image": img_crop
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
img_crop.show()