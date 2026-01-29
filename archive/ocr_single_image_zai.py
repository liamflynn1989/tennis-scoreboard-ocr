from transformers import AutoProcessor, Glm4vForConditionalGeneration
from PIL import Image
import torch

MODEL_PATH = "zai-org/GLM-4.6V-Flash"

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
                "image": "/Users/liamflynn/tennis/test/t7140.00.png",
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]


processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Glm4vForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
).eval()

# rope workaround
if getattr(model.config, "rope_scaling", None) is None:
    model.config.rope_scaling = {"mrope_section": [16, 24, 24]}
for m in model.modules():
    if hasattr(m, "rope_scaling") and getattr(m, "rope_scaling") is None:
        m.rope_scaling = model.config.rope_scaling

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)
inputs.pop("token_type_ids", None)


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)