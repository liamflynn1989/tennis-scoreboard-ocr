# from transformers import AutoModel, AutoProcessor
# from chandra.model.hf import generate_hf
# from chandra.model.schema import BatchInputItem
# from chandra.output import parse_markdown
# from PIL import Image

# model = AutoModel.from_pretrained("datalab-to/chandra")
# model.to("mps")
# model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")

# img_path = "/Users/liamflynn/tennis/test/t12900.00.png"

# # ---- load + crop to bottom-left quarter ----
# img = Image.open(img_path).convert("RGB")
# W, H = img.size
# img_crop = img.crop((0, H // 2, W // 2, H))  # (left, upper, right, lower)

# batch = [
#     BatchInputItem(
#         image=img_crop,
#         prompt_type="ocr_layout"
#     )
# ]

# result = generate_hf(batch, model)[0]
# markdown = parse_markdown(result.raw)

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
from PIL import Image

model_id = "datalab-to/chandra"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the *generative* model (this is the key fix vs AutoModel / AutoModelForCausalLM)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float32,   # <-- key change
).to("mps").eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model.processor = processor

# --- image load + crop ---
img_path = "/Users/liamflynn/tennis/test/t12900.00.png"
img = Image.open(img_path).convert("RGB")
W, H = img.size
img_crop = img.crop((0, H // 1.5, W // 2.5, H))

batch = [BatchInputItem(image=img_crop, prompt_type="ocr")]

# IMPORTANT: patch chandra/model/hf.py so it uses model.device (shown below)
with torch.inference_mode():
    result = generate_hf(batch, model)[0]

markdown = parse_markdown(result.raw)
print(result.raw)
print("---")
print(markdown)
img_crop.show()