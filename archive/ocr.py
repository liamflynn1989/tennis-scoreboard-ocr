import os, json, argparse
import torch
from PIL import Image
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration

GEMMA_ID = "google/gemma-3-4b-it"
SMOL_ID  = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

def pick_device_and_dtype(model_id: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # MPS: safest is float32 (Gemma 3 often blanks in fp16; MPS doesn't do bf16 well)
    # CUDA: bf16 is usually best
    if device == "mps":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return device, dtype

def load_model_and_processor(model_id: str, device: str, dtype):
    processor = AutoProcessor.from_pretrained(model_id)

    if "gemma-3" in model_id:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": device},
        ).eval()
    else:
        # SmolVLM2
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": device},
        ).eval()

    return processor, model

def build_messages(imgs, y_examples, prompt_text):
    # Put the *actual PIL images* into the chat content so apply_chat_template can tokenize everything.
    # (If a processor ever complains, replace {"image": img} with {"path": "..."} instead.)
    content = [{"type": "text", "text": prompt_text + "\n\n---\n\nHere are labeled examples. Learn the format exactly.\n\n"}]

    for i, (img, y) in enumerate(zip(imgs[:-1], y_examples), start=1):
        content += [
            {"type": "text", "text": f"Example {i} image:\n"},
            {"type": "image", "image": img},
            {"type": "text", "text": "Example {i} correct JSON:\n" + json.dumps(y, indent=2) + "\n\n---\n\n"},
        ]

    content += [
        {"type": "text", "text": "Now read the next image and output ONLY JSON for it.\n"},
        {"type": "image", "image": imgs[-1]},
        {"type": "text", "text": "Return ONLY JSON:\n"},
    ]

    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a vision OCR system for tennis scoreboards. Output JSON only."}]},
        {"role": "user", "content": content},
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "smolvlm"], default="smolvlm")
    args = parser.parse_args()

    model_id = GEMMA_ID if args.model == "gemma" else SMOL_ID

    IMG1 = Image.open("/Users/liamflynn/tennis/debug/frame_0322500_t12900.00_crop.png").convert("RGB")
    IMG2 = Image.open("/Users/liamflynn/tennis/debug/frame_0090000_t3600.00_crop.png").convert("RGB")
    IMG3 = Image.open("/Users/liamflynn/tennis/debug/frame_0060000_t2400.00_crop.png").convert("RGB")
    IMG4 = Image.open("/Users/liamflynn/tennis/debug/frame_0135000_t5400.00_crop.png").convert("RGB")

    Y1 = {"player_1": {"sets": [4, 6, 6], "games": 5, "points": 40}, "player_2": {"sets": [6, 4, 3], "games": 4, "points": 30}}
    Y2 = {"player_1": {"sets": [4], "games": 1, "points": 30}, "player_2": {"sets": [6], "games": 0, "points": 40}}
    Y3 = {"player_1": {"sets": [], "games": 4, "points": 15}, "player_2": {"sets": [], "games": 4, "points": 30}}

    PROMPT_TEXT = "Read the tennis scoreboard and output ONLY valid JSON following the schema in the examples."

    device, dtype = pick_device_and_dtype(model_id)
    processor, model = load_model_and_processor(model_id, device, dtype)

    messages = build_messages(
        imgs=[IMG1, IMG2, IMG3, IMG4],
        y_examples=[Y1, Y2, Y3],
        prompt_text=PROMPT_TEXT,
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # keep floats in desired dtype (esp. important on MPS)
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False):
            inputs[k] = v.to(dtype)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            min_new_tokens=20,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    # decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[-1]
    text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    print(text)

if __name__ == "__main__":
    main()