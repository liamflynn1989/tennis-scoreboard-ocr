import os, math, argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from functools import partial

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
torch.mps.empty_cache()

# -------------------------
# Task configuration (edit this per task)
# -------------------------
@dataclass
class TaskConfig:
    prompt_text: str
    # CSV columns
    image_col: str = "image_path"
    label_col: str = "label"
    # Image preprocessing knobs (can be ignored if you override image_transform)
    crop_left_frac: float = 0.0
    crop_right_frac: float = 1.0
    crop_top_frac: float = 0.0
    crop_bottom_frac: float = 1.0
    max_width: int = 640

    def image_transform(self, img: Image.Image) -> Image.Image:
        """Override this method for task-specific cropping/resizing/etc."""
        W, H = img.size
        left = int(W * self.crop_left_frac)
        right = int(W * self.crop_right_frac)
        top = int(H * self.crop_top_frac)
        bottom = int(H * self.crop_bottom_frac)
        img = img.crop((left, top, right, bottom))

        if self.max_width and img.width > self.max_width:
            scale = self.max_width / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        return img

    def label_transform(self, y: str) -> str:
        """Override if labels need normalization."""
        return str(y)


TENNIS_SCOREBOARD_TASK = TaskConfig(
    prompt_text = (
    "Locate the on-screen tennis SCOREBOARD overlay (often in a corner). "
    "Read ONLY the scoreboard table (ignore everything else).\n"
    "\n"
    "The scoreboard is a 2-row table:\n"
    "- Top row = Player A\n"
    "- Bottom row = Player B\n"
    "\n"
    "Within EACH row, read numeric score cells from LEFT to RIGHT.\n"
    "- The RIGHTMOST score cell in each row is the current GAME/POINTS.\n"
    "  * Values: 0/15/30/40/AD OR tiebreak points like 0–99.\n"
    "- ALL numeric cells immediately to the LEFT of the game/points cell are SET scores (each 0–7).\n"
    "  * Include the current set even if highlighted.\n"
    "\n"
    "Ignore non-score elements anywhere: serve markers (/, dots, icons), flags, seeds like '(3)', "
    "country codes, round labels, sponsors, clocks, speed (e.g. '109 mph'), and any numbers not inside the 2-row table.\n"
    "\n"
    "Output EXACTLY one line in this format:\n"
    "SETS:<s1A-s1B s2A-s2B s3A-s3B ...> | GAME:<gA-gB>\n"
    "\n"
    "Rules:\n"
    "- A-B ordering is ALWAYS top row minus bottom row (for both SETS and GAME).\n"
    "- If NO set columns are visible, output: 'SETS:0-0 | GAME:<gA-gB>'\n"
    "- If no scoreboard visible, output `SETS: | GAME:0-0`.\n"
    "\n"
    "Example:\n"
    "SETS:6-4 3-6 | GAME:30-40\n"
    ),
    crop_left_frac=0.0,
    crop_right_frac=1.0,
    crop_top_frac=0.0,
    crop_bottom_frac=1.0,
    max_width=640,
    )


# -------------------------
# Helpers: chat message builders (generic)
# -------------------------
def build_user_messages(prompt_text: str, img: Image.Image):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image", "image": img},
        ],
    }]

def build_full_messages(prompt_text: str, img: Image.Image, label: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": img},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label}],
        },
    ]

def _as_tensor(x, dtype=None):
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    return t.to(dtype) if dtype is not None else t

def collate(batch, processor, torch_dtype):
    pad_id = processor.tokenizer.pad_token_id

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [_as_tensor(b["input_ids"], torch.long) for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [_as_tensor(b["attention_mask"], torch.long) for b in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [_as_tensor(b["labels"], torch.long) for b in batch],
        batch_first=True,
        padding_value=-100,
    )

    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack(
            [_as_tensor(b["pixel_values"], torch_dtype) for b in batch],
            dim=0
        )

    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--output_dir", type=str, default="vlm_lora_out")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # speed / dataloader knobs
    ap.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--dataloader_num_workers", type=int, default=2)

    # train knobs
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--logging_steps", type=int, default=5)

    args = ap.parse_args()

    task = TENNIS_SCOREBOARD_TASK  # swap this for another TaskConfig

    df = pd.read_csv(args.labels_csv)
    df = df[[task.image_col, task.label_col]].dropna(subset=[task.image_col]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_val = int(len(df) * args.val_frac)
    df_val = df.iloc[:n_val].copy()
    df_train = df.iloc[n_val:].copy()

    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_val = Dataset.from_pandas(df_val, preserve_index=False)

    processor = AutoProcessor.from_pretrained(args.model_id)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
        fp16 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # NOTE: MPS fp16 support/perf can vary; keep as fp16 if it works for you.
        torch_dtype = torch.float16
        fp16 = True
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        fp16 = False

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)

    # Important training speed/memory flag for decoder models
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # LoRA config (kept generic; you can override target_modules per model family)
    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False

    # ---- Faster masking: compute prompt_len ONCE (no image needed) ----
    # We build a “dummy” user message that includes an <image> placeholder token via chat template,
    # but we don’t run the full processor with images just to get length.
    dummy_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    prompt_str = processor.apply_chat_template(
        build_user_messages(task.prompt_text, dummy_img),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_enc = processor(text=prompt_str, images=dummy_img, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    def preprocess(ex):
        img = Image.open(ex[task.image_col]).convert("RGB")
        img = task.image_transform(img)

        y = task.label_transform(ex[task.label_col])

        # prompt-only (ends right after the assistant tag)
        prompt_str = processor.apply_chat_template(
            build_user_messages(task.prompt_text, img),
            tokenize=False,
            add_generation_prompt=True,
        )

        # full example with assistant answer
        full_str = processor.apply_chat_template(
            build_full_messages(task.prompt_text, img, y),
            tokenize=False,
            add_generation_prompt=False,
        )

        # IMPORTANT: use processor for BOTH
        prompt_enc = processor(text=prompt_str, images=img, return_tensors="pt", truncation=False)
        full_enc   = processor(text=full_str,   images=img, return_tensors="pt", truncation=False)

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]
        pixel_values = full_enc.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values[0]

        prompt_ids = prompt_enc["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        # sanity check: prompt must be a prefix of full
        if prompt_len > input_ids.shape[0] or not torch.equal(input_ids[:prompt_len], prompt_ids):
            # Fallback: compute longest prefix match
            m = min(prompt_ids.shape[0], input_ids.shape[0])
            k = 0
            while k < m and int(prompt_ids[k]) == int(input_ids[k]):
                k += 1
            prompt_len = k

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Debug: warn if we somehow masked everything
        if int((labels != -100).sum()) == 0:
            print("WARNING: very few supervised tokens!", int((labels != -100).sum()),
                "prompt_len", prompt_len, "seq_len", input_ids.shape[0], "label", repr(y)[:60])

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if pixel_values is not None:
            out["pixel_values"] = pixel_values
        return out

    # Parallelise CPU preprocessing (image open/crop/resize/tokenise)
    ds_train = ds_train.map(preprocess, remove_columns=ds_train.column_names, num_proc=args.num_proc)
    ds_val = ds_val.map(preprocess, remove_columns=ds_val.column_names, num_proc=args.num_proc)

    train_steps_per_epoch = math.ceil(len(ds_train) / (args.per_device_train_batch_size * args.grad_accum))
    total_steps = train_steps_per_epoch * args.epochs
    print(f"Train examples: {len(ds_train)} | Steps/epoch: {train_steps_per_epoch} | Total steps: {total_steps}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=fp16,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,  # important for pixel_values
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=partial(collate, processor=processor, torch_dtype=torch_dtype),
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/adapter")
    processor.save_pretrained(f"{args.output_dir}/processor")
    print(f"Saved LoRA adapter to {args.output_dir}/adapter")


if __name__ == "__main__":
    main()