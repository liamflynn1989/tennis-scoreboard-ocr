"""
Finetune Qwen2-VL on cropped scoreboard images.

Usage:
    python finetune_qwen2vl_cropped.py --labels_csv /Users/liamflynn/tennis/cropped_scoreboard_data/labels.csv
"""

import os
import math
import argparse
from pathlib import Path
from functools import partial

import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

PROMPT_TEXT = """Read ONLY the numeric scores from this tennis scoreboard image.

IGNORE completely: player names, country flags, seed numbers in parentheses like (3) or (7), serve dots.

ONLY read the actual score NUMBERS which appear to the RIGHT of the player names:
- SET scores: small numbers like 0, 1, 2, 3, 4, 5, 6, 7
- GAME score: the rightmost numbers, either 0/15/30/40/AD or tiebreak points

Top row = Player A scores
Bottom row = Player B scores

Output ONLY in this format:
SETS:<setA1-setB1 setA2-setB2 ...> | GAME:<gameA-gameB>

For example if top row shows "6 4 30" and bottom shows "3 6 15", output:
SETS:6-3 4-6 | GAME:30-15"""


# -------------------------
# Helpers
# -------------------------
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
        out["pixel_values"] = torch.cat(
            [_as_tensor(b["pixel_values"], torch_dtype) for b in batch],
            dim=0
        )

    if "image_grid_thw" in batch[0]:
        out["image_grid_thw"] = torch.cat(
            [_as_tensor(b["image_grid_thw"], torch.long) for b in batch],
            dim=0
        )

    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default="/Users/liamflynn/tennis/cropped_scoreboard_data/labels.csv")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--output_dir", type=str, default="/Users/liamflynn/tennis/qwen2vl_cropped_lora")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # Speed / dataloader
    ap.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--dataloader_num_workers", type=int, default=2)

    # Training params - conservative for 7B model
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--logging_steps", type=int, default=5)

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)

    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.labels_csv)
    df = df[["image_path", "label"]].dropna().reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_val = int(len(df) * args.val_frac)
    df_val = df.iloc[:n_val].copy()
    df_train = df.iloc[n_val:].copy()

    print(f"Train: {len(df_train)}, Val: {len(df_val)}")

    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_val = Dataset.from_pandas(df_val, preserve_index=False)

    processor = AutoProcessor.from_pretrained(args.model_id)

    # Device selection
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        bf16 = True
        fp16 = False
    elif torch.backends.mps.is_available():
        torch_dtype = torch.float16
        bf16 = False
        fp16 = True
    else:
        torch_dtype = torch.float32
        bf16 = False
        fp16 = False

    print(f"dtype: {torch_dtype}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # LoRA config for Qwen2-VL
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    def preprocess(ex):
        img = Image.open(ex["image_path"]).convert("RGB")
        label = ex["label"]

        # Build messages for Qwen2-VL
        user_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }]

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

        # Process with Qwen2-VL
        from qwen_vl_utils import process_vision_info

        prompt_text = processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Get image inputs
        image_inputs, _ = process_vision_info(user_messages)

        prompt_enc = processor(
            text=[prompt_text],
            images=image_inputs,
            padding=False,
            return_tensors="pt",
        )

        full_enc = processor(
            text=[full_text],
            images=image_inputs,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]
        pixel_values = full_enc.get("pixel_values", None)
        image_grid_thw = full_enc.get("image_grid_thw", None)

        prompt_ids = prompt_enc["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        if prompt_len > input_ids.shape[0] or not torch.equal(input_ids[:prompt_len], prompt_ids):
            m = min(prompt_ids.shape[0], input_ids.shape[0])
            k = 0
            while k < m and int(prompt_ids[k]) == int(input_ids[k]):
                k += 1
            prompt_len = k

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        if int((labels != -100).sum()) == 0:
            print("WARNING: no supervised tokens!", "label:", repr(label)[:60])

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if pixel_values is not None:
            out["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw
        return out

    print("Preprocessing train set...")
    ds_train = ds_train.map(preprocess, remove_columns=ds_train.column_names, num_proc=1)  # num_proc=1 for qwen_vl_utils
    print("Preprocessing val set...")
    ds_val = ds_val.map(preprocess, remove_columns=ds_val.column_names, num_proc=1)

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
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=False,  # Disabled - causes issues with LoRA on MPS
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=partial(collate, processor=processor, torch_dtype=torch_dtype),
    )

    trainer.train()

    # Save
    model.save_pretrained(f"{args.output_dir}/adapter")
    processor.save_pretrained(f"{args.output_dir}/processor")
    print(f"Saved LoRA adapter to {args.output_dir}/adapter")


if __name__ == "__main__":
    main()
