#!/usr/bin/env python3
"""
evaluate.py

Evaluate a SmolVLM2 + LoRA adapter on a labeled CSV of images.

Example:
  python evaluate.py \
    --labels_csv new_labels.csv \
    --adapter_dir smolvlm_score_lora2/adapter \
    --processor_dir smolvlm_score_lora2/processor \
    --base_model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --image_col image_path \
    --label_col label \
    --max_new_tokens 64
"""

import argparse
import os
import re
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from peft import PeftModel


# ---------------------------
# Utilities
# ---------------------------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def safe_open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


_SCORE_RE = re.compile(
    r"^SETS:(?P<sets>(?:\d-\d(?:\s+\d-\d)*)?)\s*\|\s*GAME:(?P<game>(?:AD|\d{1,2})-(?:AD|\d{1,2}))$"
)

def parse_score(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse one-line output like:
      SETS:4-6 6-4 1-1 | GAME:15-15
    Returns dict with sets list and game tuple, or None if invalid.
    """
    line = line.strip()
    m = _SCORE_RE.match(line)
    if not m:
        return None

    sets_str = m.group("sets").strip()
    sets = sets_str.split() if sets_str else []
    game = m.group("game")
    g_left, g_right = game.split("-", 1)

    return {"sets": sets, "game": (g_left, g_right), "raw": line}


def set_token_accuracy(pred_sets: List[str], true_sets: List[str]) -> float:
    """
    Token-level accuracy over set tokens (position-wise).
    If lengths differ, compare up to min length and treat extras as wrong.
    """
    if not true_sets and not pred_sets:
        return 1.0
    denom = max(len(true_sets), len(pred_sets))
    if denom == 0:
        return 1.0
    correct = 0
    for i in range(denom):
        p = pred_sets[i] if i < len(pred_sets) else None
        t = true_sets[i] if i < len(true_sets) else None
        correct += int(p == t)
    return correct / denom


def game_accuracy(pred_game: Tuple[str, str], true_game: Tuple[str, str]) -> float:
    return float(pred_game == true_game)


def build_eval_prompt(prompt_text: str, img: Image.Image) -> List[Dict[str, Any]]:
    """
    Builds the chat-style message list used at inference time.
    This should mirror your training prompt style for best results.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


# ---------------------------
# Main eval
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--processor_dir", type=str, required=True)
    ap.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument("--label_col", type=str, default="label")

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0.0 => greedy
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only first N rows")
    ap.add_argument("--save_csv", type=str, default="", help="Output CSV path (default: <adapter_dir>/predictions.csv)")

    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    df = df[[args.image_col, args.label_col]].dropna().reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    # Load processor (from training output)
    processor = AutoProcessor.from_pretrained(args.processor_dir)

    device, torch_dtype = pick_device()

    # Load base model then attach LoRA adapter
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.to(device).eval()

    # Important for generation consistency
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True  # generation ok

    # This should match your training prompt (use the improved one if you trained with it)
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
    )
    rows = []
    exact_correct = 0
    parsed_correct = 0
    set_token_acc_sum = 0.0
    game_acc_sum = 0.0
    valid_pred_count = 0
    valid_true_count = 0

    # Generation settings
    do_sample = args.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
    )
    # Remove Nones (transformers dislikes them)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.inference_mode():
        for i, r in df.iterrows():
            img_path = r[args.image_col]
            true_text = str(r[args.label_col]).strip()

            img = safe_open_image(img_path)
            messages = build_eval_prompt(prompt_text, img)

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Some processors emit token_type_ids; SmolVLM doesn't use it
            inputs.pop("token_type_ids", None)

            out_ids = model.generate(**inputs, **gen_kwargs)

            # Trim the prompt tokens
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[0][prompt_len:]

            pred_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
            # make sure single line
            pred_text = pred_text.splitlines()[0].strip() if pred_text else pred_text

            exact = (pred_text == true_text)
            exact_correct += int(exact)

            parsed_pred = parse_score(pred_text)
            parsed_true = parse_score(true_text)

            valid_pred = parsed_pred is not None
            valid_true = parsed_true is not None
            valid_pred_count += int(valid_pred)
            valid_true_count += int(valid_true)

            # "Parsed correct" means both parse and match structure+tokens
            parsed_match = False
            st_acc = None
            g_acc = None

            if valid_pred and valid_true:
                st_acc = set_token_accuracy(parsed_pred["sets"], parsed_true["sets"])
                g_acc = game_accuracy(parsed_pred["game"], parsed_true["game"])
                set_token_acc_sum += st_acc
                game_acc_sum += g_acc
                parsed_match = (st_acc == 1.0 and g_acc == 1.0)
                parsed_correct += int(parsed_match)

            rows.append(
                {
                    "image": img_path,
                    "label": true_text,
                    "pred": pred_text,
                    "exact_match": exact,
                    "pred_parse_ok": valid_pred,
                    "label_parse_ok": valid_true,
                    "set_token_acc": st_acc,
                    "game_acc": g_acc,
                    "parsed_match": parsed_match,
                }
            )

            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(df)}] exact={exact_correct/(i+1):.3f} pred_parse_ok={valid_pred_count/(i+1):.3f}")

    out_df = pd.DataFrame(rows)

    exact_acc = exact_correct / max(1, len(df))
    parsed_acc = parsed_correct / max(1, len(df))
    avg_set_token_acc = set_token_acc_sum / max(1, min(valid_pred_count, valid_true_count))
    avg_game_acc = game_acc_sum / max(1, min(valid_pred_count, valid_true_count))

    print("\n=== Results ===")
    print(f"Examples: {len(df)}")
    print(f"Exact-match accuracy: {exact_acc:.4f}")
    print(f"Parsed+token-match accuracy: {parsed_acc:.4f}")
    print(f"Pred parse rate: {valid_pred_count/len(df):.4f}")
    print(f"Avg set-token accuracy (on parseable pairs): {avg_set_token_acc:.4f}")
    print(f"Avg game accuracy (on parseable pairs): {avg_game_acc:.4f}")

    save_path = args.save_csv or os.path.join(os.path.dirname(args.adapter_dir.rstrip("/")), "predictions.csv")
    out_df.to_csv(save_path, index=False)
    print(f"Saved predictions to: {save_path}")


if __name__ == "__main__":
    main()