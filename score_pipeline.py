"""
Two-stage pipeline for tennis score extraction:
1. YOLO detects the scoreboard bounding box
2. Crop to scoreboard region
3. VLM reads the score from the cropped image

Usage:
    python score_pipeline.py --image /path/to/image.png
    python score_pipeline.py --image_dir /path/to/images/
    python score_pipeline.py --labels_csv /path/to/labels.csv  # evaluate against ground truth
"""

import argparse
import re
from pathlib import Path


def convert_label_format(label: str) -> str | None:
    """
    Convert from make_labels.py format to pipeline format.

    Input:  "p1:6,4,,,,,30|p2:3,6,,,,,15"
    Output: "SETS:6-3 4-6 | GAME:30-15"

    Returns None for empty/no scoreboard labels.
    """
    if not label or not isinstance(label, str):
        return None

    # Parse p1 and p2
    match = re.match(r"p1:([^|]*)\|p2:(.*)", label)
    if not match:
        # Already in correct format?
        if label.startswith("SETS:"):
            # Check if it's the "no scoreboard" format
            if label == "SETS: | GAME:0-0":
                return None
            return label
        return None

    p1_parts = match.group(1).split(",")
    p2_parts = match.group(2).split(",")

    # Pad to 6 elements
    p1_parts = (p1_parts + [""] * 6)[:6]
    p2_parts = (p2_parts + [""] * 6)[:6]

    # Handle empty/no scoreboard case: "p1:,,,,,|p2:,,,,,"
    if all(p.strip() == "" for p in p1_parts) and all(p.strip() == "" for p in p2_parts):
        return None

    # Sets are positions 0-4, game is position 5
    sets = []
    for i in range(5):
        s1, s2 = p1_parts[i].strip(), p2_parts[i].strip()
        if s1 or s2:
            s1 = s1 if s1 else "0"
            s2 = s2 if s2 else "0"
            sets.append(f"{s1}-{s2}")

    game1 = p1_parts[5].strip() if p1_parts[5].strip() else "0"
    game2 = p2_parts[5].strip() if p2_parts[5].strip() else "0"

    sets_str = " ".join(sets) if sets else "0-0"

    return f"SETS:{sets_str} | GAME:{game1}-{game2}"

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration


# -------------------------
# Configuration
# -------------------------
YOLO_MODEL_PATH = "/Users/liamflynn/tennis/yolo_scoreboard_model/scoreboard_detector/weights/best.pt"
VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # or "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

YOLO_CONF_THRESHOLD = 0.5
BBOX_PADDING = 10  # pixels to add around detected bbox

# Prompt optimized for cropped scoreboard image
SCOREBOARD_PROMPT = """Read ONLY the numeric scores from this tennis scoreboard image.

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


class ScorePipeline:
    def __init__(
        self,
        yolo_model_path: str = YOLO_MODEL_PATH,
        vlm_model_id: str = VLM_MODEL_ID,
        vlm_adapter_path: str = None,
        yolo_conf_threshold: float = YOLO_CONF_THRESHOLD,
        bbox_padding: int = BBOX_PADDING,
        device: str = None,
    ):
        self.yolo_conf_threshold = yolo_conf_threshold
        self.bbox_padding = bbox_padding

        # Load YOLO
        print(f"Loading YOLO from {yolo_model_path}...")
        self.yolo = YOLO(yolo_model_path)

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.torch_dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        # Load VLM
        print(f"Loading VLM from {vlm_model_id}...")
        self.vlm_model_id = vlm_model_id
        self.is_qwen = "qwen" in vlm_model_id.lower()

        if self.is_qwen:
            self.processor = AutoProcessor.from_pretrained(vlm_model_id)
            self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
        else:
            self.processor = AutoProcessor.from_pretrained(vlm_model_id)
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                vlm_model_id,
                torch_dtype=self.torch_dtype
            ).to(self.device)

        # Load LoRA adapter if provided
        if vlm_adapter_path:
            print(f"Loading LoRA adapter from {vlm_adapter_path}...")
            from peft import PeftModel
            self.vlm = PeftModel.from_pretrained(self.vlm, vlm_adapter_path)

        self.vlm.eval()
        print(f"Pipeline ready on {self.device}")

    def detect_scoreboard(self, image: Image.Image) -> dict | None:
        """Run YOLO to detect scoreboard. Returns bbox dict or None."""
        results = self.yolo.predict(image, conf=self.yolo_conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Take highest confidence detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = boxes.conf[best_idx].item()

        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf
        }

    def crop_scoreboard(self, image: Image.Image, bbox: dict) -> Image.Image:
        """Crop image to scoreboard region with padding."""
        x1 = max(0, int(bbox["x1"]) - self.bbox_padding)
        y1 = max(0, int(bbox["y1"]) - self.bbox_padding)
        x2 = min(image.width, int(bbox["x2"]) + self.bbox_padding)
        y2 = min(image.height, int(bbox["y2"]) + self.bbox_padding)

        return image.crop((x1, y1, x2, y2))

    def read_score(self, cropped_image: Image.Image) -> str:
        """Run VLM on cropped scoreboard image to extract score."""
        if self.is_qwen:
            # Qwen2-VL format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": SCOREBOARD_PROMPT},
                ],
            }]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.vlm.device)

            with torch.no_grad():
                output_ids = self.vlm.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            # Decode only the generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0, input_len:]
            result = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

        else:
            # SmolVLM / other models
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCOREBOARD_PROMPT},
                    {"type": "image", "image": cropped_image},
                ],
            }]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=prompt,
                images=cropped_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.vlm.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            # Decode only the generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0, input_len:]
            result = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

        return result.strip()

    def predict(self, image_path: str | Path, save_crop_dir: Path = None) -> dict:
        """
        Full pipeline: detect scoreboard, crop, read score.

        Returns dict with:
            - scoreboard_detected: bool
            - bbox: dict or None
            - prediction: str or None
            - image_path: str
            - crop_path: str or None (if save_crop_dir provided)
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        result = {
            "image_path": str(image_path),
            "scoreboard_detected": False,
            "bbox": None,
            "prediction": None,
            "crop_path": None,
        }

        # Stage 1: Detect scoreboard
        bbox = self.detect_scoreboard(image)
        if bbox is None:
            return result

        result["scoreboard_detected"] = True
        result["bbox"] = bbox

        # Stage 2: Crop
        cropped = self.crop_scoreboard(image, bbox)

        # Save crop if requested
        if save_crop_dir:
            save_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = save_crop_dir / f"crop_{image_path.name}"
            cropped.save(crop_path)
            result["crop_path"] = str(crop_path)

        # Stage 3: Read score
        prediction = self.read_score(cropped)
        result["prediction"] = prediction

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--image_dir", type=str, help="Directory of images to process")
    parser.add_argument("--labels_csv", type=str, help="CSV with image_path and label columns for evaluation")
    parser.add_argument("--vlm_adapter", type=str, help="Path to LoRA adapter for VLM (optional)")
    parser.add_argument("--vlm_model", type=str, default=VLM_MODEL_ID, help="VLM model ID (default: Qwen2-VL-7B-Instruct)")
    parser.add_argument("--yolo_conf", type=float, default=YOLO_CONF_THRESHOLD)
    parser.add_argument("--output_csv", type=str, help="Save predictions to CSV")
    parser.add_argument("--save_crops", type=str, help="Directory to save cropped scoreboard images")
    args = parser.parse_args()

    save_crop_dir = Path(args.save_crops) if args.save_crops else None

    pipeline = ScorePipeline(
        vlm_model_id=args.vlm_model,
        vlm_adapter_path=args.vlm_adapter,
        yolo_conf_threshold=args.yolo_conf,
    )

    results = []

    if args.image:
        result = pipeline.predict(args.image, save_crop_dir=save_crop_dir)
        print(f"\nResult for {args.image}:")
        print(f"  Scoreboard detected: {result['scoreboard_detected']}")
        if result['bbox']:
            print(f"  BBox conf: {result['bbox']['conf']:.2f}")
        if result['crop_path']:
            print(f"  Crop saved to: {result['crop_path']}")
        print(f"  Prediction: {result['prediction']}")
        results.append(result)

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        print(f"Processing {len(image_paths)} images...")

        for img_path in image_paths:
            result = pipeline.predict(img_path, save_crop_dir=save_crop_dir)
            results.append(result)
            status = "detected" if result["scoreboard_detected"] else "no scoreboard"
            print(f"{img_path.name}: {status} -> {result['prediction']}")

    elif args.labels_csv:
        import pandas as pd

        df = pd.read_csv(args.labels_csv)
        print(f"Evaluating on {len(df)} images...")

        correct = 0
        detected = 0

        for _, row in df.iterrows():
            result = pipeline.predict(row["image_path"], save_crop_dir=save_crop_dir)
            # Convert label format - returns None for no scoreboard
            raw_label = row.get("label", "")
            label = convert_label_format(raw_label)
            result["label"] = label

            results.append(result)

            if result["scoreboard_detected"]:
                detected += 1

            # Check for match - both None counts as correct
            pred = result["prediction"]
            if label is None and not result["scoreboard_detected"]:
                exact_match = True  # Both agree: no scoreboard
            elif label is None and result["scoreboard_detected"]:
                exact_match = False  # Label says none, but we detected one
            elif label is not None and not result["scoreboard_detected"]:
                exact_match = False  # Label has score, but we didn't detect
            else:
                exact_match = pred == label  # Both have values, compare them

            if exact_match:
                correct += 1

            status = "correct" if exact_match else "wrong"
            print(f"{Path(row['image_path']).name}: {status}")
            print(f"  Label: {result['label']}")
            print(f"  Pred:  {result['prediction']}")

        print(f"\n--- Results ---")
        print(f"Scoreboard detected: {detected}/{len(df)} ({100*detected/len(df):.1f}%)")
        print(f"Exact match: {correct}/{len(df)} ({100*correct/len(df):.1f}%)")

    # Save to CSV if requested
    if args.output_csv and results:
        import pandas as pd
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
