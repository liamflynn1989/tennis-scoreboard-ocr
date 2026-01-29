"""
Process a tennis match video to extract scores and point winners.

Usage:
    python process_match_video.py --video /Users/liamflynn/tennis/match1.mp4
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# -------------------------
# Configuration
# -------------------------
YOLO_MODEL_PATH = "/Users/liamflynn/tennis/yolo_scoreboard_model/scoreboard_detector/weights/best.pt"
VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
VLM_ADAPTER_PATH = "/Users/liamflynn/tennis/qwen2vl_cropped_lora/adapter"

YOLO_CONF_THRESHOLD = 0.5
BBOX_PADDING = 10
SAMPLE_INTERVAL_SEC = 15.0  # Sample every N seconds

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


def parse_score(score_str: str) -> dict | None:
    """Parse score string into structured format."""
    if not score_str:
        return None

    match = re.match(r"SETS:(.+?) \| GAME:(.+)", score_str)
    if not match:
        return None

    sets_str = match.group(1).strip()
    game_str = match.group(2).strip()

    # Parse sets
    sets = []
    if sets_str:
        for s in sets_str.split():
            parts = s.split("-")
            if len(parts) == 2:
                sets.append((parts[0], parts[1]))

    # Parse game
    game_parts = game_str.split("-")
    if len(game_parts) == 2:
        game = (game_parts[0], game_parts[1])
    else:
        game = None

    return {"sets": sets, "game": game, "raw": score_str}


def determine_point_winner(prev_score: dict, curr_score: dict) -> str | None:
    """
    Determine who won the point based on score change.
    Returns '1' (top player), '2' (bottom player), or None if unclear.
    """
    if not prev_score or not curr_score:
        return None

    prev_game = prev_score.get("game")
    curr_game = curr_score.get("game")
    prev_sets = prev_score.get("sets", [])
    curr_sets = curr_score.get("sets", [])

    if not prev_game or not curr_game:
        return None

    # Game point values for comparison
    point_order = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

    prev_a, prev_b = prev_game
    curr_a, curr_b = curr_game

    # Check if game was won (game score reset to 0-0)
    if curr_a == "0" and curr_b == "0" and not (prev_a == "0" and prev_b == "0"):
        # Game was won - check who won by comparing set scores
        if len(curr_sets) > 0 and len(prev_sets) > 0:
            # Check current set game count
            curr_set_idx = len(curr_sets) - 1
            if curr_set_idx < len(prev_sets):
                prev_set = prev_sets[curr_set_idx]
                curr_set = curr_sets[curr_set_idx]
                try:
                    if int(curr_set[0]) > int(prev_set[0]):
                        return "1"
                    elif int(curr_set[1]) > int(prev_set[1]):
                        return "2"
                except (ValueError, IndexError):
                    pass
        # Check if new set started
        if len(curr_sets) > len(prev_sets):
            # Previous set was won
            return None  # Can't easily determine from set change
        return None

    # Check if point was won within game
    try:
        prev_a_val = point_order.get(prev_a, int(prev_a) if prev_a.isdigit() else -1)
        prev_b_val = point_order.get(prev_b, int(prev_b) if prev_b.isdigit() else -1)
        curr_a_val = point_order.get(curr_a, int(curr_a) if curr_a.isdigit() else -1)
        curr_b_val = point_order.get(curr_b, int(curr_b) if curr_b.isdigit() else -1)

        # Player 1 won point if their score increased or Player 2's AD was cancelled
        if curr_a_val > prev_a_val:
            return "1"
        if curr_b_val > prev_b_val:
            return "2"
        # Deuce scenarios
        if prev_a == "AD" and curr_a == "40" and curr_b == "40":
            return "2"
        if prev_b == "AD" and curr_a == "40" and curr_b == "40":
            return "1"
    except (ValueError, TypeError):
        pass

    return None


class VideoScoreExtractor:
    def __init__(self, yolo_path: str, vlm_model_id: str, vlm_adapter_path: str = None):
        print("Loading YOLO...")
        self.yolo = YOLO(yolo_path)

        print("Loading VLM...")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(vlm_model_id)
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

        if vlm_adapter_path:
            print(f"Loading LoRA adapter from {vlm_adapter_path}...")
            self.vlm = PeftModel.from_pretrained(self.vlm, vlm_adapter_path)

        self.vlm.eval()
        self.yolo_conf = YOLO_CONF_THRESHOLD
        self.bbox_padding = BBOX_PADDING
        print("Ready!")

    def detect_and_crop(self, frame: Image.Image) -> Image.Image | None:
        """Detect scoreboard and return cropped image."""
        results = self.yolo.predict(frame, conf=self.yolo_conf, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        # Crop with padding
        crop_x1 = max(0, int(x1) - self.bbox_padding)
        crop_y1 = max(0, int(y1) - self.bbox_padding)
        crop_x2 = min(frame.width, int(x2) + self.bbox_padding)
        crop_y2 = min(frame.height, int(y2) + self.bbox_padding)

        return frame.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def read_score(self, cropped: Image.Image) -> str:
        """Extract score from cropped scoreboard image."""
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": cropped},
                {"type": "text", "text": SCOREBOARD_PROMPT},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.vlm.device)

        with torch.no_grad():
            output_ids = self.vlm.generate(**inputs, max_new_tokens=64, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        result = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

        return result.strip()

    def process_video(self, video_path: str, output_csv: str, sample_interval: float = 1.0):
        """Process video and extract scores."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")
        print(f"Sampling every {sample_interval}s...")

        frame_interval = int(fps * sample_interval)

        results = []
        prev_score = None
        prev_score_str = None
        frame_num = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_num / fps

            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            # Detect and crop scoreboard
            cropped = self.detect_and_crop(pil_frame)

            score_str = None
            winner = None

            if cropped:
                score_str = self.read_score(cropped)
                curr_score = parse_score(score_str)

                # Determine point winner if score changed
                if score_str != prev_score_str and curr_score:
                    winner = determine_point_winner(prev_score, curr_score)
                    prev_score = curr_score
                    prev_score_str = score_str

            results.append({
                "frame": frame_num,
                "time": f"{time_sec:.2f}",
                "score": score_str or "",
                "winner": winner or "",
            })

            # Progress
            print(f"Frame {frame_num} ({time_sec:.1f}s): {score_str or 'no scoreboard'}" +
                  (f" - Point to Player {winner}" if winner else ""))

            frame_num += frame_interval
            if frame_num >= total_frames:
                break

        cap.release()

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "time", "score", "winner"])
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved {len(results)} rows to {output_csv}")

        # Summary
        points_1 = sum(1 for r in results if r["winner"] == "1")
        points_2 = sum(1 for r in results if r["winner"] == "2")
        print(f"Points detected - Player 1: {points_1}, Player 2: {points_2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL_SEC, help="Sample interval in seconds")
    parser.add_argument("--yolo", type=str, default=YOLO_MODEL_PATH)
    parser.add_argument("--vlm", type=str, default=VLM_MODEL_ID)
    parser.add_argument("--adapter", type=str, default=VLM_ADAPTER_PATH)
    args = parser.parse_args()

    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.with_suffix(".scores.csv"))

    extractor = VideoScoreExtractor(args.yolo, args.vlm, args.adapter)
    extractor.process_video(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()
