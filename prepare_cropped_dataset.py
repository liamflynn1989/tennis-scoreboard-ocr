"""
Prepare a training dataset of cropped scoreboard images using the YOLO detector.

1. Loads labels from match1 and match2
2. Runs YOLO to detect and crop scoreboards
3. Saves cropped images and creates a new labels CSV

Usage:
    python prepare_cropped_dataset.py
"""

import re
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

# -------------------------
# Configuration
# -------------------------
LABEL_CSV_PATHS = [
    "/Users/liamflynn/tennis/match1/labels.csv",
    "/Users/liamflynn/tennis/match2/labels.csv",
]

YOLO_MODEL_PATH = "/Users/liamflynn/tennis/yolo_scoreboard_model/scoreboard_detector/weights/best.pt"
OUTPUT_DIR = Path("/Users/liamflynn/tennis/cropped_scoreboard_data")

YOLO_CONF_THRESHOLD = 0.5
BBOX_PADDING = 10


def convert_label_format(label: str) -> str:
    """
    Convert from make_labels.py format to pipeline format.

    Input:  "p1:6,4,,,,,30|p2:3,6,,,,,15"
    Output: "SETS:6-3 4-6 | GAME:30-15"
    """
    if not label or not isinstance(label, str):
        return ""

    match = re.match(r"p1:([^|]*)\|p2:(.*)", label)
    if not match:
        if label.startswith("SETS:"):
            return label
        return ""

    p1_parts = match.group(1).split(",")
    p2_parts = match.group(2).split(",")

    p1_parts = (p1_parts + [""] * 6)[:6]
    p2_parts = (p2_parts + [""] * 6)[:6]

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

    if all(p.strip() == "" for p in p1_parts) and all(p.strip() == "" for p in p2_parts):
        return ""

    return f"SETS:{sets_str} | GAME:{game1}-{game2}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUTPUT_DIR / "images"
    crops_dir.mkdir(exist_ok=True)

    print(f"Loading YOLO from {YOLO_MODEL_PATH}...")
    yolo = YOLO(YOLO_MODEL_PATH)

    # Load and combine labels
    dfs = []
    for csv_path in LABEL_CSV_PATHS:
        df = pd.read_csv(csv_path)
        df["source"] = Path(csv_path).parent.name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total images: {len(df)}")

    rows = []
    skipped_no_detection = 0
    skipped_no_label = 0

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"Warning: {image_path} not found")
            continue

        # Convert label
        label = convert_label_format(row.get("label", ""))
        if not label:
            skipped_no_label += 1
            continue

        # Run YOLO
        image = Image.open(image_path).convert("RGB")
        results = yolo.predict(image, conf=YOLO_CONF_THRESHOLD, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            skipped_no_detection += 1
            continue

        # Get best detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        # Crop with padding
        crop_x1 = max(0, int(x1) - BBOX_PADDING)
        crop_y1 = max(0, int(y1) - BBOX_PADDING)
        crop_x2 = min(image.width, int(x2) + BBOX_PADDING)
        crop_y2 = min(image.height, int(y2) + BBOX_PADDING)

        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Save cropped image
        crop_filename = f"{row['source']}_{image_path.name}"
        crop_path = crops_dir / crop_filename
        cropped.save(crop_path)

        rows.append({
            "image_path": str(crop_path),
            "original_image": str(image_path),
            "label": label,
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} images...")

    # Save labels CSV
    out_df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "labels.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n--- Summary ---")
    print(f"Total input images: {len(df)}")
    print(f"Skipped (no label): {skipped_no_label}")
    print(f"Skipped (no YOLO detection): {skipped_no_detection}")
    print(f"Cropped images saved: {len(rows)}")
    print(f"Output CSV: {out_csv}")
    print(f"Crops directory: {crops_dir}")


if __name__ == "__main__":
    main()
