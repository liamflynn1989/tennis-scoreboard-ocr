"""
Finetune YOLOv8 to detect tennis scoreboards.

Usage:
    python finetune_yolo_scoreboard.py

Requires: pip install ultralytics
"""

import argparse
import shutil
import random
from pathlib import Path

import pandas as pd
from PIL import Image

# -------------------------
# Configuration
# -------------------------
BBOX_CSV_PATHS = [
    "/Users/liamflynn/tennis/match1_bbox/bbox_labels.csv",
    "/Users/liamflynn/tennis/match2_bbox/bbox_labels.csv",
]

OUTPUT_DIR = Path("/Users/liamflynn/tennis/yolo_scoreboard_data")
MODEL_OUTPUT_DIR = Path("/Users/liamflynn/tennis/yolo_scoreboard_model")

VAL_FRAC = 0.2
SEED = 42

# YOLO training params
YOLO_MODEL = "yolov8n.pt"  # nano model, fast. Use yolov8s.pt or yolov8m.pt for more accuracy
EPOCHS = 100
IMGSZ = 640
BATCH = 16


def convert_to_yolo_format(x1, y1, x2, y2, img_w, img_h):
    """
    Convert (x1, y1, x2, y2) absolute coords to YOLO format:
    (x_center, y_center, width, height) normalized to 0-1
    """
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def prepare_dataset():
    """Load CSVs, combine, split, and create YOLO directory structure."""

    # Load and combine all CSVs
    dfs = []
    for csv_path in BBOX_CSV_PATHS:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    print(f"Total images: {len(df)}")
    print(f"With scoreboard: {df['has_scoreboard'].sum()}")
    print(f"Without scoreboard: {(df['has_scoreboard'] == 0).sum()}")

    # Shuffle and split
    random.seed(SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)

    n_val = int(len(df) * VAL_FRAC)
    val_indices = set(indices[:n_val])

    # Create directory structure
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process each image
    train_count, val_count = 0, 0
    for idx, row in df.iterrows():
        split = "val" if idx in val_indices else "train"

        img_path = Path(row["image_path"])
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Copy image to YOLO structure
        new_img_name = f"{img_path.parent.name}_{img_path.name}"
        new_img_path = OUTPUT_DIR / "images" / split / new_img_name
        shutil.copy(img_path, new_img_path)

        # Create label file
        label_path = OUTPUT_DIR / "labels" / split / new_img_name.replace(".png", ".txt").replace(".jpg", ".txt")

        if row["has_scoreboard"] == 1:
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            x_c, y_c, w, h = convert_to_yolo_format(x1, y1, x2, y2, img_w, img_h)
            # Class 0 = scoreboard
            with open(label_path, "w") as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        else:
            # Empty label file = no objects (negative example)
            label_path.touch()

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    print(f"Train images: {train_count}, Val images: {val_count}")

    # Create data.yaml
    data_yaml = OUTPUT_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: scoreboard\n")

    print(f"Created {data_yaml}")
    return data_yaml


def train_yolo(data_yaml: Path, epochs: int, batch: int, model_name: str):
    """Train YOLOv8 model."""
    from ultralytics import YOLO

    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        project=str(MODEL_OUTPUT_DIR),
        name="scoreboard_detector",
        exist_ok=True,
        # Augmentation settings good for overlays/scoreboards
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0.0,      # No rotation - scoreboards are always horizontal
        translate=0.1,
        scale=0.3,
        flipud=0.0,       # No vertical flip
        fliplr=0.0,       # No horizontal flip - text would be backwards
        mosaic=0.5,
        mixup=0.0,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: {MODEL_OUTPUT_DIR}/scoreboard_detector/weights/best.pt")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train, assume dataset exists")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--model", type=str, default=YOLO_MODEL, help="Base YOLO model (yolov8n.pt, yolov8s.pt, etc.)")
    args = parser.parse_args()

    data_yaml = OUTPUT_DIR / "data.yaml"

    if not args.train_only:
        data_yaml = prepare_dataset()

    if not args.prepare_only:
        if not data_yaml.exists():
            print(f"Error: {data_yaml} not found. Run without --train-only first.")
            return
        train_yolo(data_yaml, args.epochs, args.batch, args.model)


if __name__ == "__main__":
    main()
