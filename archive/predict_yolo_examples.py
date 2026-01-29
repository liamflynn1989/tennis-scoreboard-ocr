"""
Show example predictions from the trained YOLO scoreboard detector.
"""

from pathlib import Path
from ultralytics import YOLO
import random

MODEL_PATH = "/Users/liamflynn/tennis/yolo_scoreboard_model/scoreboard_detector/weights/best.pt"
VAL_IMAGES_DIR = Path("/Users/liamflynn/tennis/yolo_scoreboard_data/images/val")
OUTPUT_DIR = Path("/Users/liamflynn/tennis/yolo_predictions")

N_EXAMPLES = 10

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = YOLO(MODEL_PATH)

    # Get validation images
    val_images = list(VAL_IMAGES_DIR.glob("*.png"))
    if not val_images:
        val_images = list(VAL_IMAGES_DIR.glob("*.jpg"))

    print(f"Found {len(val_images)} validation images")

    # Sample random images
    random.seed(42)
    sample_images = random.sample(val_images, min(N_EXAMPLES, len(val_images)))

    # Run predictions and save
    results = model.predict(
        source=sample_images,
        save=True,
        project=str(OUTPUT_DIR),
        name="examples",
        exist_ok=True,
        conf=0.25,
    )

    # Print results summary
    print(f"\nPredictions saved to: {OUTPUT_DIR}/examples/")
    print("\nSummary:")
    for img_path, result in zip(sample_images, results):
        boxes = result.boxes
        if len(boxes) > 0:
            conf = boxes.conf[0].item()
            x1, y1, x2, y2 = boxes.xyxy[0].tolist()
            print(f"  {img_path.name}: detected (conf={conf:.2f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}])")
        else:
            print(f"  {img_path.name}: no detection")


if __name__ == "__main__":
    main()
