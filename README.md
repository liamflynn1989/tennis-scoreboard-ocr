# Tennis Scoreboard OCR Pipeline

Extract scores from tennis match videos using a two-stage ML pipeline:
1. **YOLO** - Detects and crops the scoreboard region
2. **Qwen2-VL** - Reads the score from the cropped image

## Quick Start

Process a video to extract scores:

```bash
python process_match_video.py --video match1.mp4
```

Output: `match1.scores.csv` with columns: `frame, time, score, winner`

## Pipeline Overview

```
Video Frame → YOLO (detect scoreboard) → Crop → Qwen2-VL (read score) → Structured Output
```

## Setup

### Requirements

```bash
pip install torch torchvision transformers peft ultralytics qwen-vl-utils pandas pillow opencv-python
```

### Pre-trained Models Required

1. **YOLO scoreboard detector**: `yolo_scoreboard_model/scoreboard_detector/weights/best.pt`
2. **Qwen2-VL with LoRA adapter**: `qwen2vl_cropped_lora/adapter/`

## Training Your Own Models

### Step 1: Label Bounding Boxes

Label scoreboard locations in video frames:

```bash
python make_bbox_labels.py
```

- Edit the script to point to your video
- Click and drag to draw bounding boxes
- Press Enter to submit, "Pass" if no scoreboard visible
- Output: `bbox_labels.csv`

### Step 2: Train YOLO Detector

```bash
python finetune_yolo_scoreboard.py
```

- Combines bbox labels and trains YOLOv8
- Output: `yolo_scoreboard_model/scoreboard_detector/weights/best.pt`

### Step 3: Label Scores

Label the actual score values:

```bash
python make_labels.py
```

- Edit the script to point to your video
- Fill in score values for each player
- Output: `labels.csv`

### Step 4: Prepare Cropped Dataset

Use YOLO to crop scoreboards for VLM training:

```bash
python prepare_cropped_dataset.py
```

- Uses trained YOLO to detect and crop scoreboards
- Output: `cropped_scoreboard_data/` with images and labels

### Step 5: Finetune VLM

```bash
python finetune_qwen2vl_cropped.py
```

- Finetunes Qwen2-VL-7B with LoRA on cropped scoreboards
- Output: `qwen2vl_cropped_lora/adapter/`

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `process_match_video.py` | Process video and extract scores + point winners |
| `score_pipeline.py` | Run inference on images (single or batch) |
| `make_bbox_labels.py` | GUI tool to label scoreboard bounding boxes |
| `make_labels.py` | GUI tool to label score values |
| `finetune_yolo_scoreboard.py` | Train YOLO scoreboard detector |
| `prepare_cropped_dataset.py` | Crop scoreboards using YOLO for VLM training |
| `finetune_qwen2vl_cropped.py` | Finetune Qwen2-VL on cropped scoreboards |

## Output Format

### Score Format

```
SETS:6-4 3-6 | GAME:30-15
```

- Sets are listed left to right (e.g., `6-4 3-6` = Player 1 won set 1, Player 2 won set 2)
- First number is Player 1 (top row), second is Player 2 (bottom row)

### Video Processing Output

```csv
frame,time,score,winner
0,0.00,SETS:0-0 | GAME:0-0,
900,15.00,SETS:0-0 | GAME:15-0,1
1800,30.00,SETS:0-0 | GAME:30-0,1
```

- `winner`: `1` = Player 1 (top), `2` = Player 2 (bottom), empty = no point detected

## Directory Structure

```
tennis/
├── README.md
├── process_match_video.py      # Main video processing script
├── score_pipeline.py           # Inference pipeline
├── make_labels.py              # Score labeling tool
├── make_bbox_labels.py         # Bbox labeling tool
├── finetune_yolo_scoreboard.py # YOLO training
├── prepare_cropped_dataset.py  # Dataset preparation
├── finetune_qwen2vl_cropped.py # VLM finetuning
├── yolo_scoreboard_model/      # Trained YOLO model
├── qwen2vl_cropped_lora/       # Trained VLM adapter
├── cropped_scoreboard_data/    # Training data for VLM
├── match1/                     # Labeled frames from match 1
├── match1_bbox/                # Bbox labels for match 1
├── match2/                     # Labeled frames from match 2
├── match2_bbox/                # Bbox labels for match 2
└── archive/                    # Old experimental scripts
```
