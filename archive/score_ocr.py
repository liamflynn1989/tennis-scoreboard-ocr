import cv2
import csv
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import torch.nn.functional as F


import os

@dataclass
class ROI:
    # ROI in fractional coordinates (relative to frame width/height)
    # Adjust these once by printing frames and tuning.
    x0: float
    y0: float
    x1: float
    y1: float


def crop_roi_bgr(frame_bgr, roi: ROI):
    h, w = frame_bgr.shape[:2]
    x0 = int(roi.x0 * w)
    y0 = int(roi.y0 * h)
    x1 = int(roi.x1 * w)
    y1 = int(roi.y1 * h)
    cropped = frame_bgr[y0:y1, x0:x1]
    return cropped


def draw_roi(frame_bgr, roi: ROI):
    h, w = frame_bgr.shape[:2]
    x0 = int(roi.x0 * w); y0 = int(roi.y0 * h)
    x1 = int(roi.x1 * w); y1 = int(roi.y1 * h)
    out = frame_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return out, (x0, y0, x1, y1)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def preprocess_for_ocr(crop_bgr):
    """
    Light preprocessing helps OCR a lot.
    You can tweak these based on your broadcast graphics.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # denoise + contrast
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)
    # binarize
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr




def ocr_trocr_with_confidence(image_uint8, processor, model, device, max_new_tokens=16):
    # Ensure 3-channel RGB for the processor
    if image_uint8.ndim == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    elif image_uint8.shape[2] == 3:
        # if it's BGR from OpenCV, convert to RGB
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(image_uint8)
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        out = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = out.sequences[0]  # token ids, includes special tokens
    text = processor.decode(seq, skip_special_tokens=True).strip()

    # Compute per-token probabilities for *generated* tokens only
    # out.scores is a list: length = number of generated tokens (steps)
    # Each scores[t] has shape [batch, vocab]
    token_probs = []
    for step_logits, token_id in zip(out.scores, seq[-len(out.scores):]):
        probs = F.softmax(step_logits[0], dim=-1)
        token_probs.append(probs[token_id].item())

    if len(token_probs) == 0:
        return text, 0.0, 0.0, []

    # Confidence summaries
    avg_logprob = float(sum(torch.log(torch.tensor(token_probs))).item() / len(token_probs))
    avg_prob = float(torch.exp(torch.tensor(avg_logprob)).item())     # geometric mean prob
    min_prob = float(min(token_probs))                                # strict

    return text, avg_prob, min_prob, token_probs

def main(
    video_path: str,
    out_csv: str = "score_ocr.csv",
    sample_every_s: float = 300.0,
    roi: ROI = ROI(0.040625, 0.827778, 0.392969, 0.966667),  # <-- EDIT THIS
    debug: bool = False,
    debug_dir: str = "debug",
    debug_every: int = 1,      # save every N sampled frames
    debug_show: bool = False,   # pop up windows (optional)
):
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / fps if frame_count else None

    # HF OCR model (TrOCR)
    model_id = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # sampling
    last_sample_t = -1e9  # very negative so first frame triggers

    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds

        if t - last_sample_t >= sample_every_s:
            # print(f"Processing frame {frame_idx} at t={t:.2f}s {last_sample_t=:.2f}s")
            last_sample_t = t

            crop = crop_roi_bgr(frame, roi)
            pre = preprocess_for_ocr(crop)

            if debug:
                ensure_dir(debug_dir)

                sample_idx = len(rows)  # number of sampled frames so far
                if sample_idx % debug_every == 0:
                    framed, (x0, y0, x1, y1) = draw_roi(frame, roi)

                    # Save full frame with ROI box
                    cv2.imwrite(f"{debug_dir}/frame_{frame_idx:07d}_t{t:07.2f}_roi.png", framed)

                    # Save raw crop
                    cv2.imwrite(f"{debug_dir}/frame_{frame_idx:07d}_t{t:07.2f}_crop.png", crop)

                    # Save the preprocessed image fed to OCR
                    # If it's grayscale (H,W), it's fine to write directly
                    # cv2.imwrite(f"{debug_dir}/frame_{frame_idx:07d}_t{t:07.2f}_pre.png", pre)

                    if debug_show:
                        cv2.imshow("frame (ROI boxed)", framed)
                        cv2.imshow("crop", crop)
                        cv2.imshow("preprocessed", pre)
                        # press any key to continue; ESC to quit early
                        k = cv2.waitKey(0) & 0xFF
                        if k == 27:  # ESC
                            break

                # text, avg_prob, min_prob, token_probs = ocr_trocr_with_confidence(crop, processor, model, device)
            
                # # Example gating rule (tune thresholds)
                # if avg_prob < 0.25 or min_prob < 0.05:
                #     text = ""   # or None / keep last good value

                # rows.append((t, frame_idx, text, avg_prob, min_prob, token_probs))
                # print(f"t={t:7.2f}s  frame={frame_idx:7d}  text='{text}'  avg_prob={avg_prob:.4f}  min_prob={min_prob:.4f}")

        frame_idx += 1

    cap.release()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame_idx", "ocr_text"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}")
    if duration_s is not None:
        print(f"Video duration ~ {duration_s:.1f}s, sampled at ~{sample_fps} fps")


if __name__ == "__main__":
    main("match.mp4", debug=True, debug_show=False)