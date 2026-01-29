import cv2

# -------- config --------
VIDEO_PATH = "match.mp4"
# Time (seconds) to grab the frame from:
T = 7200.0
WINDOW_NAME = "ROI picker (drag a box around the scoreboard, press ENTER to accept)"
# ------------------------


roi_box = None  # (x, y, w, h)


def main():
    global roi_box

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, T * 1000.0)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame (try a different T)")

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}  (W x H), sampled at t={T:.2f}s")

    # OpenCV built-in interactive rectangle selector
    # Returns (x, y, w, h)
    r = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True, fromCenter=False)
    x, y, rw, rh = map(int, r)
    cv2.destroyAllWindows()

    if rw == 0 or rh == 0:
        print("No ROI selected.")
        return

    x0, y0 = x, y
    x1, y1 = x + rw, y + rh

    # Print pixel coords
    print("\nPixel ROI:")
    print(f"  x0={x0}, y0={y0}, x1={x1}, y1={y1}")

    # Print fractional coords (drop into ROI dataclass)
    fx0, fy0 = x0 / w, y0 / h
    fx1, fy1 = x1 / w, y1 / h
    print("\nFractional ROI (for ROI(x0,y0,x1,y1)):")
    print(f"  ROI({fx0:.6f}, {fy0:.6f}, {fx1:.6f}, {fy1:.6f})")

    # Optional: write debug crop image
    crop = frame[y0:y1, x0:x1]
    cv2.imwrite("debug_roi_crop.png", crop)
    print("\nSaved crop preview to: debug_roi_crop.png")


if __name__ == "__main__":
    main()