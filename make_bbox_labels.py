import csv
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class BBoxLabelApp:
    def __init__(self, video_path: str, out_dir: str, sample_every_s: float, min_time: float):
        self.video_path = str(video_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.sample_every_s = float(sample_every_s)
        self.min_time = float(min_time)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.rows = []
        # Last bbox in original image coords (x1, y1, x2, y2) or None
        self.last_bbox = None

        self.target_t = max(0.0, self.min_time)
        self.sample_idx = 0

        # Current frame data
        self.current_frame = None
        self.current_t = None
        self.orig_w = 0
        self.orig_h = 0
        self.scale = 1.0

        # Bbox drawing state (in canvas/display coords)
        self.bbox = None  # (x1, y1, x2, y2) in display coords
        self.drag_start = None
        self.drag_mode = None  # 'create', 'move', 'resize_tl', 'resize_br', etc.
        self.drag_offset = (0, 0)

        # ---- UI ----
        self.root = tk.Tk()
        self.root.title("Bounding Box Labeler")

        self.root.protocol("WM_DELETE_WINDOW", self.on_stop)
        self.root.bind("<Escape>", lambda _e: self.on_stop())
        self.root.bind("<Return>", lambda _e: self.on_submit())

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.info = ttk.Label(
            top,
            text="Draw bbox: click+drag. Adjust: drag corners/edges/center. (Enter=Submit, Esc=Stop)",
            justify="left",
        )
        self.info.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.meta = ttk.Label(top, text="", justify="left")
        self.meta.grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.bbox_info = ttk.Label(top, text="BBox: None", justify="left")
        self.bbox_info.grid(row=2, column=0, sticky="w", pady=(0, 8))

        # Canvas for image + bbox drawing
        self.canvas = tk.Canvas(top, cursor="cross")
        self.canvas.grid(row=3, column=0, pady=(0, 10))

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Buttons
        btns = ttk.Frame(top)
        btns.grid(row=4, column=0, pady=(12, 0))

        ttk.Button(btns, text="Submit", command=self.on_submit, width=12).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Pass (no scoreboard)", command=self.on_pass, width=18).grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="Clear", command=self.on_clear, width=12).grid(row=0, column=2, padx=6)
        ttk.Button(btns, text="Stop", command=self.on_stop, width=12).grid(row=0, column=3, padx=6)

        self._photo = None
        self.load_next_sample()

    def seek_and_read(self, t_s: float):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        actual_t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return frame, actual_t

    def load_next_sample(self):
        frame, actual_t = self.seek_and_read(self.target_t)
        if frame is None:
            self.finish_and_quit()
            return

        self.current_frame = frame
        self.current_t = actual_t
        self.orig_h, self.orig_w = frame.shape[:2]

        self.meta.config(text=f"sample #{self.sample_idx}   target_t={self.target_t:.2f}s   actual_t={actual_t:.2f}s")

        # Convert BGR -> RGB -> PhotoImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Scale down if huge
        max_w = 1100
        if img.width > max_w:
            self.scale = max_w / img.width
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)))
        else:
            self.scale = 1.0

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Initialize bbox from last_bbox (convert to display coords)
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            self.bbox = (
                x1 * self.scale,
                y1 * self.scale,
                x2 * self.scale,
                y2 * self.scale,
            )
        else:
            self.bbox = None

        self.draw_bbox()
        self.root.after(50, lambda: self.root.focus_force())

    def draw_bbox(self):
        self.canvas.delete("bbox")
        self.canvas.delete("handles")

        if self.bbox is None:
            self.bbox_info.config(text="BBox: None (draw one or press Pass)")
            return

        x1, y1, x2, y2 = self.bbox
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.bbox = (x1, y1, x2, y2)

        # Draw rectangle
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=2, tags="bbox")

        # Draw corner handles
        handle_size = 8
        handles = [
            (x1, y1), (x2, y1),  # top-left, top-right
            (x1, y2), (x2, y2),  # bottom-left, bottom-right
        ]
        for hx, hy in handles:
            self.canvas.create_rectangle(
                hx - handle_size // 2, hy - handle_size // 2,
                hx + handle_size // 2, hy + handle_size // 2,
                fill="lime", outline="darkgreen", tags="handles"
            )

        # Show bbox in original coords
        orig_bbox = self.display_to_orig(self.bbox)
        self.bbox_info.config(
            text=f"BBox: x1={orig_bbox[0]:.0f}, y1={orig_bbox[1]:.0f}, x2={orig_bbox[2]:.0f}, y2={orig_bbox[3]:.0f}"
        )

    def display_to_orig(self, bbox):
        """Convert display coords to original image coords."""
        x1, y1, x2, y2 = bbox
        return (x1 / self.scale, y1 / self.scale, x2 / self.scale, y2 / self.scale)

    def get_drag_mode(self, x, y):
        """Determine what part of the bbox was clicked."""
        if self.bbox is None:
            return "create"

        x1, y1, x2, y2 = self.bbox
        threshold = 12

        # Check corners first
        if abs(x - x1) < threshold and abs(y - y1) < threshold:
            return "resize_tl"
        if abs(x - x2) < threshold and abs(y - y1) < threshold:
            return "resize_tr"
        if abs(x - x1) < threshold and abs(y - y2) < threshold:
            return "resize_bl"
        if abs(x - x2) < threshold and abs(y - y2) < threshold:
            return "resize_br"

        # Check edges
        if abs(x - x1) < threshold and y1 < y < y2:
            return "resize_l"
        if abs(x - x2) < threshold and y1 < y < y2:
            return "resize_r"
        if abs(y - y1) < threshold and x1 < x < x2:
            return "resize_t"
        if abs(y - y2) < threshold and x1 < x < x2:
            return "resize_b"

        # Check inside (move)
        if x1 < x < x2 and y1 < y < y2:
            return "move"

        # Outside - create new
        return "create"

    def on_mouse_down(self, event):
        x, y = event.x, event.y
        self.drag_mode = self.get_drag_mode(x, y)
        self.drag_start = (x, y)

        if self.drag_mode == "create":
            self.bbox = (x, y, x, y)
        elif self.drag_mode == "move" and self.bbox:
            x1, y1, x2, y2 = self.bbox
            self.drag_offset = (x - x1, y - y1)

    def on_mouse_drag(self, event):
        x, y = event.x, event.y

        if self.drag_mode == "create":
            x1, y1 = self.drag_start
            self.bbox = (x1, y1, x, y)

        elif self.drag_mode == "move" and self.bbox:
            ox, oy = self.drag_offset
            x1, y1, x2, y2 = self.bbox
            w, h = x2 - x1, y2 - y1
            new_x1 = x - ox
            new_y1 = y - oy
            self.bbox = (new_x1, new_y1, new_x1 + w, new_y1 + h)

        elif self.drag_mode and self.bbox:
            x1, y1, x2, y2 = self.bbox
            if "l" in self.drag_mode:
                x1 = x
            if "r" in self.drag_mode:
                x2 = x
            if "t" in self.drag_mode:
                y1 = y
            if "b" in self.drag_mode:
                y2 = y
            self.bbox = (x1, y1, x2, y2)

        self.draw_bbox()

    def on_mouse_up(self, event):
        self.drag_mode = None
        self.drag_start = None
        self.draw_bbox()

    def on_submit(self):
        if self.bbox is None:
            self.bbox_info.config(text="BBox: None - draw a box or press Pass!")
            return

        orig_bbox = self.display_to_orig(self.bbox)
        self.write_row(orig_bbox, update_autofill=True)
        self.advance()

    def on_pass(self):
        self.write_row(None, update_autofill=False)
        self.advance()

    def on_clear(self):
        self.bbox = None
        self.draw_bbox()

    def on_stop(self):
        self.finish_and_quit()

    def write_row(self, bbox, update_autofill: bool):
        t = float(self.current_t)
        img_path = self.out_dir / f"t{t:07.2f}.png"
        cv2.imwrite(str(img_path), self.current_frame)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            row = {
                "image_path": str(img_path),
                "t": f"{t:.3f}",
                "x1": f"{x1:.1f}",
                "y1": f"{y1:.1f}",
                "x2": f"{x2:.1f}",
                "y2": f"{y2:.1f}",
                "has_scoreboard": "1",
            }
            if update_autofill:
                self.last_bbox = (x1, y1, x2, y2)
        else:
            row = {
                "image_path": str(img_path),
                "t": f"{t:.3f}",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "has_scoreboard": "0",
            }

        self.rows.append(row)

    def advance(self):
        self.sample_idx += 1
        self.target_t = max(self.min_time, self.target_t + self.sample_every_s)
        self.load_next_sample()

    def finish_and_quit(self):
        try:
            self.cap.release()
        except Exception:
            pass

        out_csv = self.out_dir / "bbox_labels.csv"
        fieldnames = ["image_path", "t", "x1", "y1", "x2", "y2", "has_scoreboard"]
        with open(out_csv, "w+", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)

        print(f"Wrote {len(self.rows)} rows to {out_csv}")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(video_path: str, out_dir: str, sample_every_s: float = 60.0, min_time: float = 0.0):
    app = BBoxLabelApp(video_path, out_dir, sample_every_s, min_time)
    app.run()


if __name__ == "__main__":
    main(
        video_path="/Users/liamflynn/tennis/match1.mp4",
        out_dir="/Users/liamflynn/tennis/match1_bbox",
        sample_every_s=60.0,
        min_time=0.0,
    )
