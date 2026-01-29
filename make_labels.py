import csv
import re
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


VALID_INT_RE = re.compile(r"^\d{1,2}$")  # 0..99


def validate_cell(s: str) -> str:
    s = (s or "").strip()
    if s == "":
        return ""
    if s.upper() == "AD":
        return "AD"
    if VALID_INT_RE.fullmatch(s):
        return s
    raise ValueError("Each cell must be blank, 'AD', or a 1–2 digit number (0–99).")


def build_label_str(entries12):
    p1 = entries12[:6]
    p2 = entries12[6:]
    return f"p1:{','.join(p1)}|p2:{','.join(p2)}"


class LabelApp:
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
        self.last_entries12 = [""] * 12  # autofill (updated on Submit only)

        # We will seek to target timestamps instead of decoding the whole video
        self.target_t = max(0.0, self.min_time)
        self.sample_idx = 0

        # ---- UI ----
        self.root = tk.Tk()
        self.root.title("Frame Labeler")

        self.root.protocol("WM_DELETE_WINDOW", self.on_stop)
        self.root.bind("<Escape>", lambda _e: self.on_stop())
        self.root.bind("<Return>", lambda _e: self.on_submit())

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.info = ttk.Label(
            top,
            text="Allowed per cell: blank (null), AD, or 0–99.  (Enter=Submit, Esc=Stop)",
            justify="left",
        )
        self.info.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

        self.meta = ttk.Label(top, text="", justify="left")
        self.meta.grid(row=1, column=0, columnspan=6, sticky="w", pady=(0, 8))

        # Image panel
        self.image_label = ttk.Label(top)
        self.image_label.grid(row=2, column=0, columnspan=6, pady=(0, 10))

        # Entry grids
        ttk.Label(top, text="Player 1", font=("Arial", 11, "bold")).grid(row=3, column=0, columnspan=6, pady=(0, 4))
        self.vars = []
        for i in range(6):
            v = tk.StringVar(value=self.last_entries12[i])
            self.vars.append(v)
            e = ttk.Entry(top, textvariable=v, width=7, justify="center")
            e.grid(row=4, column=i, padx=4, pady=2)

        ttk.Label(top, text="Player 2", font=("Arial", 11, "bold")).grid(row=5, column=0, columnspan=6, pady=(8, 4))
        for i in range(6):
            v = tk.StringVar(value=self.last_entries12[6 + i])
            self.vars.append(v)
            e = ttk.Entry(top, textvariable=v, width=7, justify="center")
            e.grid(row=6, column=i, padx=4, pady=2)

        # Buttons
        btns = ttk.Frame(top)
        btns.grid(row=7, column=0, columnspan=6, pady=(12, 0))

        ttk.Button(btns, text="Submit", command=self.on_submit, width=12).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Pass", command=self.on_pass, width=12).grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="Stop", command=self.on_stop, width=12).grid(row=0, column=2, padx=6)

        # Keep a reference to the PhotoImage so it doesn't get GC'd
        self._photo = None

        # Load first sample
        self.current_frame = None
        self.current_t = None
        self.load_next_sample()

    def seek_and_read(self, t_s: float):
        # Seek near t_s (note: seeking isn't always frame-perfect)
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

        self.meta.config(text=f"sample #{self.sample_idx}   target_t={self.target_t:.2f}s   actual_t={actual_t:.2f}s")

        # Convert BGR -> RGB -> PhotoImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Optional: scale down if huge
        max_w = 1100
        if img.width > max_w:
            scale = max_w / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))

        self._photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._photo)

        # Autofill entries from last_entries12
        for i, v in enumerate(self.vars):
            v.set(self.last_entries12[i])

        # Focus first box
        self.root.after(50, lambda: self.root.focus_force())

    def on_submit(self):
        try:
            entries12 = [validate_cell(v.get()) for v in self.vars]
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.root)
            return

        self.write_row(entries12, update_autofill=True)
        self.advance()

    def on_pass(self):
        # all nulls
        entries12 = [""] * 12
        self.write_row(entries12, update_autofill=False)
        self.advance()

    def on_stop(self):
        self.finish_and_quit()

    def write_row(self, entries12, update_autofill: bool):
        t = float(self.current_t)
        img_path = self.out_dir / f"t{t:07.2f}.png"
        cv2.imwrite(str(img_path), self.current_frame)

        row = {
            "image_path": str(img_path),
            "t": f"{t:.3f}",
            "label": build_label_str(entries12),
        }
        for i in range(6):
            row[f"p1_{i+1}"] = entries12[i]
        for i in range(6):
            row[f"p2_{i+1}"] = entries12[6 + i]
        self.rows.append(row)

        if update_autofill:
            self.last_entries12 = entries12[:]

    def advance(self):
        self.sample_idx += 1
        self.target_t = max(self.min_time, self.target_t + self.sample_every_s)
        self.load_next_sample()

    def finish_and_quit(self):
        try:
            self.cap.release()
        except Exception:
            pass

        out_csv = self.out_dir / "labels.csv"
        fieldnames = (
            ["image_path", "t", "label"]
            + [f"p1_{i}" for i in range(1, 7)]
            + [f"p2_{i}" for i in range(1, 7)]
        )
        with open(out_csv, "w+", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)

        print(f"Wrote {len(self.rows)} rows to {out_csv}")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(video_path: str, out_dir: str, sample_every_s: float = 300.0, min_time: float = 60.0):
    app = LabelApp(video_path, out_dir, sample_every_s, min_time)
    app.run()


if __name__ == "__main__":
    main(
        video_path="/Users/liamflynn/tennis/match1.mp4",
        out_dir="/Users/liamflynn/tennis/match1",
        sample_every_s=60.0,
        min_time=0.0,
    )