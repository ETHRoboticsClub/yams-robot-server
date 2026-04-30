"""Recording control panel with live camera preview and episode log."""
import time
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path

from pynput.keyboard import Controller, Key

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_keyboard = Controller()

_SHM_CAMERAS = [
    ("Left Wrist",  "/dev/shm/cam_right_wrist.jpg"),
    ("Topdown",     "/dev/shm/cam_topdown.jpg"),
    ("Right Wrist", "/dev/shm/cam_left_wrist.jpg"),
]
_PREVIEW_W = 280
_PREVIEW_H = 210
_REFRESH_MS = 100


def _fmt_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


class TeleopControlApp:
    def __init__(self) -> None:
        self.episodes: list[dict] = []
        self.episode_start = time.time()
        self.cam_labels: list[tk.Label] = []
        self.cam_images: list = []

    def _inject(self, key, saved: bool | None = None) -> None:
        if saved is not None:
            duration = time.time() - self.episode_start
            self.episodes.append({"duration": duration, "saved": saved})
            self.episode_start = time.time()
            self._update_log()
        _keyboard.press(key)
        _keyboard.release(key)

    def _update_log(self) -> None:
        saved_count = sum(1 for e in self.episodes if e["saved"])
        discarded_count = len(self.episodes) - saved_count
        self.stats_label.config(text=f"{saved_count} saved   {discarded_count} discarded")

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for i, ep in enumerate(self.episodes, 1):
            dur = _fmt_duration(ep["duration"])
            if ep["saved"]:
                self.log_text.insert("end", f"#{i:>3}  {dur}  ✓ Saved\n", "saved")
            else:
                self.log_text.insert("end", f"#{i:>3}  {dur}  ✗ Discarded\n", "discarded")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _update_frames(self) -> None:
        if _PIL_AVAILABLE:
            for i, (_, path) in enumerate(_SHM_CAMERAS):
                p = Path(path)
                if p.exists():
                    try:
                        img = Image.open(p)
                        img = img.resize((_PREVIEW_W, _PREVIEW_H), Image.BILINEAR)
                        photo = ImageTk.PhotoImage(img)
                        self.cam_labels[i].configure(image=photo, width=_PREVIEW_W, height=_PREVIEW_H)
                        self.cam_images[i] = photo
                    except Exception:
                        pass
        self.root.after(_REFRESH_MS, self._update_frames)

    def run(self) -> None:
        self.root = tk.Tk()
        root = self.root
        root.title("Teleop Control")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        bold14 = tkfont.Font(family="Monospace", size=14, weight="bold")
        bold13 = tkfont.Font(family="Monospace", size=13, weight="bold")
        mono10 = tkfont.Font(family="Monospace", size=10)
        mono9  = tkfont.Font(family="Monospace", size=9)

        # Camera previews
        cam_frame = tk.Frame(root, bg="#1e1e2e")
        cam_frame.pack(padx=12, pady=(12, 4))
        for i, (name, _) in enumerate(_SHM_CAMERAS):
            col = tk.Frame(cam_frame, bg="#1e1e2e")
            col.grid(row=0, column=i, padx=6)
            tk.Label(col, text=name, bg="#1e1e2e", fg="#89b4fa", font=mono9).pack()
            lbl = tk.Label(col, bg="#313244", width=_PREVIEW_W, height=_PREVIEW_H)
            lbl.pack()
            self.cam_labels.append(lbl)
            self.cam_images.append(None)

        # Title
        tk.Label(root, text="Recording Controls", bg="#1e1e2e", fg="#89b4fa",
                 font=bold14).pack(pady=(10, 6))

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(padx=20, pady=4)
        tk.Button(btn_frame, text="✓  Save", width=11, font=bold13,
                  bg="#a6e3a1", fg="#1e1e2e", activebackground="#a6e3a1",
                  relief="flat", padx=8, pady=10,
                  command=lambda: self._inject(Key.right, saved=True)).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="✗  Discard", width=11, font=bold13,
                  bg="#f38ba8", fg="#1e1e2e", activebackground="#f38ba8",
                  relief="flat", padx=8, pady=10,
                  command=lambda: self._inject(Key.left, saved=False)).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text="■  Stop", width=11, font=bold13,
                  bg="#313244", fg="#cdd6f4", activebackground="#313244",
                  relief="flat", padx=8, pady=10,
                  command=lambda: self._inject(Key.esc)).grid(row=0, column=2, padx=6)

        tk.Label(root, text="→ Save   ← Discard   Esc Stop",
                 bg="#1e1e2e", fg="#585b70", font=mono9).pack(pady=(8, 4))

        # Episode log
        log_outer = tk.Frame(root, bg="#1e1e2e")
        log_outer.pack(padx=12, pady=(4, 12), fill="x")

        self.stats_label = tk.Label(log_outer, text="0 saved   0 discarded",
                                    bg="#1e1e2e", fg="#a6e3a1", font=mono10, anchor="w")
        self.stats_label.pack(fill="x")

        scrollbar = tk.Scrollbar(log_outer)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(log_outer, height=6, width=38, bg="#181825", fg="#cdd6f4",
                                font=mono10, state="disabled", relief="flat",
                                yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="x")
        self.log_text.tag_configure("saved", foreground="#a6e3a1")
        self.log_text.tag_configure("discarded", foreground="#f38ba8")
        scrollbar.config(command=self.log_text.yview)

        root.after(_REFRESH_MS, self._update_frames)
        root.mainloop()


def main() -> None:
    TeleopControlApp().run()


if __name__ == "__main__":
    main()
