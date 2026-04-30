"""Recording control panel with live camera preview."""
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
    ("Right Wrist", "/dev/shm/cam_right_wrist.jpg"),
    ("Left Wrist",  "/dev/shm/cam_left_wrist.jpg"),
]
_PREVIEW_W = 320
_PREVIEW_H = 240
_REFRESH_MS = 100


def _inject(key) -> None:
    _keyboard.press(key)
    _keyboard.release(key)


def main() -> None:
    root = tk.Tk()
    root.title("Teleop Control")
    root.configure(bg="#1e1e2e")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    f = tkfont.Font(family="Monospace", size=13, weight="bold")
    hint = tkfont.Font(family="Monospace", size=9)
    small = tkfont.Font(family="Monospace", size=9)

    cam_frame = tk.Frame(root, bg="#1e1e2e")
    cam_frame.pack(padx=12, pady=(12, 4))

    cam_labels = []
    cam_images = []
    for i, (name, _) in enumerate(_SHM_CAMERAS):
        col = tk.Frame(cam_frame, bg="#1e1e2e")
        col.grid(row=0, column=i, padx=6)
        tk.Label(col, text=name, bg="#1e1e2e", fg="#89b4fa", font=small).pack()
        lbl = tk.Label(col, bg="#313244", width=_PREVIEW_W, height=_PREVIEW_H)
        lbl.pack()
        cam_labels.append(lbl)
        cam_images.append(None)

    tk.Label(root, text="Recording Controls", bg="#1e1e2e", fg="#89b4fa",
             font=tkfont.Font(family="Monospace", size=14, weight="bold")).pack(pady=(10, 6))

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(padx=20, pady=4)

    tk.Button(btn_frame, text="✓  Save", width=11, font=f,
              bg="#a6e3a1", fg="#1e1e2e", activebackground="#a6e3a1",
              relief="flat", padx=8, pady=10,
              command=lambda: _inject(Key.right)).grid(row=0, column=0, padx=6)

    tk.Button(btn_frame, text="✗  Discard", width=11, font=f,
              bg="#f38ba8", fg="#1e1e2e", activebackground="#f38ba8",
              relief="flat", padx=8, pady=10,
              command=lambda: _inject(Key.left)).grid(row=0, column=1, padx=6)

    tk.Button(btn_frame, text="■  Stop", width=11, font=f,
              bg="#313244", fg="#cdd6f4", activebackground="#313244",
              relief="flat", padx=8, pady=10,
              command=lambda: _inject(Key.esc)).grid(row=0, column=2, padx=6)

    tk.Label(root, text="→ Save   ← Discard   Esc Stop",
             bg="#1e1e2e", fg="#585b70", font=hint).pack(pady=(8, 14))

    if not _PIL_AVAILABLE:
        def update_frames():
            root.after(_REFRESH_MS, update_frames)
    else:
        def update_frames():
            for i, (_, path) in enumerate(_SHM_CAMERAS):
                p = Path(path)
                if p.exists():
                    try:
                        img = Image.open(p)
                        img = img.resize((_PREVIEW_W, _PREVIEW_H), Image.BILINEAR)
                        photo = ImageTk.PhotoImage(img)
                        cam_labels[i].configure(image=photo, width=_PREVIEW_W, height=_PREVIEW_H)
                        cam_images[i] = photo
                    except Exception:
                        pass
            root.after(_REFRESH_MS, update_frames)

    root.after(_REFRESH_MS, update_frames)
    root.mainloop()


if __name__ == "__main__":
    main()
