"""Minimal recording control panel — Save / Discard / Stop buttons."""
import tkinter as tk
from tkinter import font as tkfont

from pynput.keyboard import Controller, Key

_keyboard = Controller()


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

    tk.Label(root, text="Recording Controls", bg="#1e1e2e", fg="#89b4fa",
             font=tkfont.Font(family="Monospace", size=14, weight="bold")).pack(pady=(14, 10))

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

    root.mainloop()


if __name__ == "__main__":
    main()
