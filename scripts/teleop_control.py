"""Recording control panel with live camera preview and episode log."""
import io
import math
import struct
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

from pynput.keyboard import Controller, Key

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_keyboard = Controller()

_SHM_CAMERAS = [
    ("Left Wrist",  "/dev/shm/cam_right_wrist.jpg"),
    ("Topdown",     "/dev/shm/cam_topdown.jpg"),
    ("Right Wrist", "/dev/shm/cam_left_wrist.jpg"),
]
_PREVIEW_W      = 280
_PREVIEW_H      = 210
_REFRESH_MS     = 100
_MIN_EPISODE_S  = 5
_REPLAY_BUFFER  = 30   # frames buffered for replay (~3 s at 10 fps)
_REPLAY_STEP_MS = 80   # playback interval


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _play_tone(freq: float, duration_ms: int) -> None:
    sample_rate = 44100
    n = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f'<{n}h', *(
            int(32767 * 0.4 * math.sin(2 * math.pi * freq * t / sample_rate))
            for t in range(n)
        )))
    data = buf.getvalue()
    def _run():
        try:
            p = subprocess.Popen(['aplay', '-q', '-'], stdin=subprocess.PIPE)
            p.communicate(input=data)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


_fps_font: ImageFont.ImageFont | None = None

def _get_fps_font():
    global _fps_font
    if _fps_font is None:
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]:
            try:
                _fps_font = ImageFont.truetype(path, 13)
                break
            except Exception:
                pass
        if _fps_font is None:
            _fps_font = ImageFont.load_default()
    return _fps_font


class TeleopControlApp:
    def __init__(self) -> None:
        self.episodes:      list[dict]       = []
        self.episode_start: float            = time.time()
        self.cam_labels:    list[tk.Label]   = []
        self.cam_images:    list             = []
        self.frame_times:   list[deque]      = [deque(maxlen=10) for _ in _SHM_CAMERAS]
        self.frame_buffers: list[deque]      = [deque(maxlen=_REPLAY_BUFFER) for _ in _SHM_CAMERAS]
        self.replay_frames: list[list] | None = None
        self.replay_idx:    int              = 0
        self.pending_save:  bool             = False
        # widgets assigned in run()
        self.root:         tk.Tk
        self.save_btn:     tk.Button
        self.timer_label:  tk.Label
        self.warn_label:   tk.Label
        self.stats_label:  tk.Label
        self.log_text:     tk.Text

    # ── actions ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        duration = time.time() - self.episode_start
        if duration < _MIN_EPISODE_S and not self.pending_save:
            self.pending_save = True
            self.save_btn.config(bg="#fab387")
            self.warn_label.config(
                text=f"Short episode (<{_MIN_EPISODE_S}s) — press Save again to confirm"
            )
            self.root.after(2500, self._clear_warn)
            return
        self.pending_save = False
        self.warn_label.config(text="")
        self.save_btn.config(bg="#a6e3a1")
        self.replay_frames = [list(buf) for buf in self.frame_buffers]
        self.replay_idx = 0
        self._record(saved=True)
        _play_tone(880, 140)
        _keyboard.press(Key.right)
        _keyboard.release(Key.right)

    def _discard(self) -> None:
        self.pending_save = False
        self._clear_warn()
        self._record(saved=False)
        _play_tone(330, 220)
        _keyboard.press(Key.left)
        _keyboard.release(Key.left)

    def _stop(self) -> None:
        _keyboard.press(Key.esc)
        _keyboard.release(Key.esc)

    def _clear_warn(self) -> None:
        self.pending_save = False
        self.warn_label.config(text="")
        self.save_btn.config(bg="#a6e3a1")

    def _record(self, saved: bool) -> None:
        self.episodes.append({"duration": time.time() - self.episode_start, "saved": saved})
        self.episode_start = time.time()
        self._update_log()

    # ── log ───────────────────────────────────────────────────────────────

    def _update_log(self) -> None:
        n_saved     = sum(1 for e in self.episodes if e["saved"])
        n_discarded = len(self.episodes) - n_saved
        self.stats_label.config(text=f"{n_saved} saved   {n_discarded} discarded")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for i, ep in enumerate(self.episodes, 1):
            dur = _fmt(ep["duration"])
            if ep["saved"]:
                self.log_text.insert("end", f"#{i:>3}  {dur}  ✓ Saved\n", "saved")
            else:
                self.log_text.insert("end", f"#{i:>3}  {dur}  ✗ Discarded\n", "discarded")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ── timer ─────────────────────────────────────────────────────────────

    def _tick_timer(self) -> None:
        elapsed = time.time() - self.episode_start
        self.timer_label.config(
            text=_fmt(elapsed),
            fg="#f38ba8" if elapsed < _MIN_EPISODE_S else "#a6e3a1",
        )
        self.root.after(500, self._tick_timer)

    # ── camera frames ─────────────────────────────────────────────────────

    def _update_frames(self) -> None:
        if not _PIL_AVAILABLE:
            self.root.after(_REFRESH_MS, self._update_frames)
            return

        # replay mode: loop through buffered frames once then return to live
        if self.replay_frames is not None:
            max_len = max((len(f) for f in self.replay_frames), default=0)
            if max_len == 0 or self.replay_idx >= max_len:
                self.replay_frames = None
                self.replay_idx = 0
            else:
                for i, frames in enumerate(self.replay_frames):
                    if not frames:
                        continue
                    frame = frames[self.replay_idx % len(frames)]
                    photo = ImageTk.PhotoImage(frame)
                    self.cam_labels[i].configure(image=photo)
                    self.cam_images[i] = photo
                self.replay_idx += 1
                self.root.after(_REPLAY_STEP_MS, self._update_frames)
                return

        # live feed
        now = time.time()
        font = _get_fps_font()
        for i, (_, path) in enumerate(_SHM_CAMERAS):
            p = Path(path)
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                img.load()
                img = img.resize((_PREVIEW_W, _PREVIEW_H), Image.BILINEAR)

                # fps
                self.frame_times[i].append(now)
                times = self.frame_times[i]
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0
                fps_text = f"{fps:.0f} fps"
                color    = (100, 220, 100) if fps > 5 else (220, 80, 80)
                draw = ImageDraw.Draw(img)
                draw.rectangle([2, 2, 66, 18], fill=(0, 0, 0))
                draw.text((4, 3), fps_text, fill=color, font=font)

                self.frame_buffers[i].append(img.copy())

                photo = ImageTk.PhotoImage(img)
                self.cam_labels[i].configure(image=photo, width=_PREVIEW_W, height=_PREVIEW_H)
                self.cam_images[i] = photo
            except Exception:
                pass

        self.root.after(_REFRESH_MS, self._update_frames)

    # ── build ui ──────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root = tk.Tk()
        root = self.root
        root.title("Teleop Control")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        bold14 = tkfont.Font(family="Monospace", size=14, weight="bold")
        bold13 = tkfont.Font(family="Monospace", size=13, weight="bold")
        bold22 = tkfont.Font(family="Monospace", size=22, weight="bold")
        mono10 = tkfont.Font(family="Monospace", size=10)
        mono9  = tkfont.Font(family="Monospace", size=9)

        # cameras
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

        # title + live timer
        header = tk.Frame(root, bg="#1e1e2e")
        header.pack(pady=(10, 4))
        tk.Label(header, text="Recording Controls", bg="#1e1e2e", fg="#89b4fa",
                 font=bold14).pack(side="left", padx=(0, 20))
        self.timer_label = tk.Label(header, text="00:00", bg="#1e1e2e",
                                    fg="#a6e3a1", font=bold22)
        self.timer_label.pack(side="left")

        # buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(padx=20, pady=4)
        self.save_btn = tk.Button(btn_frame, text="✓  Save", width=11, font=bold13,
                                  bg="#a6e3a1", fg="#1e1e2e", activebackground="#a6e3a1",
                                  relief="flat", padx=8, pady=10, command=self._save)
        self.save_btn.grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="✗  Discard", width=11, font=bold13,
                  bg="#f38ba8", fg="#1e1e2e", activebackground="#f38ba8",
                  relief="flat", padx=8, pady=10,
                  command=self._discard).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text="■  Stop", width=11, font=bold13,
                  bg="#313244", fg="#cdd6f4", activebackground="#313244",
                  relief="flat", padx=8, pady=10,
                  command=self._stop).grid(row=0, column=2, padx=6)

        # warning + hint
        self.warn_label = tk.Label(root, text="", bg="#1e1e2e", fg="#fab387", font=mono9)
        self.warn_label.pack()
        tk.Label(root, text="Enter/→ Save   ← Discard   Esc Stop",
                 bg="#1e1e2e", fg="#585b70", font=mono9).pack(pady=(2, 4))

        # keyboard bindings
        root.bind("<Return>", lambda e: self._save())
        root.bind("<Right>",  lambda e: self._save())
        root.bind("<Left>",   lambda e: self._discard())
        root.bind("<Escape>", lambda e: self._stop())

        # episode log
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
        self.log_text.tag_configure("saved",     foreground="#a6e3a1")
        self.log_text.tag_configure("discarded", foreground="#f38ba8")
        scrollbar.config(command=self.log_text.yview)

        root.after(_REFRESH_MS, self._update_frames)
        root.after(500, self._tick_timer)
        root.mainloop()


def main() -> None:
    TeleopControlApp().run()


if __name__ == "__main__":
    main()
