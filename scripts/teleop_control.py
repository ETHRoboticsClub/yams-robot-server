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
_REPLAY_BUFFER  = 30
_REPLAY_STEP_MS = 80
_CAM_FRESH_S    = 2.0
_SPINNER        = ["|", "/", "—", "\\"]


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

def _get_fps_font() -> ImageFont.ImageFont:
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
        self.episodes:       list[dict]        = []
        self.episode_start:  float             = time.time()
        self.cam_labels:     list[tk.Label]    = []
        self.cam_images:     list              = []
        self.frame_times:    list[deque]       = [deque(maxlen=10) for _ in _SHM_CAMERAS]
        self.frame_buffers:  list[deque]       = [deque(maxlen=_REPLAY_BUFFER) for _ in _SHM_CAMERAS]
        self.replay_frames:  list[list] | None = None
        self.replay_idx:     int               = 0
        self._paused:        bool              = False
        self._last_action:   float             = 0.0   # debounce injected keys
        self._init_start:    float             = time.time()
        self._in_init:       bool              = True
        self._spinner_idx:   int               = 0

    # ── init screen ───────────────────────────────────────────────────────

    def _build_init_screen(self, root: tk.Tk) -> tk.Frame:
        bold20 = tkfont.Font(family="Monospace", size=20, weight="bold")
        bold14 = tkfont.Font(family="Monospace", size=14, weight="bold")
        bold28 = tkfont.Font(family="Monospace", size=28, weight="bold")
        mono11 = tkfont.Font(family="Monospace", size=11)
        mono10 = tkfont.Font(family="Monospace", size=10)

        frame = tk.Frame(root, bg="#1e1e2e")
        tk.Label(frame, text="YAMS Robot", bg="#1e1e2e", fg="#cdd6f4",
                 font=bold20).pack(pady=(40, 4))
        self.init_status_label = tk.Label(frame, text="Initializing...",
                                          bg="#1e1e2e", fg="#585b70", font=mono11)
        self.init_status_label.pack()
        self.init_timer_label = tk.Label(frame, text="00:00", bg="#1e1e2e",
                                         fg="#89b4fa", font=bold28)
        self.init_timer_label.pack(pady=(8, 24))

        cam_box = tk.Frame(frame, bg="#313244", padx=20, pady=16)
        cam_box.pack(padx=40, pady=(0, 40))
        tk.Label(cam_box, text="Camera Status", bg="#313244", fg="#89b4fa",
                 font=bold14).pack(anchor="w", pady=(0, 10))

        self.cam_status_labels: list[tk.Label] = []
        for name, _ in _SHM_CAMERAS:
            row = tk.Frame(cam_box, bg="#313244")
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"  {name}", bg="#313244", fg="#cdd6f4",
                     font=mono10, width=14, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="○  Waiting", bg="#313244", fg="#585b70", font=mono10)
            lbl.pack(side="right")
            self.cam_status_labels.append(lbl)

        return frame

    def _tick_init_timer(self) -> None:
        if not self._in_init:
            return
        elapsed = time.time() - self._init_start
        self.init_timer_label.config(text=_fmt(elapsed))
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
        self.init_status_label.config(
            text=f"{_SPINNER[self._spinner_idx]}  Waiting for cameras..."
        )
        self.root.after(500, self._tick_init_timer)

    def _check_init(self) -> None:
        if not self._in_init:
            return
        now = time.time()
        all_online = True
        for i, (_, path) in enumerate(_SHM_CAMERAS):
            p = Path(path)
            try:
                online = p.exists() and (now - p.stat().st_mtime) < _CAM_FRESH_S
            except Exception:
                online = False
            self.cam_status_labels[i].config(
                text="●  Online" if online else "○  Waiting",
                fg="#a6e3a1" if online else "#585b70",
            )
            if not online:
                all_online = False

        if all_online:
            self._transition_to_main()
        else:
            self.root.after(500, self._check_init)

    def _transition_to_main(self) -> None:
        self._in_init = False
        self.init_status_label.config(text="✓  All systems ready!", fg="#a6e3a1")
        self.init_timer_label.config(fg="#a6e3a1")
        _play_tone(523, 80)
        self.root.after(80,  lambda: _play_tone(659, 80))
        self.root.after(160, lambda: _play_tone(784, 120))
        self.root.after(900, self._show_main)

    def _show_main(self) -> None:
        self.init_frame.pack_forget()
        self.episode_start = time.time()
        self.main_frame.pack(fill="both", expand=True)
        self.root.after(500, self._tick_timer)
        self.root.after(_REFRESH_MS, self._update_frames)

    # ── actions ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._paused:
            return
        now = time.time()
        if now - self._last_action < 0.5:
            return
        self._last_action = now
        frames = [list(buf) for buf in self.frame_buffers]
        self.replay_frames = frames
        self.replay_idx = 0
        self._record(saved=True, frames=frames)
        _play_tone(880, 140)
        _keyboard.press(Key.right)
        _keyboard.release(Key.right)

    def _discard(self) -> None:
        if self._paused:
            return
        now = time.time()
        if now - self._last_action < 0.5:
            return
        self._last_action = now
        self._record(saved=False, frames=None)
        _play_tone(330, 220)
        _keyboard.press(Key.left)
        _keyboard.release(Key.left)

    def _stop(self) -> None:
        now = time.time()
        if now - self._last_action < 0.5:
            return
        self._last_action = now
        _keyboard.press(Key.esc)
        _keyboard.release(Key.esc)

    def _pause(self) -> None:
        self._paused = True
        self.pause_btn.config(text="▶  Resume", bg="#89b4fa", command=self._resume)
        self.save_btn.config(state="disabled")
        self.discard_btn.config(state="disabled")
        self.timer_label.config(fg="#585b70")
        self.log_outer.pack_forget()
        self.traj_frame.pack(padx=12, pady=(4, 12), fill="x")
        self._refresh_traj_list()

    def _resume(self) -> None:
        self._paused = False
        self.replay_frames = None
        self.pause_btn.config(text="⏸  Pause", bg="#313244", command=self._pause)
        self.save_btn.config(state="normal")
        self.discard_btn.config(state="normal")
        self.episode_start = time.time()
        self.traj_frame.pack_forget()
        self.log_outer.pack(padx=12, pady=(4, 12), fill="x")

    def _record(self, saved: bool, frames: list[list] | None) -> None:
        self.episodes.append({
            "duration": time.time() - self.episode_start,
            "saved": saved,
            "frames": frames,
        })
        self.episode_start = time.time()
        self._update_log()

    # ── trajectory browser ────────────────────────────────────────────────

    def _refresh_traj_list(self) -> None:
        for w in self.traj_inner.winfo_children():
            w.destroy()
        mono10 = tkfont.Font(family="Monospace", size=10)
        saved = [(i, ep) for i, ep in enumerate(self.episodes) if ep["saved"]]
        for ep_idx, ep in saved:
            dur = _fmt(ep["duration"])
            def _play(idx: int = ep_idx) -> None:
                self.replay_frames = self.episodes[idx]["frames"]
                self.replay_idx = 0
            tk.Button(
                self.traj_inner,
                text=f"  #{ep_idx + 1:>3}  {dur}  ▶ Play",
                bg="#181825", fg="#a6e3a1", activebackground="#313244",
                font=mono10, relief="flat", anchor="w",
                command=_play,
            ).pack(fill="x", padx=4, pady=2)
        self.traj_canvas.configure(scrollregion=self.traj_canvas.bbox("all"))

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
        if not self._paused:
            elapsed = time.time() - self.episode_start
            self.timer_label.config(text=_fmt(elapsed), fg="#a6e3a1")
        self.root.after(500, self._tick_timer)

    # ── camera frames ─────────────────────────────────────────────────────

    def _update_frames(self) -> None:
        if not _PIL_AVAILABLE:
            self.root.after(_REFRESH_MS, self._update_frames)
            return

        if self.replay_frames is not None:
            max_len = max((len(f) for f in self.replay_frames), default=0)
            if max_len == 0:
                self.replay_frames = None
                self.replay_idx = 0
            else:
                if self.replay_idx >= max_len:
                    if self._paused:
                        self.replay_idx = 0  # loop while paused
                    else:
                        self.replay_frames = None
                        self.replay_idx = 0
                if self.replay_frames is not None:
                    for i, frames in enumerate(self.replay_frames):
                        if not frames:
                            continue
                        photo = ImageTk.PhotoImage(frames[self.replay_idx % len(frames)])
                        self.cam_labels[i].configure(image=photo)
                        self.cam_images[i] = photo
                    self.replay_idx += 1
                    self.root.after(_REPLAY_STEP_MS, self._update_frames)
                    return

        now  = time.time()
        font = _get_fps_font()
        for i, (_, path) in enumerate(_SHM_CAMERAS):
            p = Path(path)
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                img.load()
                img = img.resize((_PREVIEW_W, _PREVIEW_H), Image.BILINEAR)
                self.frame_times[i].append(now)
                times = self.frame_times[i]
                fps   = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0
                color = (100, 220, 100) if fps > 5 else (220, 80, 80)
                draw  = ImageDraw.Draw(img)
                draw.rectangle([2, 2, 66, 18], fill=(0, 0, 0))
                draw.text((4, 3), f"{fps:.0f} fps", fill=color, font=font)
                self.frame_buffers[i].append(img.copy())
                photo = ImageTk.PhotoImage(img)
                self.cam_labels[i].configure(image=photo, width=_PREVIEW_W, height=_PREVIEW_H)
                self.cam_images[i] = photo
            except Exception:
                pass

        self.root.after(_REFRESH_MS, self._update_frames)

    # ── build main screen ─────────────────────────────────────────────────

    def _build_main_screen(self, root: tk.Tk) -> tk.Frame:
        bold14 = tkfont.Font(family="Monospace", size=14, weight="bold")
        bold13 = tkfont.Font(family="Monospace", size=13, weight="bold")
        bold22 = tkfont.Font(family="Monospace", size=22, weight="bold")
        mono10 = tkfont.Font(family="Monospace", size=10)
        mono9  = tkfont.Font(family="Monospace", size=9)

        frame = tk.Frame(root, bg="#1e1e2e")

        # cameras
        cam_frame = tk.Frame(frame, bg="#1e1e2e")
        cam_frame.pack(padx=12, pady=(12, 4))
        for i, (name, _) in enumerate(_SHM_CAMERAS):
            col = tk.Frame(cam_frame, bg="#1e1e2e")
            col.grid(row=0, column=i, padx=6)
            tk.Label(col, text=name, bg="#1e1e2e", fg="#89b4fa", font=mono9).pack()
            lbl = tk.Label(col, bg="#313244", width=_PREVIEW_W, height=_PREVIEW_H)
            lbl.pack()
            self.cam_labels.append(lbl)
            self.cam_images.append(None)

        # title + timer
        header = tk.Frame(frame, bg="#1e1e2e")
        header.pack(pady=(10, 4))
        tk.Label(header, text="Recording Controls", bg="#1e1e2e", fg="#89b4fa",
                 font=bold14).pack(side="left", padx=(0, 20))
        self.timer_label = tk.Label(header, text="00:00", bg="#1e1e2e",
                                    fg="#a6e3a1", font=bold22)
        self.timer_label.pack(side="left")

        # buttons
        btn_frame = tk.Frame(frame, bg="#1e1e2e")
        btn_frame.pack(padx=20, pady=4)
        self.save_btn = tk.Button(btn_frame, text="✓  Save", width=11, font=bold13,
                                  bg="#a6e3a1", fg="#1e1e2e", activebackground="#a6e3a1",
                                  relief="flat", padx=8, pady=10, command=self._save)
        self.save_btn.grid(row=0, column=0, padx=6)
        self.discard_btn = tk.Button(btn_frame, text="✗  Discard", width=11, font=bold13,
                                     bg="#f38ba8", fg="#1e1e2e", activebackground="#f38ba8",
                                     relief="flat", padx=8, pady=10, command=self._discard)
        self.discard_btn.grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text="■  Stop", width=11, font=bold13,
                  bg="#313244", fg="#cdd6f4", activebackground="#313244",
                  relief="flat", padx=8, pady=10,
                  command=self._stop).grid(row=0, column=2, padx=6)
        self.pause_btn = tk.Button(btn_frame, text="⏸  Pause", width=11, font=bold13,
                                   bg="#313244", fg="#cdd6f4", activebackground="#313244",
                                   relief="flat", padx=8, pady=10, command=self._pause)
        self.pause_btn.grid(row=0, column=3, padx=6)

        tk.Label(frame, text="Enter/→ Save   ← Discard   Esc Stop",
                 bg="#1e1e2e", fg="#585b70", font=mono9).pack(pady=(6, 4))

        # episode log (normal mode)
        self.log_outer = tk.Frame(frame, bg="#1e1e2e")
        self.log_outer.pack(padx=12, pady=(4, 12), fill="x")
        self.stats_label = tk.Label(self.log_outer, text="0 saved   0 discarded",
                                    bg="#1e1e2e", fg="#a6e3a1", font=mono10, anchor="w")
        self.stats_label.pack(fill="x")
        scrollbar = tk.Scrollbar(self.log_outer)
        scrollbar.pack(side="right", fill="y")
        self.log_text = tk.Text(self.log_outer, height=5, width=38, bg="#181825",
                                fg="#cdd6f4", font=mono10, state="disabled", relief="flat",
                                yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="x")
        self.log_text.tag_configure("saved",     foreground="#a6e3a1")
        self.log_text.tag_configure("discarded", foreground="#f38ba8")
        scrollbar.config(command=self.log_text.yview)

        # trajectory browser (pause mode, hidden by default)
        self.traj_frame = tk.Frame(frame, bg="#1e1e2e")
        tk.Label(self.traj_frame, text="Saved Trajectories", bg="#1e1e2e",
                 fg="#89b4fa", font=bold13, anchor="w").pack(fill="x")
        traj_scroll = tk.Scrollbar(self.traj_frame)
        traj_scroll.pack(side="right", fill="y")
        self.traj_canvas = tk.Canvas(self.traj_frame, bg="#181825", height=140,
                                     highlightthickness=0,
                                     yscrollcommand=traj_scroll.set)
        self.traj_canvas.pack(side="left", fill="x", expand=True)
        traj_scroll.config(command=self.traj_canvas.yview)
        self.traj_inner = tk.Frame(self.traj_canvas, bg="#181825")
        self.traj_canvas.create_window((0, 0), window=self.traj_inner, anchor="nw")

        return frame

    # ── entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        self.root = tk.Tk()
        root = self.root
        root.title("Teleop Control")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        root.bind("<Return>", lambda e: self._save())
        root.bind("<Right>",  lambda e: self._save())
        root.bind("<Left>",   lambda e: self._discard())
        root.bind("<Escape>", lambda e: self._stop())

        self.init_frame = self._build_init_screen(root)
        self.main_frame = self._build_main_screen(root)
        self.init_frame.pack(fill="both", expand=True)

        root.after(500, self._tick_init_timer)
        root.after(500, self._check_init)
        root.mainloop()


def main() -> None:
    TeleopControlApp().run()


if __name__ == "__main__":
    main()
