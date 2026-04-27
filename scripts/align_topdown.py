"""Real-time topdown camera alignment TUI.

Streams the RealSense topdown live and shows roll/tx/ty drift vs a reference
image, updating a few times per second. Use while physically adjusting the
mount. When all three axes are in tolerance for a few consecutive reads,
prints a big CORRECT banner.

Reference source, in priority order:
  --reference PATH      any PNG
  --dataset   PATH      lerobot dataset root; averages first N frames of
                        observation.images.topdown
  --latest              picks most-recently-modified dataset under
                        ~/.cache/huggingface/lerobot/ETHRC/* that has a
                        topdown video
  (default)             outputs/camera_reference_images/topdown.png
                        (same file the check_setup gate uses)

Exit with Ctrl+C.
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.camera_pose import (  # noqa: E402
    TOL_ROLL_DEG,
    TOL_TX_PX,
    TOL_TY_PX,
    Pose,
    evaluate_pose,
)
from lerobot_camera_cached.camera_realsense_cached import RealSenseCameraCached  # noqa: E402
from lerobot_camera_cached.realsense_cached_config import RealSenseCameraCachedConfig  # noqa: E402

ARMS_CONFIG = ROOT / "configs" / "arms.yaml"
COMMITTED_REFERENCE = ROOT / "outputs" / "camera_reference_images" / "topdown.png"
LEROBOT_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "ETHRC"

# ANSI
CSI = "\033["
CLEAR_SCREEN = f"{CSI}2J"
HOME = f"{CSI}H"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
DIM = f"{CSI}2m"
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
CYAN = f"{CSI}36m"


def _resolve_reference(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.reference:
        return _load_png(Path(args.reference)), f"image: {args.reference}"
    if args.dataset:
        return _reference_from_dataset(Path(args.dataset)), f"dataset: {args.dataset}"
    if args.latest:
        dataset = _latest_dataset()
        return _reference_from_dataset(dataset), f"latest dataset: {dataset}"
    return _load_png(COMMITTED_REFERENCE), f"committed reference: {COMMITTED_REFERENCE}"


def _load_png(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"reference PNG not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"failed to decode {path}")
    return img


def _find_topdown_mp4(dataset_root: Path) -> Path:
    topdown_dir = dataset_root / "videos" / "observation.images.topdown"
    mp4s = sorted(topdown_dir.rglob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"no topdown mp4 under {topdown_dir}")
    return mp4s[0]


def _reference_from_dataset(dataset_root: Path, n_frames: int = 30) -> np.ndarray:
    mp4 = _find_topdown_mp4(dataset_root)
    cap = cv2.VideoCapture(str(mp4))
    try:
        frames: list[np.ndarray] = []
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            raise SystemExit(f"could not decode frames from {mp4}")
        return np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
    finally:
        cap.release()


def _latest_dataset() -> Path:
    candidates = [
        d for d in LEROBOT_ROOT.iterdir()
        if d.is_dir() and (d / "videos" / "observation.images.topdown").exists()
    ]
    if not candidates:
        raise SystemExit(f"no datasets with topdown videos under {LEROBOT_ROOT}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _start_realsense() -> tuple[RealSenseCameraCached, str]:
    with open(ARMS_CONFIG) as f:
        config_yaml = yaml.safe_load(f)
    cam = config_yaml["cameras"]["configs"]["topdown"]
    serial = str(cam["serial_number_or_name"])
    width = int(cam.get("width", 640))
    height = int(cam.get("height", 480))
    fps = int(cam.get("fps", 30))

    camera = RealSenseCameraCached(
        RealSenseCameraCachedConfig(
            serial_number_or_name=serial,
            fps=fps,
            width=width,
            height=height,
            use_depth=False,
            warmup_s=1.0,
        )
    )
    # Reuse the production busy-retry / profile-load / warmup path.
    camera.connect()
    return camera, f"{serial} @ {width}x{height}"


def _grab_frame(camera: RealSenseCameraCached) -> np.ndarray | None:
    try:
        return camera.async_read(timeout_ms=1000)
    except TimeoutError:
        return None


def _bar(value: float, tol: float, display_max: float, width: int) -> str:
    """Horizontal bar centered at 0. Tolerance band drawn explicitly.

    `value` is the raw signed drift. `display_max` clamps visible range.
    Marker moves within the bar; off-scale shows an edge arrow.
    """
    if width < 10:
        width = 10
    center = width // 2
    half = center - 1

    clamped = max(-display_max, min(display_max, value))
    pos_frac = clamped / display_max
    marker = int(round(center + pos_frac * half))
    marker = max(0, min(width - 1, marker))

    tol_frac = tol / display_max
    tol_half_cells = max(1, int(round(tol_frac * half)))
    tol_lo = max(0, center - tol_half_cells)
    tol_hi = min(width - 1, center + tol_half_cells)

    cells = [" "] * width
    # tolerance band
    for i in range(tol_lo, tol_hi + 1):
        cells[i] = f"{GREEN}·{RESET}"
    # zero line
    cells[center] = f"{DIM}|{RESET}"
    # off-scale arrows
    if value < -display_max:
        cells[0] = f"{RED}<{RESET}"
    elif value > display_max:
        cells[width - 1] = f"{RED}>{RESET}"
    # marker: green if within tolerance, red otherwise
    in_tol = abs(value) <= tol
    marker_color = GREEN if in_tol else YELLOW if abs(value) < 3 * tol else RED
    cells[marker] = f"{BOLD}{marker_color}O{RESET}"
    return "[" + "".join(cells) + "]"


def _correct_banner(term_width: int) -> list[str]:
    lines = [
        "",
        f"{BOLD}{GREEN}   ████████╗ ██████╗ ██████╗ ██████╗ ███████╗ ██████╗████████╗{RESET}",
        f"{BOLD}{GREEN}   ╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝{RESET}",
        f"{BOLD}{GREEN}      ██║   ██║   ██║██████╔╝██████╔╝█████╗  ██║        ██║   {RESET}",
        f"{BOLD}{GREEN}      ██║   ██║   ██║██╔══██╗██╔══██╗██╔══╝  ██║        ██║   {RESET}",
        f"{BOLD}{GREEN}      ██║   ╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗   ██║   {RESET}",
        f"{BOLD}{GREEN}      ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝   ╚═╝   {RESET}",
        "",
        f"{BOLD}{GREEN}   Pose locked. Tighten the mount now and re-run check_setup.py.{RESET}",
        "",
    ]
    # center roughly
    out = []
    for line in lines:
        visible = _visible_len(line)
        pad = max(0, (term_width - visible) // 2)
        out.append(" " * pad + line)
    return out


def _visible_len(s: str) -> int:
    # strip ANSI for width calc
    import re
    return len(re.sub(r"\033\[[0-9;?]*[A-Za-z]", "", s))


def _median_pose(poses: list[Pose]) -> Pose:
    """Element-wise median across roll/tx/ty; pass-through if no valid reads.

    Low-confidence raw reads are dropped; if the window has no valid reads
    we return a low-confidence placeholder so the UI shows the signal
    warning rather than a stale median.
    """
    valid = [p for p in poses if not p.low_conf and not np.isnan(p.roll_deg)]
    if not valid:
        last = poses[-1]
        return last
    roll = float(np.median([p.roll_deg for p in valid]))
    tx = float(np.median([p.tx_px for p in valid]))
    ty = float(np.median([p.ty_px for p in valid]))
    # Carry inlier stats from the latest valid read (most representative of
    # current signal health).
    latest = valid[-1]
    return Pose(
        roll_deg=roll,
        tx_px=tx,
        ty_px=ty,
        n_inliers=latest.n_inliers,
        n_matches=latest.n_matches,
        low_conf=False,
    )


def _status_label(value: float, tol: float) -> tuple[str, str]:
    """Return (color, short text) based on how far off `value` is."""
    a = abs(value)
    if a <= tol:
        return GREEN, "in tolerance"
    ratio = a / tol
    if ratio < 3:
        return YELLOW, "close"
    if ratio < 10:
        return YELLOW, "off"
    return RED, "far off"


# Flicker thresholds — readings below these between adjacent samples are
# considered "no real change," to stop the trend arrow from bouncing at rest.
FLICKER_ROLL_DEG = 0.15
FLICKER_TX_PX = 0.7
FLICKER_TY_PX = 0.7


def _trend_arrow(prev: float | None, curr: float, flicker: float) -> tuple[str, str]:
    """Show whether |value| is shrinking (good), growing (wrong way), or flat.

    Returns (color, text). Compares magnitudes so direction advice works
    regardless of which side of zero the value is on.
    """
    if prev is None:
        return DIM, "…"
    delta = abs(curr) - abs(prev)
    if abs(delta) < flicker:
        return DIM, "— steady"
    if delta < 0:
        return GREEN, "v better (keep going)"
    return RED, "^ wrong way (reverse)"


def _streak_bar(current: int, required: int, width: int = 14) -> str:
    filled = int(round((current / required) * width)) if required > 0 else 0
    filled = max(0, min(width, filled))
    bar = "#" * filled + "." * (width - filled)
    color = GREEN if current >= required else YELLOW if current > 0 else DIM
    return f"{color}[{bar}]{RESET} {current}/{required}"


def _signal_health(pose) -> tuple[str, str]:
    """inlier count → (color, human label)."""
    if np.isnan(pose.roll_deg):
        return RED, "low confidence (obstructed / too dark / too few features)"
    n = pose.n_inliers
    if n >= 60:
        return GREEN, f"strong match ({n} inliers)"
    if n >= 30:
        return YELLOW, f"ok match ({n} inliers) — some jitter expected"
    return RED, f"weak match ({n} inliers) — readings may be unreliable"


def _render(
    pose,
    ok: bool,
    ok_streak: int,
    required_streak: int,
    reference_label: str,
    cam_label: str,
    last_hz: float,
    prev_values: dict[str, float | None],
    buffer_fill: int,
    buffer_size: int,
    term_size: os.terminal_size,
) -> None:
    w = term_size.columns
    bar_w = max(30, min(60, w - 46))

    out: list[str] = [HOME]

    # ─── Title bar ──────────────────────────────────────────────
    rule = "=" * max(10, min(w, 100))
    out.append(f"{BOLD}{CYAN}{rule}{RESET}{CSI}K")
    out.append(
        f"{BOLD}{CYAN} TOPDOWN ALIGNMENT{RESET}  "
        f"{DIM}device {cam_label}   |   refresh {last_hz:4.1f} Hz   |   Ctrl+C to quit{RESET}{CSI}K"
    )
    out.append(f"{DIM} ref: {reference_label}{RESET}{CSI}K")
    out.append(f"{BOLD}{CYAN}{rule}{RESET}{CSI}K")
    out.append(CSI + "K")

    # ─── How-to ─────────────────────────────────────────────────
    out.append(f"{BOLD} HOW TO ALIGN{RESET}{CSI}K")
    out.append(f"   1. {BOLD}ROLL first.{RESET} Twist the mount slowly. Watch the arrow:"
               f"  {GREEN}v better{RESET} = keep going,  {RED}^ wrong way{RESET} = reverse."
               f"  Target {BOLD}{TOL_ROLL_DEG}°{RESET}.{CSI}K")
    out.append(f"   2. {BOLD}TX next.{RESET}   Slide the mount along image X.  "
               f"Same arrow rule. Target {BOLD}{TOL_TX_PX}px{RESET}.{CSI}K")
    out.append(f"   3. {BOLD}TY last.{RESET}   Slide the mount along image Y.  "
               f"Same arrow rule. Target {BOLD}{TOL_TY_PX}px{RESET}.{CSI}K")
    out.append(f"   Smoothing: {BOLD}{buffer_size}-frame{RESET} average "
               f"(~{buffer_size/30.0:.1f}s) + {BOLD}5-read median{RESET} filter. "
               f"After a nudge, wait ~{(buffer_size/30.0 + 5/4.0):.0f}s for the number to "
               f"fully settle.{CSI}K")
    out.append(CSI + "K")

    # ─── Signal health ─────────────────────────────────────────
    sig_color, sig_msg = _signal_health(pose)
    buffer_color = GREEN if buffer_fill >= buffer_size else YELLOW
    out.append(f"{BOLD} SIGNAL{RESET}  {sig_color}{sig_msg}{RESET}"
               f"   {DIM}(inliers {pose.n_inliers}/{pose.n_matches}, "
               f"need >=20 for a trustworthy reading){RESET}{CSI}K")
    out.append(f"{BOLD} BUFFER{RESET}  {buffer_color}{buffer_fill}/{buffer_size} frames{RESET}"
               f"   {DIM}(averaged before pose eval — same method as check_setup gate){RESET}{CSI}K")
    out.append(CSI + "K")

    if np.isnan(pose.roll_deg):
        out.append(f"{RED} pose: cannot measure — clear obstructions, add light, "
                   f"or check lens.{RESET}{CSI}K")
        # Pad bottom to avoid leftover output.
        for _ in range(14):
            out.append(CSI + "K")
        out.append(f"{CSI}J")
        sys.stdout.write("\n".join(out))
        sys.stdout.flush()
        return

    # ─── Axes ──────────────────────────────────────────────────
    sub_rule = "-" * max(10, min(w, 100))
    out.append(f"{DIM}{sub_rule}{RESET}{CSI}K")

    def row(label: str, value: float, unit: str, tol: float,
            display_max: float, prev: float | None, flicker: float) -> str:
        color, status = _status_label(value, tol)
        bar = _bar(value, tol, display_max, bar_w)
        trend_color, trend_text = _trend_arrow(prev, value, flicker)
        return (
            f" {BOLD}{label:<5}{RESET} "
            f"{color}{value:+8.2f}{unit}{RESET}  "
            f"{bar}  "
            f"{color}{status:<13}{RESET}  "
            f"{trend_color}{trend_text}{RESET}{CSI}K"
        )

    roll, tx, ty = pose.roll_deg, pose.tx_px, pose.ty_px
    out.append(row("roll", roll, "°",  TOL_ROLL_DEG, 20.0,
                   prev_values.get("roll"), FLICKER_ROLL_DEG))
    out.append(row("tx",   tx,   "px", TOL_TX_PX,    50.0,
                   prev_values.get("tx"),   FLICKER_TX_PX))
    out.append(row("ty",   ty,   "px", TOL_TY_PX,    50.0,
                   prev_values.get("ty"),   FLICKER_TY_PX))
    out.append(f"{DIM}{sub_rule}{RESET}{CSI}K")
    out.append(CSI + "K")

    # ─── Big status / streak ───────────────────────────────────
    if ok and ok_streak >= required_streak:
        for line in _correct_banner(w):
            out.append(line + CSI + "K")
    else:
        if ok:
            head = f"{BOLD}{GREEN}  HOLD STEADY{RESET}{DIM} — confirming...{RESET}"
        else:
            head = f"{BOLD}{YELLOW}  ADJUST{RESET}"
        out.append(head + CSI + "K")
        out.append(f"  hold steady: {_streak_bar(ok_streak, required_streak)}{CSI}K")
        # Pad to match banner height (banner is 10 lines) so we don't leave
        # stale text on screen when alignment drops out of CORRECT.
        for _ in range(8):
            out.append(CSI + "K")

    # clear to end of screen
    out.append(f"{CSI}J")
    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument("--reference", type=Path, help="path to reference PNG")
    ref_group.add_argument("--dataset", type=Path, help="lerobot dataset root")
    ref_group.add_argument("--latest", action="store_true",
                           help="use most recent ETHRC dataset")
    parser.add_argument("--hz", type=float, default=4.0,
                        help="pose evaluation rate (default: 4)")
    parser.add_argument("--confirm-seconds", type=float, default=1.5,
                        help="seconds in tolerance before declaring CORRECT")
    parser.add_argument("--smooth-frames", type=int, default=60,
                        help="rolling frame-buffer size averaged before each "
                             "pose eval (default: 60 ≈ 2.0s at 30Hz capture). "
                             "Larger = more stable readings, more motion-lag.")
    parser.add_argument("--median-window", type=int, default=5,
                        help="size of the median filter applied to pose "
                             "readings (default: 5). 1 disables filtering.")
    args = parser.parse_args()

    reference, reference_label = _resolve_reference(args)
    camera, cam_label = _start_realsense()

    interval_s = 1.0 / max(0.5, args.hz)
    required_streak = max(1, int(round(args.confirm_seconds * args.hz)))

    sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR + HOME)
    sys.stdout.flush()

    def _cleanup(*_):
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()
        try:
            camera.disconnect()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    ok_streak = 0
    last_eval_time = 0.0
    last_hz = 0.0
    # prev_values feeds the trend arrow: "is my last nudge making |value|
    # smaller or bigger?" None → first read, no arrow yet.
    prev_values: dict[str, float | None] = {"roll": None, "tx": None, "ty": None}

    # Per-frame pose is dominated by RANSAC noise + sensor noise (empirically
    # ±20° / ±60px with the camera stationary). check_setup averages 30 frames
    # before measuring, so the live tool must do the same or its readings
    # won't match the gate. We keep a rolling buffer of recent captures and
    # run pose against the average — same semantics as check_setup.
    buffer_size = max(1, int(args.smooth_frames))
    median_window = max(1, int(args.median_window))
    # Capture at 30 Hz (the RealSense frame rate) to fill the buffer quickly.
    # Duplicate reads are harmless — they just weight the average slightly.
    capture_interval_s = 1.0 / 30.0
    frame_buffer: collections.deque = collections.deque(maxlen=buffer_size)
    pose_window: collections.deque = collections.deque(maxlen=median_window)
    last_capture_time = 0.0

    try:
        while True:
            now = time.monotonic()

            # Capture tick: pull the latest frame from the background thread
            # and append to the rolling buffer.
            if now - last_capture_time >= capture_interval_s:
                frame = _grab_frame(camera)
                last_capture_time = now
                if frame is not None:
                    frame_buffer.append(frame)

            # Eval tick: average the buffer and compute pose.
            if now - last_eval_time < interval_s or len(frame_buffer) == 0:
                time.sleep(0.01)
                continue
            dt = now - last_eval_time if last_eval_time > 0 else interval_s
            last_eval_time = now
            last_hz = 1.0 / dt if dt > 0 else 0.0

            if len(frame_buffer) == 1:
                averaged = frame_buffer[0]
            else:
                averaged = np.mean(
                    np.stack(list(frame_buffer)).astype(np.float32), axis=0
                ).astype(np.uint8)

            # findHomography(RANSAC) is stochastic — seed for repeatability.
            cv2.setRNGSeed(0)
            raw_pose, _raw_ok, _msg = evaluate_pose(averaged, reference)

            # Median filter over last K poses. ORB feature selection is
            # sensitive to sensor noise, so even with frame averaging the
            # raw readings can swing (~30px tx) between evals. Median is
            # robust to these outlier reads.
            pose_window.append(raw_pose)
            pose = _median_pose(list(pose_window))
            ok = (
                not pose.low_conf
                and abs(pose.roll_deg) <= TOL_ROLL_DEG
                and abs(pose.tx_px) <= TOL_TX_PX
                and abs(pose.ty_px) <= TOL_TY_PX
            )
            if ok:
                ok_streak += 1
            else:
                ok_streak = 0

            _render(
                pose=pose,
                ok=ok,
                ok_streak=ok_streak,
                required_streak=required_streak,
                reference_label=reference_label,
                cam_label=cam_label,
                last_hz=last_hz,
                prev_values=prev_values,
                buffer_fill=len(frame_buffer),
                buffer_size=buffer_size,
                term_size=shutil.get_terminal_size(fallback=(100, 30)),
            )

            if not np.isnan(pose.roll_deg):
                prev_values["roll"] = pose.roll_deg
                prev_values["tx"] = pose.tx_px
                prev_values["ty"] = pose.ty_px
    finally:
        _cleanup()


if __name__ == "__main__":
    sys.exit(main())
