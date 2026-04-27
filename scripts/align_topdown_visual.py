"""Visual topdown alignment aid.

Captures a smoothed live frame from the RealSense topdown and writes six
overlays against the committed reference to `outputs/alignment_diff/`. Open
any of them in VS Code — its image preview auto-reloads when files change,
so you can watch them update live while you re-aim the mount.

All overlays restrict analysis to the top band (the rigid mat/table
background), matching the pose gate in `utils/camera_pose.py`. The region
below is masked out because the gripper and scene objects change every
session and would otherwise dominate the visual diff.

Outputs (all 640x480 unless noted):
  reference.png     committed reference
  live.png          averaged live frame
  blend.png         50/50 blend in the band (ghosted doubles → misaligned)
  checkerboard.png  alternating ref/live tiles in the band (broken edges → misaligned)
  diff.png          |ref - live| contrast-stretched, band only (bright = mismatch)
  sidebyside.png    reference | live with shared grid in the band (1280x480)

Ctrl+C to exit.
"""
from __future__ import annotations

import atexit
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lerobot_camera_cached.camera_realsense_cached import RealSenseCameraCached  # noqa: E402
from lerobot_camera_cached.realsense_cached_config import RealSenseCameraCachedConfig  # noqa: E402
from utils.camera_pose import BAND_TOP_FRACTION  # noqa: E402

ARMS_CONFIG = ROOT / "configs" / "arms.yaml"
REFERENCE_PATH = ROOT / "outputs" / "camera_reference_images" / "topdown.png"
OUT_DIR = ROOT / "outputs" / "alignment_diff"
BUFFER_SIZE = 60          # frames averaged per update
REFRESH_S = 1.0            # seconds between output updates
TILE = 48                  # checkerboard tile size (px)
CLEANUP_DELAY_S = 120     # delete stale overlays this many seconds after exit

# The pose gate in utils/camera_pose.py only measures the top band of the
# frame (rigid warehouse background). Everything below is gripper + scene
# objects, which change every session. Mask overlays to that same band so
# the VS Code diffs reflect what the gate actually checks.
MASKED_GRAY = 64           # dim value for ignored region — visible but not distracting


def schedule_cleanup(out_dir: Path, delay_s: int) -> None:
    """Spawn a detached shell that deletes files in `out_dir` older than
    `delay_s` after sleeping for the same duration.

    Uses `find -mmin +N -delete` so if the user re-runs the script within
    the window, fresh writes survive — only truly stale files are pruned.
    The subprocess runs in its own session, so it survives this process
    exiting and even an SSH disconnect.
    """
    minutes = max(1, delay_s // 60)
    cmd = (
        f"sleep {delay_s} && "
        f"find {out_dir} -type f -mmin +{minutes} -delete 2>/dev/null; "
        f"rmdir {out_dir} 2>/dev/null; true"
    )
    subprocess.Popen(
        ["sh", "-c", cmd],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_camera() -> RealSenseCameraCached:
    cam_yaml = yaml.safe_load(ARMS_CONFIG.read_text())["cameras"]["configs"]["topdown"]
    camera = RealSenseCameraCached(
        RealSenseCameraCachedConfig(
            serial_number_or_name=str(cam_yaml["serial_number_or_name"]),
            fps=int(cam_yaml.get("fps", 30)),
            width=int(cam_yaml.get("width", 640)),
            height=int(cam_yaml.get("height", 480)),
            use_depth=False,
            warmup_s=1.0,
        )
    )
    camera.connect()
    return camera


def averaged_frame(camera: RealSenseCameraCached, n: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    while len(frames) < n:
        try:
            frames.append(camera.async_read(timeout_ms=500))
        except TimeoutError:
            continue
        time.sleep(1 / 30)
    return np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def band_height(h: int) -> int:
    return int(h * BAND_TOP_FRACTION)


def mask_below_band(img: np.ndarray, fill: int = MASKED_GRAY) -> np.ndarray:
    """Return a copy of `img` with everything below the top band set to a
    uniform gray so the ignored region can't distract from true misalignment.
    """
    out = img.copy()
    out[band_height(img.shape[0]):] = fill
    return out


def draw_band_boundary(img: np.ndarray, offset_x: int = 0) -> None:
    """Draw a horizontal line at the band boundary with an 'ignored below'
    label. Mutates `img` in place.
    """
    y = band_height(img.shape[0])
    cv2.line(img, (offset_x, y), (offset_x + img.shape[1] - offset_x, y),
             (0, 0, 255), 2)
    cv2.putText(img, "ignored below (gripper + scene objects)",
                (offset_x + 8, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def make_blend(ref: np.ndarray, live: np.ndarray) -> np.ndarray:
    blend = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
    blend = mask_below_band(blend)
    draw_band_boundary(blend)
    return blend


def make_checkerboard(ref: np.ndarray, live: np.ndarray, tile: int) -> np.ndarray:
    h, w = ref.shape[:2]
    out = ref.copy()
    band_h = band_height(h)
    for y in range(0, band_h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 1:
                y_end = min(y + tile, band_h)
                out[y:y_end, x:x + tile] = live[y:y_end, x:x + tile]
    out = mask_below_band(out)
    draw_band_boundary(out)
    return out


def make_diff(ref: np.ndarray, live: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(ref, live)
    # Contrast-stretch so small misalignments are visible.
    d = np.clip(d.astype(np.int16) * 3, 0, 255).astype(np.uint8)
    # Zero out the ignored region — any diff there is irrelevant.
    d[band_height(d.shape[0]):] = 0
    draw_band_boundary(d)
    return d


def make_sidebyside(ref: np.ndarray, live: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    ref_m = mask_below_band(ref)
    live_m = mask_below_band(live)
    pair = np.concatenate([ref_m, live_m], axis=1)
    band_h = band_height(h)
    # Shared horizontal grid lines within the band — anything misaligned
    # vertically shows as a broken line crossing the seam.
    for y in range(0, band_h, 48):
        cv2.line(pair, (0, y), (pair.shape[1], y), (0, 255, 255), 1)
    cv2.line(pair, (w, 0), (w, h), (0, 0, 255), 2)
    cv2.putText(pair, "REFERENCE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(pair, "LIVE", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # Band boundary on both halves.
    cv2.line(pair, (0, band_h), (pair.shape[1], band_h), (0, 0, 255), 2)
    cv2.putText(pair, "ignored below (gripper + scene objects)",
                (8, band_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return pair


def atomic_write(path: Path, img: np.ndarray) -> None:
    # Keep the .png extension on the temp file so cv2.imwrite recognizes it,
    # then rename over the target. This prevents VS Code's preview from
    # catching a half-written PNG.
    tmp = path.with_name(f".{path.name}.tmp.png")
    cv2.imwrite(str(tmp), img)
    tmp.replace(path)


def main() -> int:
    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    if reference is None:
        print(f"reference not found: {REFERENCE_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(OUT_DIR / "reference.png", reference)

    # Register auto-cleanup and signal handlers BEFORE opening the camera
    # (which can take several seconds). Without the handlers, a SIGTERM
    # during start_camera() skips atexit entirely and leaves stale files.
    atexit.register(schedule_cleanup, OUT_DIR, CLEANUP_DELAY_S)

    camera: RealSenseCameraCached | None = None

    def _cleanup(*_):
        if camera is not None:
            try:
                camera.disconnect()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    camera = start_camera()

    h = reference.shape[0]
    band_h = band_height(h)
    print(f"writing overlays to {OUT_DIR}/ every {REFRESH_S:.1f}s")
    print(f"measurement region: top {band_h}px of {h}px "
          f"({int(BAND_TOP_FRACTION * 100)}% — the rigid mat/table background).")
    print("everything below is ignored (gripper + scene objects vary per session).")
    print(f"on exit: files older than {CLEANUP_DELAY_S}s will be auto-deleted "
          f"(safe against re-runs within the window)")
    print("open these in VS Code (image preview auto-reloads):")
    for name in ("blend.png", "checkerboard.png", "diff.png",
                 "sidebyside.png", "live.png", "reference.png"):
        print(f"  {OUT_DIR / name}")
    print("Ctrl+C to exit.")

    iteration = 0
    while True:
        t0 = time.monotonic()
        live = averaged_frame(camera, BUFFER_SIZE)
        atomic_write(OUT_DIR / "live.png", live)
        atomic_write(OUT_DIR / "blend.png", make_blend(reference, live))
        atomic_write(OUT_DIR / "checkerboard.png",
                     make_checkerboard(reference, live, TILE))
        atomic_write(OUT_DIR / "diff.png", make_diff(reference, live))
        atomic_write(OUT_DIR / "sidebyside.png",
                     make_sidebyside(reference, live))
        iteration += 1
        dt = time.monotonic() - t0
        band_diff = float(
            cv2.absdiff(reference[:band_h], live[:band_h]).mean()
        )
        print(f"[{iteration:03d}] updated in {dt:.2f}s — "
              f"mean diff in band {band_diff:5.2f}")
        sleep_left = REFRESH_S - dt
        if sleep_left > 0:
            time.sleep(sleep_left)


if __name__ == "__main__":
    sys.exit(main())
