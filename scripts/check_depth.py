"""Verify the topdown RealSense can stream a depth map alongside color.

Step 1 of the depth-sensing rollout — runs independently of the record
pipeline so we can confirm the hardware and pyrealsense2 build support
depth at the resolution/FPS we use today before touching any recording
code.

Run:
    PYTHONPATH=src uv run python scripts/check_depth.py

Writes a color PNG, a colorized-depth preview PNG, and a raw 16-bit PNG
(depth in millimeters) to outputs/depth_check/ for eyeball inspection.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS_CONFIG = ROOT / "configs" / "arms.yaml"
OUT_DIR = ROOT / "outputs" / "depth_check"

# Central ROI fraction for stats — ignores the image border where the
# projected IR pattern thins out and depth is unreliable.
ROI_FRAC = 0.5
# Frames captured for the FPS/validity measurement after warmup.
MEASURE_FRAMES = 150
# Warmup frames discarded before measuring (auto-exposure + projector settle).
WARMUP_FRAMES = 60


def load_topdown_config() -> dict:
    with open(ARMS_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    cameras = cfg["cameras"]["configs"]
    if "topdown" not in cameras:
        raise RuntimeError("configs/arms.yaml has no cameras.configs.topdown entry")
    topdown = cameras["topdown"]
    if topdown.get("type") not in ("intelrealsense", "intelrealsense-cached"):
        raise RuntimeError(
            f"topdown camera type is {topdown.get('type')!r}; expected a RealSense type."
        )
    return topdown


def find_device(serial: str):
    devices = list(rs.context().query_devices())
    available = [d.get_info(rs.camera_info.serial_number) for d in devices]
    for d in devices:
        if d.get_info(rs.camera_info.serial_number) == serial:
            return d
    raise RuntimeError(
        f"RealSense serial {serial} not found. Visible serials: {available or 'none'}.\n"
        "Replug the RealSense into a USB3 port, avoid passive hubs, and close realsense-viewer."
    )


def device_supports_depth(device) -> bool:
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.depth:
                return True
    return False


def get_depth_scale(profile) -> float:
    depth_sensor = profile.get_device().first_depth_sensor()
    return float(depth_sensor.get_depth_scale())


def roi_slice(h: int, w: int, frac: float) -> tuple[slice, slice]:
    rh, rw = int(h * frac), int(w * frac)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return slice(y0, y0 + rh), slice(x0, x0 + rw)


def colorize_depth(depth_mm: np.ndarray, clip_m: float = 2.0) -> np.ndarray:
    clip = int(clip_m * 1000)
    valid = depth_mm > 0
    vis = np.zeros_like(depth_mm, dtype=np.uint8)
    if valid.any():
        clamped = np.clip(depth_mm, 1, clip)
        normalized = ((clamped - 1) / (clip - 1) * 255).astype(np.uint8)
        vis = np.where(valid, normalized, 0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = load_topdown_config()
    serial = str(cam["serial_number_or_name"])
    width = int(cam.get("width", 640))
    height = int(cam.get("height", 480))
    fps = int(cam.get("fps", 30))
    warmup_s = float(cam.get("warmup_s", 3))

    print(f"Topdown config: serial={serial} {width}x{height}@{fps} warmup={warmup_s}s")

    device = find_device(serial)
    if not device_supports_depth(device):
        print("FAIL: device does not expose a depth sensor.")
        return 1
    print(
        f"Device: {device.get_info(rs.camera_info.name)} "
        f"fw={device.get_info(rs.camera_info.firmware_version)} "
        f"usb={device.get_info(rs.camera_info.usb_type_descriptor)}"
    )

    # Hardware reset matches check_setup.py's cold-start behavior — without
    # this a prior aborted lerobot-record can leave the device in a state
    # where pipeline.start() succeeds but frames never arrive.
    device.hardware_reset()
    time.sleep(2)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipeline.start(rs_config)
    except RuntimeError as exc:
        print(f"FAIL: pipeline.start raised: {exc}")
        print(
            "Most common cause: this (width, height, fps) combination isn't a supported "
            "depth profile. Try 640x480@30 which every D4xx supports. Enumerate with "
            "`rs-enumerate-devices -c` from librealsense."
        )
        return 1

    align = rs.align(rs.stream.color)
    depth_scale = get_depth_scale(profile)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()
    print(
        f"Color stream: {color_stream.width()}x{color_stream.height()}@{color_stream.fps()} "
        f"{color_stream.format().name}"
    )
    print(
        f"Depth stream: {depth_stream.width()}x{depth_stream.height()}@{depth_stream.fps()} "
        f"{depth_stream.format().name} scale={depth_scale:.6f} m/unit"
    )
    print(f"Color intrinsics: fx={color_intr.fx:.1f} fy={color_intr.fy:.1f} ppx={color_intr.ppx:.1f} ppy={color_intr.ppy:.1f}")
    print(f"Depth intrinsics: fx={depth_intr.fx:.1f} fy={depth_intr.fy:.1f} ppx={depth_intr.ppx:.1f} ppy={depth_intr.ppy:.1f}")

    try:
        # Warmup.
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames(timeout_ms=5000)

        start = time.perf_counter()
        depth_frames_ok = 0
        color_frames_ok = 0
        valid_ratios: list[float] = []
        roi_depths_mm: list[float] = []
        last_color_rgb = None
        last_depth_mm = None
        last_aligned_depth_mm = None
        for _ in range(MEASURE_FRAMES):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if color:
                color_frames_ok += 1
                last_color_rgb = np.asanyarray(color.get_data()).copy()
            if depth:
                depth_frames_ok += 1
                depth_mm = np.asanyarray(depth.get_data()).copy()
                last_aligned_depth_mm = depth_mm

                h, w = depth_mm.shape
                ys, xs = roi_slice(h, w, ROI_FRAC)
                roi = depth_mm[ys, xs]
                valid = roi > 0
                valid_ratios.append(float(valid.mean()))
                if valid.any():
                    roi_depths_mm.append(float(np.median(roi[valid])))

            raw_depth = frames.get_depth_frame()
            if raw_depth:
                last_depth_mm = np.asanyarray(raw_depth.get_data()).copy()

        elapsed = time.perf_counter() - start
    finally:
        pipeline.stop()

    measured_fps = depth_frames_ok / elapsed if elapsed > 0 else 0.0
    mean_valid = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    median_depth_m = (float(np.median(roi_depths_mm)) / 1000.0) if roi_depths_mm else 0.0

    print("")
    print(f"Captured: color={color_frames_ok}/{MEASURE_FRAMES}  depth={depth_frames_ok}/{MEASURE_FRAMES}")
    print(f"Measured depth FPS: {measured_fps:.1f}  (requested {fps})")
    print(f"ROI valid-pixel ratio (center {int(ROI_FRAC*100)}%): {mean_valid*100:.1f}%")
    print(f"ROI median depth: {median_depth_m:.3f} m")

    if last_color_rgb is not None:
        bgr = cv2.cvtColor(last_color_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT_DIR / "color.png"), bgr)
    if last_depth_mm is not None:
        cv2.imwrite(str(OUT_DIR / "depth_raw_u16_mm.png"), last_depth_mm)
        cv2.imwrite(str(OUT_DIR / "depth_preview.png"), colorize_depth(last_depth_mm))
    if last_aligned_depth_mm is not None:
        cv2.imwrite(
            str(OUT_DIR / "depth_aligned_to_color_u16_mm.png"), last_aligned_depth_mm
        )
    print(f"Sample frames written to: {OUT_DIR}")

    ok = (
        depth_frames_ok >= int(MEASURE_FRAMES * 0.9)
        and mean_valid > 0.7
        and measured_fps > fps * 0.8
    )
    if ok:
        print("")
        print("PASS: depth stream is healthy.")
        return 0

    print("")
    print("FAIL: depth stream is not healthy.")
    if depth_frames_ok < int(MEASURE_FRAMES * 0.9):
        print("- Missed too many depth frames; check USB3 bandwidth and cable.")
    if mean_valid <= 0.7:
        print(
            "- Valid-pixel ratio too low. Scene may be too close/far, too dark, "
            "or specular. D4xx typical range is ~0.3–3 m."
        )
    if measured_fps <= fps * 0.8:
        print("- Achieved FPS far below requested; try USB3 / different port.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
