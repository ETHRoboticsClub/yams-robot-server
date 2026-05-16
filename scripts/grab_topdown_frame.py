"""One-shot topdown RealSense capture.

Grabs a single color frame from the topdown camera (serial from configs/arms.yaml),
converts BGR → RGB, saves as PNG. Used to feed run_video2world.py for the
standalone video-generation diagnostic.
"""
from pathlib import Path
import argparse
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


ROOT = Path(__file__).resolve().parents[1]
ARMS_CONFIG = ROOT / "configs" / "arms.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(ARMS_CONFIG.read_text())
    topdown = cfg["cameras"]["configs"]["topdown"]
    serial = str(topdown["serial_number_or_name"])
    width = int(topdown.get("width", 640))
    height = int(topdown.get("height", 480))
    fps = int(topdown.get("fps", 30))

    devices = list(rs.context().query_devices())
    serials = {d.get_info(rs.camera_info.serial_number) for d in devices}
    if serial not in serials:
        print(f"topdown {serial} not found. visible: {serials}", file=sys.stderr)
        sys.exit(2)
    for d in devices:
        if d.get_info(rs.camera_info.serial_number) == serial:
            d.hardware_reset()
            break
    time.sleep(2)

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_device(serial)
    rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(rs_cfg)
    try:
        pipeline.wait_for_frames(timeout_ms=8000)
        for _ in range(int(args.warmup_s * fps)):
            pipeline.wait_for_frames(timeout_ms=2000)
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        bgr = np.asanyarray(frames.get_color_frame().get_data()).copy()
    finally:
        pipeline.stop()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR; we pass RGB-as-BGR so the file decodes back to RGB
    # when run_video2world.py loads it with PIL/imageio (which give RGB).
    # Cleaner: convert RGB→BGR for cv2.imwrite, since cv2 writes whatever it gets
    # as BGR by convention. So we save the original BGR.
    cv2.imwrite(str(out), bgr)
    print(f"saved {out}  shape={bgr.shape}  range={bgr.min()}..{bgr.max()}")


if __name__ == "__main__":
    main()
