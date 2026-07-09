"""
Perception: find one packet to pick, and provide a close-up frame for reading.

Backends:
  MockPerception  emits N detections carrying a fixed image (dry-run).
  SimPerception   emits synthetic packets whose label images carry REAL Code128
                  barcodes (via zxing-cpp writer), so read→decide→place is a true
                  end-to-end test with no hardware. Also renders a live "source
                  box" overview frame for the dashboard.
  RealSensePerception  depth-threshold segments the topmost packet from the
                  top-down RealSense (color+depth). Camera acquisition is
                  import-guarded (pyrealsense2) so mock/sim need no camera.

PacketDetection.reading_frame is what the reader decodes; in real mode that comes
from a close-up reading-station pose (≥720p — the measured decode floor).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

try:
    import zxingcpp
    _HAVE_ZXING = True
except Exception:
    _HAVE_ZXING = False

log = logging.getLogger("sorting.perception")


@dataclass
class PacketDetection:
    present: bool
    pick_xy: tuple[int, int] | None = None      # image px of grasp point
    label_frame: np.ndarray | None = None       # close-up frame to read
    mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Label rendering (shared by Sim; makes a Bühler-ish card with a real barcode)
# --------------------------------------------------------------------------

def render_label(part_number: str, name: str = "", width: int = 960,
                 height: int = 540) -> np.ndarray:
    """White label card with header text and a real Code128 barcode. BGR."""
    img = np.full((height, width, 3), 255, np.uint8)
    if not _HAVE_CV2:
        return img
    black = (20, 20, 20)
    gray = (120, 120, 120)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (210, 210, 210), 2)
    cv2.putText(img, "BUHLER", (width - 300, 70), cv2.FONT_HERSHEY_DUPLEX, 1.6, black, 3)
    cv2.putText(img, "Identifikation", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gray, 2)
    cv2.putText(img, part_number, (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, black, 3)
    if name:
        cv2.putText(img, name, (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.9, black, 2)

    if _HAVE_ZXING:
        try:
            bc = zxingcpp.create_barcode(part_number, zxingcpp.BarcodeFormat.Code128)
            code = np.array(bc.to_image())          # grayscale HxW
            code = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
            bw, bh = width - 120, 200
            code = cv2.resize(code, (bw, bh), interpolation=cv2.INTER_NEAREST)
            y0 = 250
            img[y0:y0 + bh, 60:60 + bw] = code
            cv2.putText(img, part_number, (60, y0 + bh + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, black, 2)
        except Exception:
            pass
    return img


class MockPerception:
    """Yields `count` identical detections carrying `image`, then empties."""

    def __init__(self, image: np.ndarray, count: int = 3):
        self._image = image
        self._left = count

    def next_detection(self) -> PacketDetection:
        if self._left <= 0:
            return PacketDetection(present=False)
        self._left -= 1
        h, w = self._image.shape[:2]
        return PacketDetection(True, (w // 2, h // 2), self._image.copy())

    def reading_frame(self, det: PacketDetection) -> np.ndarray:
        return det.label_frame

    def overview_frame(self) -> np.ndarray | None:
        return self._image

    def remaining(self) -> int:
        return self._left


class SimPerception:
    """Synthetic multi-packet source box for demos. Real barcodes, real reader."""

    def __init__(self, parts: list[str], names: dict[str, str] | None = None,
                 seed: int = 7, shuffle: bool = True):
        names = names or {}
        self._queue = list(parts)
        if shuffle:
            random.Random(seed).shuffle(self._queue)
        self._labels = {p: render_label(p, names.get(p, "")) for p in parts}
        self._names = names
        self._total = len(self._queue)
        self._current: str | None = None

    def next_detection(self) -> PacketDetection:
        if not self._queue:
            self._current = None
            return PacketDetection(present=False)
        self._current = self._queue.pop(0)
        label = self._labels[self._current]
        h, w = label.shape[:2]
        return PacketDetection(True, (w // 2, h // 2), label.copy(),
                               meta={"true_part": self._current})

    def reading_frame(self, det: PacketDetection) -> np.ndarray:
        return det.label_frame

    def overview_frame(self) -> np.ndarray | None:
        """A simple 'source box' scene shrinking as packets are removed."""
        if not _HAVE_CV2:
            return None
        W, H = 640, 480
        img = np.full((H, W, 3), 30, np.uint8)
        cv2.rectangle(img, (60, 60), (W - 60, H - 60), (90, 90, 90), 3)
        cv2.putText(img, "SOURCE BOX", (70, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 180, 180), 2)
        rng = random.Random(1)
        for i in range(len(self._queue)):
            x = 90 + rng.randint(0, W - 220)
            y = 90 + rng.randint(0, H - 200)
            col = (rng.randint(120, 230), rng.randint(120, 230), rng.randint(120, 230))
            cv2.rectangle(img, (x, y), (x + 110, y + 70), col, -1)
            cv2.rectangle(img, (x + 15, y + 12), (x + 95, y + 40), (245, 245, 245), -1)
        cv2.putText(img, f"remaining: {len(self._queue)}", (70, H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 200, 255), 2)
        return img

    def remaining(self) -> int:
        return len(self._queue)


class RealSensePerception:
    """Depth-threshold segmentation of the topmost packet from the top-down cam.

    In real integration, `color`/`depth` come from the LeRobot robot's `topdown`
    RealSense observation. `acquire()` is a standalone pyrealsense2 path for
    testing the segmentation live without the full robot stack.
    """

    def __init__(self, near_m: float = 0.20, band_m: float = 0.08,
                 min_area_px: int = 1500, crop_pad: int = 40):
        self.near_m = near_m
        self.band_m = band_m
        self.min_area_px = min_area_px
        self.crop_pad = crop_pad
        self._pipe = None

    # -- segmentation on provided frames (unit-testable) ---------------

    def detect_topmost(self, color: np.ndarray, depth_m: np.ndarray) -> PacketDetection:
        if not _HAVE_CV2:
            raise RuntimeError("cv2 required for RealSense segmentation")
        valid = depth_m > 0
        if not valid.any():
            return PacketDetection(present=False)
        nearest = np.min(depth_m[valid])
        band = (depth_m >= nearest) & (depth_m <= nearest + self.band_m) & valid
        mask = (band.astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area_px]
        if not contours:
            return PacketDetection(present=False)
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-6))
        cy = int(M["m01"] / (M["m00"] + 1e-6))
        x, y, w, h = cv2.boundingRect(c)
        p = self.crop_pad
        crop = color[max(0, y - p):y + h + p, max(0, x - p):x + w + p].copy()
        obj_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(obj_mask, [c], -1, 255, -1)
        return PacketDetection(True, (cx, cy), crop, obj_mask,
                               meta={"nearest_m": float(nearest), "area": int(cv2.contourArea(c))})

    # -- optional standalone camera path -------------------------------

    def acquire(self) -> tuple[np.ndarray, np.ndarray]:
        import pyrealsense2 as rs  # deferred
        if self._pipe is None:
            self._pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            self._profile = self._pipe.start(cfg)
            self._align = rs.align(rs.stream.color)
            self._scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
        frames = self._align.process(self._pipe.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self._scale
        return color, depth

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
