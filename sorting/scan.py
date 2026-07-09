"""
Scene scan / inventory from the fixed vertical scan camera.

Finding from real Buehler labels (2026-07-09): at full-mat framing each barcode
spans ~120-180 px, ~4-6x under the ~800 px Code128 decode floor, and no upscale
recovers it. But the human-readable part number ("UNN-10015-007") OCRs at
0.98-0.99 confidence. So identity on the scan cam is by OCR, made safe by:

  1. a tight part pattern (letter-prefix + two dash-separated numeric groups),
  2. validation against a CLOSED set of known SKUs (reject anything not in it),
  3. an abstain threshold on OCR confidence.

Result: the scan produces a scene INVENTORY (which parts are present, how many,
where) that drives declutter -> place-ordered. A packet that can't be read with
confidence is flagged for a human / present-to-wrist re-read, never mis-placed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

# Buehler part number, e.g. UNN-10015-007, MDDY-11065-001, UNN-16022-009.
# Requires the dash structure so Auftrag ids (PL0194564) and Zuordnung material
# numbers (3050762479) do NOT match.
PART_RE = re.compile(r"[A-Z]{2,4}-\d{4,5}-\d{3}")
QTY_RE = re.compile(r"(\d{1,3})\s*ST\b")


def normalize_part(text: str) -> str | None:
    t = re.sub(r"\s+", "", text.upper()).replace("--", "-")
    m = PART_RE.search(t)
    return m.group(0) if m else None


@dataclass
class InventoryItem:
    part: str | None                 # normalized SKU, or None if unread
    qty: int | None                  # 'N ST' if found near the label
    conf: float                      # OCR confidence of the part token
    center: tuple[int, int]          # px center of the part text (grasp/track hint)
    accepted: bool                   # passed known-set + confidence gates
    reason: str = ""                 # why rejected / abstained


@dataclass
class ScanResult:
    items: list[InventoryItem] = field(default_factory=list)
    raw_text: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[InventoryItem]:
        return [i for i in self.items if i.accepted]

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for i in self.accepted:
            c[i.part] = c.get(i.part, 0) + 1
        return c


def _box_center(box) -> tuple[int, int]:
    pts = np.array(box, dtype=float)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


class ScanReader:
    """OCR-based scene inventory. Lazy-loads rapidocr; safe to construct without
    a camera. Pass known_parts to reject anything outside the kit's SKU set."""

    def __init__(self, known_parts: set[str] | None = None,
                 min_conf: float = 0.80, dedup_px: int = 90):
        self.known = {p.upper() for p in known_parts} if known_parts else None
        self.min_conf = min_conf
        self.dedup_px = dedup_px
        self._ocr = None

    def _ocr_lines(self, image: np.ndarray):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        res, _ = self._ocr(image)
        return res or []

    def read_inventory(self, image: np.ndarray) -> ScanResult:
        lines = self._ocr_lines(image)
        raw = [t for _b, t, _c in lines]

        parts, qtys = [], []
        for box, text, conf in lines:
            cx, cy = _box_center(box)
            p = normalize_part(text)
            if p:
                parts.append((p, float(conf), (cx, cy)))
            for m in QTY_RE.findall(text.upper()):
                qtys.append((int(m), (cx, cy)))

        # associate each qty with the nearest part token by pixel distance
        items: list[InventoryItem] = []
        used_centers: list[tuple[int, int]] = []
        for part, conf, ctr in parts:
            # dedup: the same physical label often yields the part twice
            # (printed text + barcode caption). Keep the highest-confidence one.
            dup = next((c for c in used_centers
                        if abs(c[0] - ctr[0]) < self.dedup_px
                        and abs(c[1] - ctr[1]) < self.dedup_px), None)
            if dup is not None:
                continue
            used_centers.append(ctr)

            qty = None
            if qtys:
                qty = min(qtys, key=lambda q: (q[1][0]-ctr[0])**2 + (q[1][1]-ctr[1])**2)[0]

            accepted, reason = True, ""
            if conf < self.min_conf:
                accepted, reason = False, f"low_conf<{self.min_conf}"
            elif self.known is not None and part not in self.known:
                accepted, reason = False, "not_in_known_set"
            items.append(InventoryItem(part, qty, round(conf, 3), ctr, accepted, reason))

        return ScanResult(items=items, raw_text=raw)


# --- optional live grab from the tuned scan camera ---------------------------
# RealSense D435i color node, validated settings: 1280x720, manual exposure 150,
# gain 48, contrast 75, sharpness 90. Uses ffmpeg (matches label_tuner) so it
# does not fight the RealSense SDK for the device.
SCAN_DEV = "/dev/video14"
SCAN_CTRLS = {"auto_exposure": 1, "exposure_time_absolute": 150, "gain": 48,
              "brightness": 0, "contrast": 75, "sharpness": 90}


def grab_scan_frame(dev: str = SCAN_DEV, w: int = 1280, h: int = 720) -> np.ndarray | None:
    """Single frame from the scan cam via ffmpeg. Returns BGR or None.
    NOTE: the camera is exclusive — stop any tuner/teleop consumer first."""
    import subprocess
    for name, val in SCAN_CTRLS.items():
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       capture_output=True)
    try:
        raw = subprocess.check_output(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "v4l2",
             "-input_format", "yuyv422", "-video_size", f"{w}x{h}",
             "-i", dev, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            stderr=subprocess.DEVNULL)
    except Exception:
        return None
    if not raw or not _HAVE_CV2:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
