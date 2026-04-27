"""Out-of-band depth sidecar writer for LeRobot datasets.

LeRobot's dataset writer is RGB-only — add_frame() explodes on uint16 or
non-3-channel arrays. We keep depth entirely out of the parquet/video
tree and drop it as PNG-16 files next to it:

    <dataset_root>/depth/<feature_name>/episode_NNNNNN/frame_NNNNNN.png

PNG-16 because depth is uint16 millimeters (D4xx scale = 0.001 m/unit)
and we want lossless. One episode at 30 FPS × 120 s × ~200 KB ≈ 700 MB,
so plan disk accordingly.

meta/depth_info.json is written once per dataset with units + shape so
the training loader doesn't have to guess.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


DEPTH_META_FILENAME = "depth_info.json"
DEPTH_DIR_NAME = "depth"


def depth_feature_dir(dataset_root: Path, feature_name: str) -> Path:
    return Path(dataset_root) / DEPTH_DIR_NAME / feature_name


def depth_frame_path(
    dataset_root: Path, feature_name: str, episode_index: int, frame_index: int
) -> Path:
    return (
        depth_feature_dir(dataset_root, feature_name)
        / f"episode_{episode_index:06d}"
        / f"frame_{frame_index:06d}.png"
    )


INVALID_SENTINEL = np.uint16(0)  # D4xx convention: 0 means "no measurement"


def min_pool_ignore_zero(depth_u16: NDArray[Any], block: int) -> NDArray[Any]:
    """Downsample depth by `block` using min-pooling that treats 0 as invalid.

    In each BxB source block, return the smallest positive value. If every
    pixel in the block is 0 (fully invalid region), the output stays 0.
    This preserves the nearest real object in each patch — the common case
    for top-down manipulation views where the near object is the signal
    and the floor behind it is the noise.

    Pure numpy: reshape to (dh, B, dw, B), mask zeros to a high sentinel,
    min over the two B axes, then restore 0 where the whole block was 0.
    Crops to the nearest (h//B)*B, (w//B)*B when the source dims don't
    divide evenly — matches cv2.resize's silent-crop behavior.
    """
    if block == 1:
        return depth_u16
    h, w = depth_u16.shape
    dh, dw = h // block, w // block
    clipped = depth_u16[: dh * block, : dw * block]
    reshaped = clipped.reshape(dh, block, dw, block)
    all_invalid = (reshaped == 0).all(axis=(1, 3))
    # Promote to uint32 so the "max sentinel" trick can't wrap around. We
    # could also use np.where with a large uint16, but 65535 is a plausible
    # real value post-clip; widening makes the intent explicit.
    widened = reshaped.astype(np.uint32)
    masked = np.where(widened > 0, widened, np.uint32(np.iinfo(np.uint32).max))
    pooled = masked.min(axis=(1, 3))
    pooled = np.where(all_invalid, 0, pooled).astype(np.uint16)
    return pooled


class DepthSidecar:
    """Async PNG-16 writer for one dataset root.

    Thread-safe; one instance handles any number of depth features. Writes
    are queued to a small pool so encoding does not block the record loop.

    Per-frame processing applied BEFORE PNG encoding, in order:

      1. `clip_max_mm` (if > 0): any pixel with depth > clip_max_mm becomes
         0 (invalid). Rationale: D4xx depth past ~3 m indoors is noisy
         enough to hurt more than help, and clipping those to "invalid"
         (rather than leaving large noisy values) also prevents them from
         poisoning the subsequent min-pool output.

      2. `downsample` (integer divisor, default 1): min-pooling that
         ignores invalid (=0) pixels. See min_pool_ignore_zero for why.
         Divisor 2 on 640x480 yields 320x240 at ~4x less disk.
    """

    def __init__(
        self,
        dataset_root: Path,
        max_workers: int = 2,
        downsample: int = 1,
        clip_max_mm: int = 0,
    ) -> None:
        if downsample < 1 or not isinstance(downsample, int):
            raise ValueError(f"downsample must be a positive int, got {downsample!r}")
        if clip_max_mm < 0:
            raise ValueError(f"clip_max_mm must be >= 0, got {clip_max_mm}")
        self.root = Path(dataset_root)
        self.downsample = downsample
        self.clip_max_mm = clip_max_mm
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="depth-sidecar"
        )
        self._meta_lock = threading.Lock()
        self._meta_written: set[str] = set()
        self._mkdir_cache: set[Path] = set()
        self._mkdir_lock = threading.Lock()

    def _ensure_episode_dir(self, feature_name: str, episode_index: int) -> Path:
        ep_dir = (
            depth_feature_dir(self.root, feature_name)
            / f"episode_{episode_index:06d}"
        )
        with self._mkdir_lock:
            if ep_dir not in self._mkdir_cache:
                ep_dir.mkdir(parents=True, exist_ok=True)
                self._mkdir_cache.add(ep_dir)
        return ep_dir

    def _write_now(self, path: Path, depth_u16: NDArray[Any]) -> None:
        # PNG compression is a good tradeoff for depth: D4xx depth maps are
        # smooth with lots of near-constant regions, so zlib compresses
        # well. imwrite PNG default is compression level 3; bump to 6 for
        # ~30% smaller files with negligible CPU impact at 30 FPS.
        ok = cv2.imwrite(str(path), depth_u16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")

    def _preprocess(self, depth_u16: NDArray[Any]) -> NDArray[Any]:
        """Apply clip then min-pool. Clip first so pooling never sees far
        noise — a 2x2 block of [500, 500, 9000, 9000] clipped at 4000
        becomes [500, 500, 0, 0] and min-pools to 500 (the real object),
        which is exactly the behavior we want at occluding edges.
        """
        out = depth_u16
        if self.clip_max_mm > 0:
            # np.where writes into a new array — the input buffer stays
            # intact for other consumers (e.g. live display).
            out = np.where(out > self.clip_max_mm, INVALID_SENTINEL, out)
        if self.downsample > 1:
            out = min_pool_ignore_zero(out, self.downsample)
        return out

    def write_frame(
        self,
        feature_name: str,
        episode_index: int,
        frame_index: int,
        depth_u16: NDArray[Any],
    ) -> None:
        if depth_u16.dtype != np.uint16:
            raise TypeError(
                f"depth sidecar expects uint16 mm, got dtype={depth_u16.dtype}"
            )
        # Process synchronously so the shape we record in the manifest
        # matches what hits disk, even if the async write queue is deep.
        processed = self._preprocess(depth_u16)
        self._ensure_episode_dir(feature_name, episode_index)
        self._write_meta_once(feature_name, processed.shape)
        path = depth_frame_path(self.root, feature_name, episode_index, frame_index)
        # Defensive copy — _preprocess may have returned the same buffer at
        # scale=1 + clip=0 and the caller is free to reuse it after this.
        self._pool.submit(self._write_now, path, processed.copy())

    def _write_meta_once(
        self, feature_name: str, shape: tuple[int, ...]
    ) -> None:
        with self._meta_lock:
            if feature_name in self._meta_written:
                return
            self._meta_written.add(feature_name)
        meta_path = self.root / "meta" / DEPTH_META_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any]
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}
        features = existing.setdefault("features", {})
        features[feature_name] = {
            "dtype": "uint16",
            "shape": list(shape),
            "encoding": "png16",
            "units": "mm",
            "scale_m_per_unit": 0.001,
            "downsample": self.downsample,
            "clip_max_mm": self.clip_max_mm,
        }
        existing.setdefault("version", 1)
        meta_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")

    def drop_episode(self, feature_name: str, episode_index: int) -> None:
        """Remove all depth frames for one episode — called when the operator
        re-records or the episode is otherwise discarded, so depth stays in
        sync with the parquet row count.
        """
        ep_dir = (
            depth_feature_dir(self.root, feature_name)
            / f"episode_{episode_index:06d}"
        )
        if not ep_dir.is_dir():
            return
        for f in ep_dir.glob("frame_*.png"):
            f.unlink()
        try:
            ep_dir.rmdir()
        except OSError:
            pass
        with self._mkdir_lock:
            self._mkdir_cache.discard(ep_dir)

    def flush(self) -> None:
        self._pool.shutdown(wait=True)
