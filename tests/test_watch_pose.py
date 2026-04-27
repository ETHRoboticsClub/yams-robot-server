"""Integration test for scripts/watch_pose.py helpers.

The polling loop itself is exercised by a smoke run (see docs/manual test
list). Here we test the two pure helpers: reading the first N frames from
an mp4, and measuring pose on the average — because those are where the
real correctness risk lives.
"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.camera_pose import evaluate_
pose  # noqa: E402
from watch_pose import _average, _read_first_frames  # noqa: E402


def _make_synthetic_mp4(path: Path, frame: np.ndarray, n_frames: int) -> None:
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    h, w = 480, 640
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    for y in range(4, 120, 16):
        cv2.line(img, (0, y), (w, y), (230, 230, 230), 1)
    for x in range(4, w, 20):
        cv2.line(img, (x, 0), (x, 120), (20, 20, 20), 1)
    return img


class TestWatchPoseHelpers(unittest.TestCase):
    def test_read_first_frames_returns_n(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episode_000000.mp4"
            _make_synthetic_mp4(path, _textured_frame(), n_frames=35)
            frames = _read_first_frames(path, 30)
            self.assertIsNotNone(frames)
            self.assertEqual(len(frames), 30)
            self.assertEqual(frames[0].shape, (480, 640, 3))

    def test_read_first_frames_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episode_000000.mp4"
            _make_synthetic_mp4(path, _textured_frame(), n_frames=10)
            frames = _read_first_frames(path, 30)
            self.assertIsNone(frames)

    def test_average_then_evaluate_happy_path(self) -> None:
        frame = _textured_frame()
        frames = [frame] * 30
        avg = _average(frames)
        _pose, ok, msg = evaluate_pose(avg, frame)
        self.assertTrue(ok)
        self.assertIn("OK", msg)

    def test_average_then_evaluate_flags_rotation(self) -> None:
        frame = _textured_frame()
        h, w = frame.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
        rotated = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames = [rotated] * 30
        avg = _average(frames)
        _pose, ok, msg = evaluate_pose(avg, frame)
        self.assertFalse(ok)
        self.assertIn("DRIFT", msg)


if __name__ == "__main__":
    unittest.main()
