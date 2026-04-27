"""Unit tests for utils.camera_pose.

The measurement is a pure function of two BGR images and a mask. We verify
it on synthetic pairs where the ground truth is known by construction
(cv2.warpAffine of a reference image by a known rotation or translation).
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.camera_pose import (
    DRIFT_MULTIPLIER,
    TOL_TY_PX,
    evaluate_pose,
    measure_pose_topband,
    top_band_mask,
)


def _reference_image() -> np.ndarray:
    """Deterministic textured image with plenty of ORB-findable corners in
    the top band. A grid of squares gives reliable features at all the
    intersections."""
    rng = np.random.default_rng(0)
    h, w = 480, 640
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    for y in range(4, 120, 16):
        cv2.line(img, (0, y), (w, y), (230, 230, 230), 1)
    for x in range(4, w, 20):
        cv2.line(img, (x, 0), (x, 120), (20, 20, 20), 1)
    return img


def _warp(img: np.ndarray, rot_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


class TestMeasurePoseTopband(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = _reference_image()
        self.mask = top_band_mask(self.ref.shape[0], self.ref.shape[1])

    def test_identity_pair(self) -> None:
        pose = measure_pose_topband(self.ref, self.ref, self.mask)
        self.assertFalse(pose.low_conf)
        self.assertLess(abs(pose.roll_deg), 0.05)
        self.assertLess(abs(pose.tx_px), 0.5)
        self.assertLess(abs(pose.ty_px), 0.5)

    def test_rotation_sign_convention(self) -> None:
        # Sign is pinned empirically: a positive _warp rotation produces a
        # positive roll_deg of matching magnitude. Pinning so a future
        # refactor can't silently flip the sign (which would invert the
        # direction of "camera tilted CW vs CCW" in operator alerts).
        warped = _warp(self.ref, rot_deg=2.0)
        pose = measure_pose_topband(warped, self.ref, self.mask)
        self.assertFalse(pose.low_conf)
        self.assertGreater(pose.roll_deg, 0)
        self.assertLess(abs(pose.roll_deg - 2.0), 0.3)

    def test_translation_ty_recovers(self) -> None:
        warped = _warp(self.ref, ty=5.0)
        pose = measure_pose_topband(warped, self.ref, self.mask)
        self.assertFalse(pose.low_conf)
        self.assertLess(abs(pose.roll_deg), 0.3)
        self.assertLess(abs(abs(pose.ty_px) - 5.0), 1.0)

    def test_black_frame_is_low_confidence(self) -> None:
        black = np.zeros_like(self.ref)
        pose = measure_pose_topband(black, self.ref, self.mask)
        self.assertTrue(pose.low_conf)


class TestEvaluatePose(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = _reference_image()

    def test_passes_on_identity(self) -> None:
        _pose, ok, msg = evaluate_pose(self.ref, self.ref)
        self.assertTrue(ok)
        self.assertIn("OK", msg)

    def test_fails_on_large_rotation(self) -> None:
        warped = _warp(self.ref, rot_deg=5.0)
        _pose, ok, msg = evaluate_pose(warped, self.ref)
        self.assertFalse(ok)
        self.assertIn("DRIFT", msg)

    def test_marginal_translation_still_passes(self) -> None:
        # Translation between 1× and DRIFT_MULTIPLIER× the ty tolerance must
        # pass (ok=True) but tag the message as MARGINAL so the operator
        # sees the drift without a hard fail. Using pure translation avoids
        # the induced tx/ty that a rotation about image center produces.
        ty = TOL_TY_PX * (1.0 + (DRIFT_MULTIPLIER - 1.0) / 2.0)
        warped = _warp(self.ref, ty=ty)
        _pose, ok, msg = evaluate_pose(warped, self.ref)
        self.assertTrue(ok)
        self.assertIn("MARGINAL", msg)

    def test_reports_obstruction(self) -> None:
        black = np.zeros_like(self.ref)
        _pose, ok, msg = evaluate_pose(black, self.ref)
        self.assertFalse(ok)
        self.assertIn("obstructed", msg)


if __name__ == "__main__":
    unittest.main()
