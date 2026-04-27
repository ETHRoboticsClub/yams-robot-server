"""Topdown camera pose gate.

Mirror of analytics/camera_pose_gate.py measure_pose_topband — keep in sync.

Measures (roll_deg, tx_px, ty_px) of a live capture vs a committed reference
image, restricted to the top 25% of the frame (warehouse background — the
only region that is rigid across sessions). Used pre-recording to abort if
the camera is mounted at the wrong angle, and between episodes to detect
mid-session drift.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "camera_reference_images" / "topdown.png"
)

TOL_ROLL_DEG = 0.5
TOL_TX_PX = 2.0
TOL_TY_PX = 2.0

# Soft band: drift within tolerance → OK. Between 1× and DRIFT_MULTIPLIER×
# tolerance → MARGINAL (pass, but warn how off it is). Beyond that → DRIFT
# (alert loudly, prompt alignment, but NEVER fail the record script — the
# operator decides whether to fix the mount or keep going).
DRIFT_MULTIPLIER = 4.0

# Within the DRIFT zone, below this multiplier the alignment tool is
# optional — the setup is workable and the operator can skip. At or above
# it, alignment is strongly recommended. Either way the caller only warns,
# never aborts recording.
RECOMMEND_ALIGNMENT_MULTIPLIER = 8.0

BAND_TOP_FRACTION = 0.25  # top 25% = warehouse background, rigid across sessions


@dataclass
class Pose:
    roll_deg: float
    tx_px: float
    ty_px: float
    n_inliers: int
    n_matches: int
    low_conf: bool


def _nan_pose() -> Pose:
    return Pose(float("nan"), float("nan"), float("nan"), 0, 0, True)


def top_band_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[: int(height * BAND_TOP_FRACTION)] = 255
    return mask


def measure_pose_topband(ref_a: np.ndarray, ref_b: np.ndarray, band_mask: np.ndarray) -> Pose:
    """ORB+RANSAC homography ref_a → ref_b, restricted to `band_mask`.

    Extracts in-plane rotation via SVD polar decomposition of the homography's
    top-left 2×2. tx, ty read directly from the translation column.
    """
    ga = cv2.cvtColor(ref_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(ref_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    kpa, da = orb.detectAndCompute(ga, band_mask)
    kpb, db = orb.detectAndCompute(gb, band_mask)
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return _nan_pose()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(da, db, k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return _nan_pose()
    src = np.array([kpa[g.queryIdx].pt for g in good], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([kpb[g.trainIdx].pt for g in good], dtype=np.float32).reshape(-1, 1, 2)
    hmat, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if hmat is None or inl is None:
        return _nan_pose()

    u, _s, vt = np.linalg.svd(hmat[:2, :2])
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = u @ vt
    rot_deg = math.degrees(math.atan2(r[1, 0], r[0, 0]))

    n_inl = int(inl.sum())
    n_match = len(good)
    low_conf = n_inl < 20 or (n_inl / n_match) < 0.30
    return Pose(
        roll_deg=float(rot_deg),
        tx_px=float(hmat[0, 2]),
        ty_px=float(hmat[1, 2]),
        n_inliers=n_inl,
        n_matches=n_match,
        low_conf=bool(low_conf),
    )


def worst_tolerance_ratio(pose: Pose) -> float:
    """Return max(|drift_i| / tol_i) across roll/tx/ty.

    Callers use this to classify drift severity (e.g. whether to recommend
    rerunning alignment). Returns NaN for low-confidence poses where the
    individual drift values are NaN.
    """
    return max(
        abs(pose.roll_deg) / TOL_ROLL_DEG,
        abs(pose.tx_px) / TOL_TX_PX,
        abs(pose.ty_px) / TOL_TY_PX,
    )


def format_drift_breakdown(pose: Pose) -> str:
    """Human-readable per-axis drift block, each line tagged with its
    ratio against tolerance so the operator can see at a glance which
    axis is the worst offender.
    """
    lines = []
    for label, value, tol, unit in (
        ("roll", pose.roll_deg, TOL_ROLL_DEG, "°"),
        ("tx  ", pose.tx_px, TOL_TX_PX, "px"),
        ("ty  ", pose.ty_px, TOL_TY_PX, "px"),
    ):
        ratio = abs(value) / tol
        within = "OK" if ratio <= 1.0 else f"{ratio:.1f}× over"
        lines.append(
            f"  {label}  {value:+7.2f}{unit}  (tol {tol:g}{unit} — {within})"
        )
    return "\n".join(lines)


def evaluate_pose(captured: np.ndarray, reference: np.ndarray) -> tuple[Pose, bool, str]:
    """Compute pose and verdict.

    Returns (pose, ok, message). `ok` is True for both OK and MARGINAL
    verdicts — callers should proceed in both cases but print the message
    when it contains "MARGINAL" so the operator knows the setup drifted a
    bit. `ok` is False only for true DRIFT (beyond DRIFT_MULTIPLIER× tol)
    or low confidence.
    """
    h, w = reference.shape[:2]
    mask = top_band_mask(h, w)
    pose = measure_pose_topband(captured, reference, mask)
    if pose.low_conf:
        return pose, False, (
            f"pose check: view obstructed / too few features "
            f"(inliers={pose.n_inliers}/{pose.n_matches})"
        )

    worst_ratio = worst_tolerance_ratio(pose)

    if worst_ratio <= 1.0:
        tag = "OK"
        ok = True
    elif worst_ratio <= DRIFT_MULTIPLIER:
        tag = f"MARGINAL (worst {worst_ratio:.1f}× tol, within {DRIFT_MULTIPLIER:g}×; proceeding)"
        ok = True
    else:
        tag = f"DRIFT (worst {worst_ratio:.1f}× tol, beyond {DRIFT_MULTIPLIER:g}×)"
        ok = False

    msg = (
        f"pose {tag}: roll={pose.roll_deg:+.2f}° (tol {TOL_ROLL_DEG}°)  "
        f"tx={pose.tx_px:+.2f}px ty={pose.ty_px:+.2f}px (tol {TOL_TX_PX}px)  "
        f"inliers={pose.n_inliers}/{pose.n_matches}"
    )
    return pose, ok, msg


def load_reference() -> np.ndarray:
    if not REFERENCE_PATH.exists():
        raise RuntimeError(
            f"topdown reference image missing: {REFERENCE_PATH}\n"
            f"Run: uv run python scripts/save_topdown_reference.py"
        )
    img = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read {REFERENCE_PATH}")
    return img
