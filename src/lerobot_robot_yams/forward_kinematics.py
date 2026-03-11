"""Forward kinematics for YAM arm, parsed from dual_yam.urdf using only numpy."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_URDF = Path(__file__).parents[2] / "urdf" / "dual_yam.urdf"

# Joint chain for a single arm in order from base to tip (revolute joints only).
_JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _parse_urdf(urdf_path: Path = _URDF) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (xyz, rpy) for each joint in _JOINT_ORDER, from the URDF."""
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.iter("joint")}
    frames = []
    for name in _JOINT_ORDER:
        origin = by_name[name].find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        frames.append((xyz, rpy))
    return frames


_JOINT_FRAMES = _parse_urdf()


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _make_tf(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def arm_fk(joint_angles: np.ndarray) -> list[np.ndarray]:
    """Return world-frame positions of each link origin (after each joint).

    Parameters
    ----------
    joint_angles : (6,) array of joint angles in radians (joints 1-6, no gripper).

    Returns
    -------
    List of 7 xyz positions: base, then after each of the 6 joints.
    The last transform T is also returned as the 8th element (4x4 matrix) so
    callers can compute points along the last link.
    """
    T = np.eye(4)
    positions = [T[:3, 3].copy()]

    for (xyz, rpy), q in zip(_JOINT_FRAMES, joint_angles):
        # Fixed part of the joint frame
        T_joint = _make_tf(_rpy_to_rot(*rpy), xyz)
        # Revolve about local z by q
        T_q = _make_tf(_rot_z(q), np.zeros(3))
        T = T @ T_joint @ T_q
        positions.append(T[:3, 3].copy())

    return positions, T


def check_action(
    joint_angles: np.ndarray,
    last_joint_angles: np.ndarray | None,
    ground_z: float,
    end_effector_length: float,
    max_joint_step: np.ndarray,
) -> bool:
    """Return True (rejected) if:
    - joint1 (base rotation) exceeds ±90°,
    - any joint moves more than max_joint_step from last_joint_angles, or
    - any link or the last-link volume is below ground_z.
    """
    if abs(joint_angles[0]) > np.pi / 2:
        return True

    if last_joint_angles is not None and np.any(np.abs(joint_angles - last_joint_angles) > max_joint_step):
        diffs = joint_angles - last_joint_angles
        exceeded = np.abs(diffs) > max_joint_step
        for i, (prev_a, new_a, diff) in enumerate(zip(last_joint_angles, joint_angles, diffs)):
            if exceeded[i]:
                print(f"[Joint Step Warning] Joint {i}: prev={prev_a:.4f}, new={new_a:.4f}, Δ={diff:.4f} (limit={max_joint_step:.4f})")
        return True

    positions, T_tip = arm_fk(joint_angles)

    if any(pos[2] < ground_z for pos in positions[1:]):
        return True

    local_z_world = T_tip[:3, 2]
    for t in np.linspace(0, end_effector_length, 10):
        if (positions[-1] + t * local_z_world)[2] < ground_z:
            return True

    return False
