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

    return positions


def check_ground_collision(
    joint_angles: np.ndarray,
    ground_z: float = 0.0,
    margin: float = 0.01,
) -> bool:
    """Return True if any link origin (excluding base) is below ground_z + margin.

    The base_link origin sits at z=0 on the table surface.
    Both arms share the same chain (arm2 offset is purely in y).
    """
    return any(pos[2] < ground_z + margin for pos in arm_fk(joint_angles)[1:])
