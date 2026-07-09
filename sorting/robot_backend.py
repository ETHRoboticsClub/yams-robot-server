"""
Robot backends. The pipeline talks to a backend, never to hardware directly, so
mock and real are drop-in interchangeable.

RealBackend wraps the two per-arm `portal.Client` servers (ports from
configs/arms.yaml), the same interface move_arms_zero.py uses. It exposes:
  - get_joints(arm)             read current joints (+gripper)
  - slow_move_to(arm, target)   interpolate joints at fixed freq (the place move)
  - set_gripper(arm, value)
  - get_frame(camera)           latest camera frame (via LeRobot robot, optional)

MockBackend logs intents and holds fake state, so the whole loop and dashboard
run with no robot attached.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger("sorting.backend")

_ARMS_YAML = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"


class MockBackend:
    name = "mock"

    def __init__(self, dof_per_arm: int = 7):
        self.dof = dof_per_arm
        self._joints = {"left": np.zeros(dof_per_arm), "right": np.zeros(dof_per_arm)}
        self.connected = False

    def connect(self) -> None:
        self.connected = True
        log.info("[mock] backend connected")

    def disconnect(self) -> None:
        self.connected = False

    def get_joints(self, arm: str) -> np.ndarray:
        return self._joints[arm].copy()

    def slow_move_to(self, arm: str, target: list[float], duration: float = 2.0) -> None:
        target = np.asarray(target, dtype=float)
        log.info("[mock] %s slow_move_to %s over %.1fs", arm, np.round(target, 3), duration)
        self._joints[arm] = target

    def set_gripper(self, arm: str, value: float) -> None:
        log.info("[mock] %s gripper -> %.2f", arm, value)
        self._joints[arm][-1] = value

    def get_frame(self, camera: str) -> np.ndarray | None:
        return None


class RealBackend:
    """Wraps the i2rt/portal joint servers. Import-guarded so mock mode never
    needs portal or a live robot."""

    name = "real"

    def __init__(self, arms_yaml: Path = _ARMS_YAML, freq: float = 200.0):
        self.freq = freq
        cfg = yaml.safe_load(Path(arms_yaml).read_text())
        self._ports = {
            "left": int(cfg["follower"]["left_arm"]["server_port"]),
            "right": int(cfg["follower"]["right_arm"]["server_port"]),
        }
        self._clients: dict[str, Any] = {}
        self.connected = False

    def connect(self) -> None:
        import portal  # deferred: only needed for real hardware
        for arm, port in self._ports.items():
            client = portal.Client(f"localhost:{port}")
            client.get_robot_info().result()
            self._clients[arm] = client
        self.connected = True
        log.info("[real] connected to arms %s", list(self._clients))

    def disconnect(self) -> None:
        for c in self._clients.values():
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()
        self.connected = False

    def get_joints(self, arm: str) -> np.ndarray:
        obs = self._clients[arm].get_observations().result()
        return np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.array([]))])

    def slow_move_to(self, arm: str, target: list[float], duration: float = 2.0) -> None:
        client = self._clients[arm]
        current = self.get_joints(arm)
        target = np.asarray(target, dtype=float)
        if target.shape != current.shape:
            raise ValueError(f"target dof {target.shape} != current {current.shape}")
        steps = max(int(duration * self.freq), 1)
        for i in range(steps + 1):
            alpha = i / steps
            client.command_joint_pos((1 - alpha) * current + alpha * target)
            time.sleep(1 / self.freq)

    def set_gripper(self, arm: str, value: float) -> None:
        cur = self.get_joints(arm)
        cur[-1] = value
        self._clients[arm].command_joint_pos(cur)

    def get_frame(self, camera: str) -> np.ndarray | None:
        # Frames come from the LeRobot robot in real integration; the joint
        # servers don't carry video. Perception owns the camera in real mode.
        return None


def make_backend(kind: str) -> Any:
    if kind == "real":
        return RealBackend()
    return MockBackend()
