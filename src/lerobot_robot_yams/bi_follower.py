import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig
from lerobot_robot_yams.forward_kinematics import check_action

logger = logging.getLogger(__name__)

_ARMS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "arms.yaml"
_COLLISION = yaml.safe_load(_ARMS_CONFIG_PATH.read_text())["collision"]


class CameraReadError(RuntimeError):
    pass


@RobotConfig.register_subclass("bi_yams_follower")
@dataclass
class BiYamsFollowerConfig(RobotConfig):
    left_arm_can_port: str = "can_follower_l"
    left_arm_server_port: int = 11333
    right_arm_can_port: str = "can_follower_r"
    right_arm_server_port: int = 11334
    ground_z: float = field(default_factory=lambda: _COLLISION["ground_z"])
    end_effector_length: float = field(default_factory=lambda: _COLLISION["end_effector_length"])
    max_joint_step: np.ndarray = field(default_factory=lambda: np.array(_COLLISION["max_joint_step"]))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class BiYamsFollower(Robot):
    """
    Bimanual I2RT Yams Follower Arms.
    """

    config_class = BiYamsFollowerConfig
    name = "bi_yams_follower"

    def __init__(self, config: BiYamsFollowerConfig):
        super().__init__(config)

        self.config = config

        left_arm_config = YamsFollowerConfig(
            can_port=self.config.left_arm_can_port,
            server_port=self.config.left_arm_server_port,
            side="left",
        )
        right_arm_config = YamsFollowerConfig(
            can_port=self.config.right_arm_can_port,
            server_port=self.config.right_arm_server_port,
            side="right",
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        self.left_arm = YamsFollower(left_arm_config)
        self.right_arm = YamsFollower(right_arm_config)
        self._last_angles: dict[str, np.ndarray | None] = {"left": None, "right": None}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"left_{motor}.pos": float for motor in self.left_arm.config.joint_names
        } | {f"right_{motor}.pos": float for motor in self.right_arm.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self) -> None:
        for cam in self.cameras.values():
            cam.connect()

        self.left_arm.connect()
        self.right_arm.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self, with_cameras=True) -> dict[str, Any]:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})
        
        if with_cameras:
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                try:
                    obs_dict[cam_key] = cam.async_read()
                except (TimeoutError, OSError) as exc:
                    raise CameraReadError(f"{cam_key} read failed: {exc}") from exc
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        left_action = {
            key.removeprefix("left_"): value
            for key, value in action.items()
            if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value
            for key, value in action.items()
            if key.startswith("right_")
        }



        joint_names_6 = self.left_arm.config.joint_names[:6]
        for side, arm_action in [("left", left_action), ("right", right_action)]:
            angles = np.array([arm_action[f"{j}.pos"] for j in joint_names_6])
            rejected, reason = check_action(
                angles,
                self._last_angles[side],
                self.config.ground_z,
                self.config.end_effector_length,
                self.config.max_joint_step,
            )
            if rejected:
                logger.warning(f"{side} arm action rejected: {reason}")
                return self.get_observation(with_cameras=False)
            self._last_angles[side] = angles

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        prefixed_send_action_left = {
            f"left_{key}": value for key, value in send_action_left.items()
        }
        prefixed_send_action_right = {
            f"right_{key}": value for key, value in send_action_right.items()
        }

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(self.left_arm.disconnect)
            ex.submit(self.right_arm.disconnect)

        for cam in self.cameras.values():
            cam.disconnect()
