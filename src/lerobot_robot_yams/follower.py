import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

logger = logging.getLogger(__name__)

@RobotConfig.register_subclass("yams_follower")
@dataclass
class YamsFollowerConfig(RobotConfig):
    can_port: str
    server_port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper: str = "linear_3507"
    side: str = "right"
    effort_calibration_path: str | None = None
    effort_calibration_duration_s: float = 1.0
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
    )


class YamsFollower(Robot):
    config_class = YamsFollowerConfig
    name = "yams_follower"

    def __init__(self, config: YamsFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"{joint_name}.pos": float for joint_name in self.config.joint_names},
        }

    @property
    def _action_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

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
        return self._action_ft

    @property
    def is_connected(self) -> bool:
        return ( all(cam.is_connected for cam in self.cameras.values()))

    def connect(self) -> None:
        gripper_type = GripperType.from_string_name(self.config.gripper)
        self.robot = get_yam_robot(channel=self.config.can_port, gripper_type=gripper_type)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm state
        start = time.perf_counter()

        obs_dict = {}
        obs = self.robot.get_observations()
        joint_pos = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.array([]))])
        for i, key in enumerate(self.config.joint_names):
            obs_dict[f"{key}.pos"] = joint_pos[i]

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = np.array(
            [action[f"{joint_name}.pos"] for joint_name in self.config.joint_names]
        )
        self.robot.command_joint_pos(goal_pos)

        return action

    def disconnect(self):
        from lerobot_robot_yams.utils.utils import slow_move

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        zero_pos = {f"{n}.pos": 0.0 for n in self.config.joint_names}
        slow_move(self, zero_pos, duration=2.0)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
