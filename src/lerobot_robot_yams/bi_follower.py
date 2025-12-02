import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@RobotConfig.register_subclass("bi_yams_follower")
@dataclass
class BiYamsFollowerConfig(RobotConfig):
    left_arm_can_port: str = "can_follower_l"
    left_arm_server_port: int = 11333
    right_arm_can_port: str = "can_follower_r"
    right_arm_server_port: int = 11334
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
        )
        right_arm_config = YamsFollowerConfig(
            can_port=self.config.right_arm_can_port,
            server_port=self.config.right_arm_server_port,
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        self.left_arm = YamsFollower(left_arm_config)
        self.right_arm = YamsFollower(right_arm_config)

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

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
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
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
