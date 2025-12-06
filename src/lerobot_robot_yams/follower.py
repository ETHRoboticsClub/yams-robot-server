import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import portal
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_yams.robot_core.yams_server import run_robot_server

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("yams_follower")
@dataclass
class YamsFollowerConfig(RobotConfig):
    can_port: str
    server_port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper: str = "linear_3507"
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
        self._client = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
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
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._client is not None
            and self._client.get_robot_info().result() is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        ctx = mp.get_context("spawn")
        self._robot_process = ctx.Process(
            target=run_robot_server,
            args=(self.config,),
        )
        self._robot_process.start()

        self._client = portal.Client(f"localhost:{self.config.server_port}")

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_dict = {}
        for i, key in enumerate(self.config.joint_names):
            joint_pos = self._client.get_joint_pos().result()  # type: ignore
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
        self._client.command_joint_pos(goal_pos)  # type: ignore

        return action

    def disconnect(self):
        from lerobot_robot_yams.utils.utils import slow_move

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        zero_pos = {f"{n}.pos": 0.0 for n in self.config.joint_names}
        slow_move(self, zero_pos, duration=2.0)

        self._client.close()
        self._robot_process.terminate()
        self._robot_process.join()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
