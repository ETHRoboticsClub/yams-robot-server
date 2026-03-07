import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import portal
import yaml
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_yams.robot_core.yams_server import run_robot_server

logger = logging.getLogger(__name__)
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"


@RobotConfig.register_subclass("yams_follower")
@dataclass
class YamsFollowerConfig(RobotConfig):
    can_port: str
    server_port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper: str = "linear_3507"
    side: str = "right"
    effort_calibration_path: str | None = None
    effort_calibration_samples: int = 100
    effort_calibration_dt: float = 0.01
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
        self._effort_offsets = self._load_effort_offsets()

    def _calibration_path(self) -> Path:
        if self.config.effort_calibration_path is not None:
            return Path(self.config.effort_calibration_path)
        return CALIBRATION_DIR / f"follower_effort_{self.config.side}.yaml"

    def _load_effort_offsets(self) -> dict[str, float]:
        path = self._calibration_path()
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return {k: float(v) for k, v in data.get("offsets", {}).items()}

    def _save_effort_offsets(self) -> None:
        path = self._calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"offsets": self._effort_offsets}, f, sort_keys=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"{joint_name}.pos": float for joint_name in self.config.joint_names},
            **{f"{joint_name}.eff": float for joint_name in self.config.joint_names},
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
        return len(self._effort_offsets) == len(self.config.joint_names)

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        samples = []
        for _ in range(self.config.effort_calibration_samples):
            obs = self._client.get_observations().result()  # type: ignore
            samples.append(np.asarray(obs["joint_eff"], dtype=float))
            time.sleep(self.config.effort_calibration_dt)

        offsets = np.median(np.stack(samples), axis=0)
        self._effort_offsets = {
            joint_name: float(offset)
            for joint_name, offset in zip(self.config.joint_names, offsets)
        }
        self._save_effort_offsets()
        logger.info("%s effort calibrated: %s", self, self._effort_offsets)

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm state
        start = time.perf_counter()

        obs_dict = {}
        obs = self._client.get_observations().result()  # type: ignore
        joint_pos = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.array([]))])
        joint_eff = obs["joint_eff"]
        for i, key in enumerate(self.config.joint_names):
            obs_dict[f"{key}.pos"] = joint_pos[i]
            obs_dict[f"{key}.eff"] = joint_eff[i] - self._effort_offsets.get(key, 0.0)

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
