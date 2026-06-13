import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import portal
import yaml
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_robot_yams.robot_core.yams_server import run_robot_server

# from i2rt.robots.get_robot import get_yam_robot
# from i2rt.robots.utils import GripperType

logger = logging.getLogger(__name__)

@RobotConfig.register_subclass("yams_follower")
@dataclass
class YamsFollowerConfig(RobotConfig):
    can_port: str
    server_port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper: str = "linear_3507"
    side: str = "right"
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
    # Follower-side command smoothing: upsample the ~5 Hz policy command
    # stream to `smooth_hz` by interpolating between waypoints in a background
    # thread (see YamsFollower._smooth_loop). Set smooth=False for the old
    # pass-through behaviour.
    smooth: bool = True
    smooth_hz: float = 30.0


class YamsFollower(Robot):
    config_class = YamsFollowerConfig
    name = "yams_follower"

    def __init__(self, config: YamsFollowerConfig):
        super().__init__(config)
        self.config = config
        self._client = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connected_once = False

        # --- Follower-side command smoother -------------------------------
        # The policy commands joint targets at the lerobot tick rate (~5 Hz).
        # Sent straight through, each target is a step change the motor chain's
        # PD loop chases with a velocity spike -> visible jerk every ~200 ms.
        # Instead we upsample to `smooth_hz` (default 30 Hz): a background
        # thread linearly interpolates from the last commanded position to the
        # latest target over the measured inter-target interval, so the PD loop
        # sees a smoothly moving setpoint. send_action just updates the target.
        self._smooth_lock = threading.Lock()
        self._smooth_thread: threading.Thread | None = None
        self._smooth_stop = threading.Event()
        self._smooth_shutdown = False
        self._seg_start: np.ndarray | None = None
        self._seg_target: np.ndarray | None = None
        self._seg_t0: float = 0.0
        self._seg_dur: float = 1.0 / 5.0
        self._last_cmd: np.ndarray | None = None
        self._last_target_t: float | None = None

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
        
        
        if self.connected_once:
            return True
        else:
            connected = (
                self._client is not None
                and self._client.get_robot_info().result() is not None
                and all(cam.is_connected for cam in self.cameras.values())
            )
            if connected:
                self.connected_once = True
            return connected

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
        return

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
            [action[f"{joint_name}.pos"] for joint_name in self.config.joint_names],
            dtype=np.float64,
        )

        # Pass-through when smoothing is off or during shutdown (slow_move
        # streams its own dense trajectory and must reach the motors directly).
        if not self.config.smooth or self._smooth_shutdown:
            self._client.command_joint_pos(goal_pos)  # type: ignore
            return action

        if self._smooth_thread is None:
            self._start_smoother()

        now = time.perf_counter()
        with self._smooth_lock:
            start = self._last_cmd if self._last_cmd is not None else goal_pos
            if self._last_target_t is not None:
                # Upsample whatever the actual command rate is: reach each new
                # target over the interval since the previous one (clamped).
                self._seg_dur = float(np.clip(now - self._last_target_t, 0.05, 0.5))
            self._seg_start = start
            self._seg_target = goal_pos
            self._seg_t0 = now
            self._last_target_t = now
        return action

    def _read_current_joints(self) -> np.ndarray:
        """Current joint+gripper position, in the same space as send_action targets."""
        obs = self._client.get_observations().result()  # type: ignore
        return np.concatenate(
            [obs["joint_pos"], obs.get("gripper_pos", np.array([]))]
        ).astype(np.float64)

    def _start_smoother(self) -> None:
        try:
            current = self._read_current_joints()
        except Exception as exc:  # seed lazily on first command if the read fails
            logger.warning(f"{self} smoother seed read failed: {exc}")
            current = None
        with self._smooth_lock:
            self._last_cmd = current
            self._seg_start = current
            self._seg_target = current
            self._seg_t0 = time.perf_counter()
        self._smooth_stop.clear()
        self._smooth_thread = threading.Thread(
            target=self._smooth_loop,
            name=f"yams_smoother_{self.config.side}",
            daemon=True,
        )
        self._smooth_thread.start()

    def _smooth_loop(self) -> None:
        period = 1.0 / max(1.0, self.config.smooth_hz)
        while not self._smooth_stop.is_set():
            tick = time.perf_counter()
            cmd = None
            with self._smooth_lock:
                if self._seg_target is not None and self._seg_start is not None:
                    if self._seg_dur <= 0:
                        frac = 1.0
                    else:
                        frac = min(1.0, (tick - self._seg_t0) / self._seg_dur)
                    cmd = self._seg_start + frac * (self._seg_target - self._seg_start)
                    self._last_cmd = cmd
            if cmd is not None:
                try:
                    self._client.command_joint_pos(cmd)  # type: ignore
                except Exception as exc:
                    logger.warning(f"{self} smoother command failed: {exc}")
            elapsed = time.perf_counter() - tick
            self._smooth_stop.wait(max(0.0, period - elapsed))

    def _stop_smoother(self) -> None:
        if self._smooth_thread is not None:
            self._smooth_stop.set()
            self._smooth_thread.join(timeout=1.0)
            self._smooth_thread = None

    def disconnect(self):
        from lerobot_robot_yams.utils.utils import slow_move

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop upsampling and route slow_move straight to the motors.
        self._smooth_shutdown = True
        self._stop_smoother()
        zero_pos = {f"{n}.pos": 0.0 for n in self.config.joint_names}
        slow_move(self, zero_pos, duration=2.0)

        self._client.close()
        self._robot_process.terminate()
        self._robot_process.join()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
