import logging
import time
from collections import deque
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
# from utils.terminal_status import TerminalStatus

logger = logging.getLogger(__name__)
# RED_DOT = "\033[31m●\033[0m"
# GREEN_DOT = "\033[32m●\033[0m"
GRIPPER_GUARD_EFFORT = 0.25
GRIPPER_GUARD_VEL = 0.6
GRIPPER_REOPEN_STEP = 0.01

_ARMS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "arms.yaml"
_COLLISION = yaml.safe_load(_ARMS_CONFIG_PATH.read_text())["collision"]


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
        # self._status = TerminalStatus(interval=0.2)
        self._eff_history = {"left": deque(), "right": deque()}
        self._eff_max = {"left": 0.0, "right": 0.0}
        self._eff_dt = deque(maxlen=100)
        self._last_eff_t: float | None = None
        self._effort_calibration_started = False
        self._gripper_state = {
            "left": {"eff": 0.0, "vel": 0.0, "pos": 1.0},
            "right": {"eff": 0.0, "vel": 0.0, "pos": 1.0},
        }
        self._gripper_hold = {
            "left": {"active": False, "pos": 1.0},
            "right": {"active": False, "pos": 1.0},
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{motor}.pos": float for motor in self.left_arm.config.joint_names},
            **{f"left_{motor}.eff": float for motor in self.left_arm.config.joint_names},
            "left_gripper.vel": float,
            **{f"right_{motor}.pos": float for motor in self.right_arm.config.joint_names},
            **{f"right_{motor}.eff": float for motor in self.right_arm.config.joint_names},
            "right_gripper.vel": float,
        }

    @property
    def _action_ft(self) -> dict[str, type]:
        return {
            **{f"left_{motor}.pos": float for motor in self.left_arm.config.joint_names},
            **{f"right_{motor}.pos": float for motor in self.right_arm.config.joint_names},
        }

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
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self) -> None:
        for cam in self.cameras.values():
            cam.connect()

        self.left_arm.connect()
        self.right_arm.connect()
        self._effort_calibration_started = False

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def _wait_until_still(
        self,
        settle_s: float = 0.75,
        timeout_s: float = 8.0,
        pos_tol: float = 0.002,
        eff_tol: float = 0.02,
    ) -> None:
        history = deque()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            left_obs = self.left_arm.robot.get_observations()  # type: ignore
            right_obs = self.right_arm.robot.get_observations()  # type: ignore
            now = time.perf_counter()
            pos = np.concatenate([
                left_obs["joint_pos"],
                left_obs.get("gripper_pos", np.array([])),
                right_obs["joint_pos"],
                right_obs.get("gripper_pos", np.array([])),
            ])
            eff = np.array([left_obs["joint_eff"][-1], right_obs["joint_eff"][-1]])
            history.append((now, pos, eff))
            while history and now - history[0][0] > settle_s:
                history.popleft()
            if history and now - history[0][0] >= settle_s:
                pos_window = np.stack([sample[1] for sample in history])
                eff_window = np.stack([sample[2] for sample in history])
                if np.max(np.ptp(pos_window, axis=0)) < pos_tol and np.max(np.ptp(eff_window, axis=0)) < eff_tol:
                    return
            time.sleep(0.02)

    def calibrate(self) -> None:
        self.left_arm.start_effort_calibration()
        self.right_arm.start_effort_calibration()
        self._effort_calibration_started = True

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def _update_effort_max(self, left_eff: float, right_eff: float) -> None:
        now = time.perf_counter()
        if self._last_eff_t is not None:
            self._eff_dt.append(now - self._last_eff_t)
        self._last_eff_t = now
        hz = 1 / max(np.mean(self._eff_dt), 1e-6) if self._eff_dt else 0.0
        window = max(1, round(hz * 0.2))
        for side, eff in [("left", abs(left_eff)), ("right", abs(right_eff))]:
            history = self._eff_history[side]
            history.append(eff)
            while len(history) > window:
                history.popleft()
            self._eff_max[side] = max(self._eff_max[side], sum(history) / len(history))

    def _guard_gripper(self, side: str, action: dict[str, Any]) -> None:
        hold = self._gripper_hold[side]
        state = self._gripper_state[side]
        if hold["active"]:
            action["gripper.pos"] = max(action["gripper.pos"], hold["pos"])
            if action["gripper.pos"] > hold["pos"]:
                hold["active"] = False
            return
        if abs(state["vel"]) < GRIPPER_GUARD_VEL and abs(state["eff"]) > GRIPPER_GUARD_EFFORT:
            hold["active"] = True
            hold["pos"] = min(1.0, state["pos"] + GRIPPER_REOPEN_STEP)
            action["gripper.pos"] = max(action["gripper.pos"], hold["pos"])

    def get_observation(self, with_cameras=True) -> dict[str, Any]:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})
        self._gripper_state["left"] = {
            "eff": left_obs["gripper.eff"],
            "vel": left_obs["gripper.vel"],
            "pos": left_obs["gripper.pos"],
        }
        self._gripper_state["right"] = {
            "eff": right_obs["gripper.eff"],
            "vel": right_obs["gripper.vel"],
            "pos": right_obs["gripper.pos"],
        }
        self._update_effort_max(left_obs["gripper.eff"], right_obs["gripper.eff"])
        # self._status.update(
        #     "gripper.eff"
        #     f"  left={left_obs['gripper.eff']:7.2f}"
        #     f"  qvel={left_obs['gripper.vel']:7.3f}"
        #     f"  right={right_obs['gripper.eff']:7.2f}"
        #     f"  qvel={right_obs['gripper.vel']:7.3f}"
        #     f"  max left={self._eff_max['left']:7.2f}"
        #     f"  right={self._eff_max['right']:7.2f}"
        #     f"  {RED_DOT if self._gripper_hold['left']['active'] else GREEN_DOT}"
        #     f"  {RED_DOT if self._gripper_hold['right']['active'] else GREEN_DOT}"
        # )
        # print(f"gripper.eff left={left_obs['gripper.eff']:.2f} right={right_obs['gripper.eff']:.2f}", flush=True)
        
        if with_cameras:
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



        joint_names_6 = self.left_arm.config.joint_names[:6]
        for side, arm_action in [("left", left_action), ("right", right_action)]:
            angles = np.array([arm_action[f"{j}.pos"] for j in joint_names_6])
            if check_action(angles, self._last_angles[side], self.config.ground_z, self.config.end_effector_length, self.config.max_joint_step):
                logger.warning(f"{side} arm action rejected")
                return self.get_observation(with_cameras=False)
            self._last_angles[side] = angles
            self._guard_gripper(side, arm_action)

        if not self._effort_calibration_started:
            logger.info("Starting follower effort calibration during first second of active control.")
            self.calibrate()

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
        # self._status.close()
        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(self.left_arm.disconnect)
            ex.submit(self.right_arm.disconnect)

        for cam in self.cameras.values():
            cam.disconnect()
