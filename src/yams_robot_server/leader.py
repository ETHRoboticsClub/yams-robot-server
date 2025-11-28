import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("yams_leader")
@dataclass
class YamsLeaderConfig(TeleoperatorConfig):
    port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    calibration_path: str = "src/yams_robot_server"
    side: str = "right"


class YamsLeader(Teleoperator):
    config_class = YamsLeaderConfig
    name = "yams_leader1"

    def __init__(self, config: YamsLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "xl330-m077", MotorNormMode.DEGREES),
                "joint_2": Motor(2, "xl330-m077", MotorNormMode.DEGREES),
                "joint_3": Motor(3, "xl330-m077", MotorNormMode.DEGREES),
                "joint_4": Motor(4, "xl330-m077", MotorNormMode.DEGREES),
                "joint_5": Motor(5, "xl330-m077", MotorNormMode.DEGREES),
                "joint_6": Motor(6, "xl330-m077", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.DEGREES),
            },
        )
        with open(
            Path(self.config.calibration_path)
            / f"leader_calibration_{self.config.side}.yaml",
            "r",
        ) as f:
            self.calibration = yaml.safe_load(f)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.configure()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Enable torque and set to position to open
        self.bus.write("Torque_Enable", "gripper", 0, normalize=False)
        self.bus.write(
            "Operating_Mode",
            "gripper",
            OperatingMode.CURRENT_POSITION.value,
            normalize=False,
        )
        self.bus.write("Current_Limit", "gripper", 100, normalize=False)
        self.bus.write("Torque_Enable", "gripper", 1, normalize=False)
        self.bus.write(
            "Goal_Position", "gripper", self.config.gripper_open_pos, normalize=False
        )

    def setup_motors(self) -> None:
        for motor in self.bus.motors:
            input(
                f"Connect the controller board to the '{motor}' motor only and press enter."
            )
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        raw_positions = self.bus.sync_read(
            normalize=False, data_name="Present_Position"
        )
        calibration_offsets = self.calibration.get("offsets", {})
        calibration_scales = self.calibration.get("scales", {})
        action = {}

        for motor, raw_val in raw_positions.items():
            if motor == "gripper":
                action[f"{motor}.pos"] = raw_val
                continue

            offset = calibration_offsets.get(motor, 0)
            scale = calibration_scales.get(motor, 1.0)
            pos = ((raw_val + offset) * scale) / 4096 * 2 * np.pi - np.pi
            # Scale pos to be between -pi and pi
            wrapped_pos = (pos + np.pi) % (2 * np.pi) - np.pi
            action[f"{motor}.pos"] = wrapped_pos

        # Normalize gripper position between 0 (closed) and 1 (open)
        gripper_range = self.config.gripper_open_pos - self.config.gripper_closed_pos
        action["gripper.pos"] = (
            action["gripper.pos"] - self.config.gripper_closed_pos
        ) / gripper_range

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
