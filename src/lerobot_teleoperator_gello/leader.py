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
ARMS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "arms.yaml"


def _ensure_xm430_w210_support() -> None:
    model = "xm430-w210"
    base = "xm430-w350"
    if model in DynamixelMotorsBus.model_ctrl_table:
        return
    DynamixelMotorsBus.model_ctrl_table[model] = DynamixelMotorsBus.model_ctrl_table[
        base
    ]
    DynamixelMotorsBus.model_baudrate_table[model] = (
        DynamixelMotorsBus.model_baudrate_table[base]
    )
    DynamixelMotorsBus.model_encoding_table[model] = (
        DynamixelMotorsBus.model_encoding_table[base]
    )
    DynamixelMotorsBus.model_resolution_table[model] = (
        DynamixelMotorsBus.model_resolution_table[base]
    )
    DynamixelMotorsBus.model_number_table[model] = 1030


_ensure_xm430_w210_support()


def _load_motors(side: str) -> dict[str, Motor]:
    with open(ARMS_CONFIG_PATH, "r") as f:
        leader_config = yaml.safe_load(f)["leader"]
    motor_configs = leader_config[f"{side}_arm"]["motors"]
    return {
        name: Motor(
            cfg["id"],
            cfg["model"],
            MotorNormMode[cfg.get("norm_mode", "DEGREES")],
        )
        for name, cfg in motor_configs.items()
    }


@TeleoperatorConfig.register_subclass("yams_leader")
@dataclass
class YamsLeaderConfig(TeleoperatorConfig):
    port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    gripper_scale: float = 1.0
    calibration_path: str = "src/lerobot_teleoperator_gello/calibration"
    side: str = "right"


class YamsLeader(Teleoperator):
    config_class = YamsLeaderConfig
    name = "yams_leader"

    def __init__(self, config: YamsLeaderConfig):
        super().__init__(config)
        self.config = config
        motors = _load_motors(self.config.side)

        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors=motors,
        )
        calibration_path = (
            Path(self.config.calibration_path)
            / f"leader_calibration_{self.config.side}.yaml"
        )
        if calibration_path.exists():
            with open(calibration_path, "r") as f:
                self.calibration = yaml.safe_load(f)
            mtime = calibration_path.stat().st_mtime
            import datetime
            ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{self.config.side} leader] Loaded calibration from {calibration_path} (saved {ts})")
            for joint, offset in self.calibration.get("offsets", {}).items():
                print(f"  {joint}: offset={offset}")
        else:
            self.calibration = None

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

        # Retry handshake AND configure() together — the Dynamixel bus is
        # currently flaky (cut cable into a wrist motor) and both the
        # handshake (broadcast ping) and the configure-time writes drop
        # ~30-80% of the time. Each attempt is independent, so retrying
        # the whole connect+configure sequence usually lands within a few tries.
        last_error: Exception | None = None
        for attempt in range(1, 11):
            try:
                self.bus.connect()
                self.configure()
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self} connect attempt {attempt}/10 failed: {e}"
                )
                try:
                    if self.bus.is_connected:
                        self.bus.disconnect()
                except Exception:
                    pass
                time.sleep(0.2)
        else:
            raise last_error if last_error else RuntimeError(
                f"{self} failed to connect after 10 attempts"
            )

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

        if self.calibration is None:
            raise ValueError(
                "Calibration not found. Run `compute_offsets.py` to generate it."
            )

        start = time.perf_counter()

        try:
            raw_positions = self.bus.sync_read(
                normalize=False,
                data_name="Present_Position",
                num_retry=10,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read leader action from {self}") from e

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
        action["gripper.pos"] = np.clip(
            (action["gripper.pos"] - self.config.gripper_closed_pos) / gripper_range * self.config.gripper_scale,
            0.0, 1.0,
        )

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
