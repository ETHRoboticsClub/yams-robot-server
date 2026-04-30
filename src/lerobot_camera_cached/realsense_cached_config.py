from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@CameraConfig.register_subclass("intelrealsense-cached")
@dataclass
class RealSenseCameraCachedConfig(RealSenseCameraConfig):
    profile_path: str | Path | None = str(Path(__file__).resolve().parents[2] / "configs" / "realsense.json")
    shm_key: str | None = None
