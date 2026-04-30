from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@CameraConfig.register_subclass("opencv-cached")
@dataclass
class OpenCVCameraCachedConfig(OpenCVCameraConfig):
    auto_exposure_enabled: bool = False
    auto_exposure_target: float = 110.0
    auto_exposure_deadband: float = 8.0
    auto_exposure_min: int = 5
    auto_exposure_max: int = 200
    auto_exposure_speed: float = 0.25
    auto_exposure_period_s: float = 0.5
    shm_key: str | None = None
