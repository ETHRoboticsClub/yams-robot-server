from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("zed")
@dataclass
class ZEDCameraConfig(CameraConfig):
    camera_id: int = 0
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.ROTATE_180
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_mode: str = "PERFORMANCE"

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` must be {ColorMode.RGB.value} or {ColorMode.BGR.value}, got {self.color_mode}"
            )
