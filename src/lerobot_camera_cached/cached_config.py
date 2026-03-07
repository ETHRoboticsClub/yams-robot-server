from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

@CameraConfig.register_subclass("opencv-cached")
@dataclass
class OpenCVCameraConfigCached(OpenCVCameraConfig):
    pass