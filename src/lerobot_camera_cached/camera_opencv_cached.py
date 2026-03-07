from .cached_config import OpenCVCameraConfigCached
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from typing import Any

class OpenCVCameraCached(OpenCVCamera):
    def __init__(self, config: OpenCVCameraConfigCached):
        super().__init__(config)
        self.config = config

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        with self.frame_lock:
            frame = self.latest_frame
            return frame