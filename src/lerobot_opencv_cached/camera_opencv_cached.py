from lerobot.cameras.opencv import OpenCVCamera
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from typing import Any

class OpenCVCameraCached(OpenCVCamera):
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        with self.frame_lock:
            frame = self.latest_frame
            return frame