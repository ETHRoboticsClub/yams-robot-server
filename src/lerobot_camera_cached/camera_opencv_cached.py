import logging
import time

import numpy as np
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from typing import Any

from lerobot_camera_cached.cached_config import OpenCVCameraCachedConfig

logger = logging.getLogger(__name__)


class OpenCVCameraCached(OpenCVCamera):
    def __init__(self, config: OpenCVCameraCachedConfig):
        super().__init__(config)
        self.config = config
        self.ready = False
        self.last_frame = np.zeros([self.config.height, self.config.width, 3], np.uint8)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.
        It is “best effort” under high FPS.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame as a NumPy array in the format
                       (height, width, channels), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        timeout_s = timeout_ms / 1000.0
        for _ in range(3):
            if self.new_frame_event.wait(timeout=timeout_s):
                frame = self.latest_frame
                if frame is not None:
                    self.last_frame = frame
                    return frame

            # Keep teleop alive on transient camera hiccups once we have a frame.
            if self.latest_frame is not None:
                return self.latest_frame

        raise TimeoutError(
            f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
            f"Read thread alive: {self.thread.is_alive()}."
        )

    def connect(self, warmup: bool = True) -> None:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                super().connect(warmup=warmup)
                return
            except Exception as e:
                last_error = e
                logger.warning(f"{self} connect attempt {attempt + 1}/3 failed: {e}")
                try:
                    self.disconnect()
                except Exception:
                    pass
                time.sleep(0.2)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{self} failed to connect.")


if __name__ == "__main__":
    config = OpenCVCameraCachedConfig(
        index_or_path="/dev/video0",
        fps=30,
        width=640,
        height=480,
    )

    cam = OpenCVCameraCached(config)
    cam.connect()
    img = cam.async_read()
    cam.disconnect()