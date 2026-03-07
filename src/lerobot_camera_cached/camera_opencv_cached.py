import logging
import time

from lerobot.utils.errors import DeviceNotConnectedError

from .cached_config import OpenCVCameraCachedConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from typing import Any

logger = logging.getLogger(__name__)


class OpenCVCameraCached(OpenCVCamera):
    def __init__(self, config: OpenCVCameraCachedConfig):
        super().__init__(config)
        self.config = config
        self.ready = False

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if self.ready is False:
            while self.latest_frame is None:
                time.sleep(0.1)
                self.ready = True

        with self.frame_lock:
            frame = self.latest_frame
            return frame

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                self.latest_frame = color_image

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
