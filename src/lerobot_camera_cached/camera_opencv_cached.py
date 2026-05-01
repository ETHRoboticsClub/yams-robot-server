import logging
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from typing import Any

from lerobot_camera_cached.cached_config import OpenCVCameraCachedConfig
from utils.camera_auto_exposure import CameraAutoExposure, get_exposure

logger = logging.getLogger(__name__)


class OpenCVCameraCached(OpenCVCamera):
    def __init__(self, config: OpenCVCameraCachedConfig):
        super().__init__(config)
        self.config = config
        self.ready = False
        self.latest_frame_time = 0.0
        self.last_frame = np.zeros([self.config.height, self.config.width, 3], np.uint8)
        self.auto_exposure = self._build_auto_exposure()

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

        frame = self.latest_frame
        if (
            self.ready
            and frame is not None
            and time.monotonic() - self.latest_frame_time <= timeout_ms / 1000.0
        ):
            self.last_frame = frame
            return frame

        timeout_s = timeout_ms / 1000.0
        if self.new_frame_event.wait(timeout=timeout_s):
            frame = self.latest_frame
            if frame is not None:
                self.ready = True
                self.latest_frame_time = time.monotonic()
                self.last_frame = frame
                return frame

        if self.last_frame is not None:
            logger.warning(f"{self} frame timeout, returning last frame (reconnecting?)")
            return self.last_frame

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

    def _build_auto_exposure(self) -> CameraAutoExposure | None:
        if isinstance(self.index_or_path, int):
            return None
        device = Path(self.index_or_path).resolve()
        if not self.config.auto_exposure_enabled or not str(device).startswith("/dev/video"):
            return None
        try:
            exposure = get_exposure(device)
        except Exception:
            logger.warning(f"{device} does not support exposure_time_absolute, disabling auto-exposure")
            return None
        return CameraAutoExposure(
            device=device,
            exposure=exposure,
            target=self.config.auto_exposure_target,
            deadband=self.config.auto_exposure_deadband,
            speed=self.config.auto_exposure_speed,
            min_exposure=self.config.auto_exposure_min,
            max_exposure=self.config.auto_exposure_max,
            period_s=self.config.auto_exposure_period_s,
        )

    def _reconnect_hardware(self, max_attempts: int = 10, retry_delay: float = 2.0) -> bool:
        """Re-open the VideoCapture after a USB reset without touching the read thread."""
        logger.warning(f"{self} attempting hardware reconnect...")
        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
        self.ready = False

        for attempt in range(max_attempts):
            if self.stop_event is not None and self.stop_event.is_set():
                return False
            time.sleep(retry_delay)
            try:
                cap = cv2.VideoCapture(self.index_or_path, self.backend)
                if not cap.isOpened():
                    cap.release()
                    logger.warning(f"{self} reconnect attempt {attempt + 1}/{max_attempts}: device not ready")
                    continue
                self.videocapture = cap
                self._configure_capture_settings()
                self.auto_exposure = self._build_auto_exposure()
                logger.info(f"{self} reconnected on attempt {attempt + 1}.")
                return True
            except Exception as e:
                logger.warning(f"{self} reconnect attempt {attempt + 1}/{max_attempts} failed: {e}")

        logger.error(f"{self} failed to reconnect after {max_attempts} attempts.")
        return False

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()

                if self.auto_exposure is not None:
                    try:
                        exposure = self.auto_exposure.tick(processed_frame)
                        if exposure is not None:
                            logger.info("%s exposure_time_absolute=%s", self, exposure)
                    except Exception as e:
                        logger.warning(f"{self} auto-exposure error, disabling: {e}")
                        self.auto_exposure = None

                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
                if failure_count > 10:
                    if not self._reconnect_hardware():
                        break
                    failure_count = 0


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
