import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from numpy.typing import NDArray

from .realsense_cached_config import RealSenseCameraCachedConfig
from utils.connection import _free_v4l_devices

logger = logging.getLogger(__name__)


class RealSenseCameraCached(RealSenseCamera):
    def __init__(self, config: RealSenseCameraCachedConfig):
        super().__init__(config)
        self.config = config
        self.ready = False
        self.latest_frame_time = 0.0
        self.last_frame = np.zeros([self.config.height, self.config.width, 3], np.uint8)

    def _find_device(self) -> Any:
        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                return device
        raise RuntimeError(f"RealSense device {self.serial_number} not found.")

    def _wait_for_device(self, timeout_s: float = 15.0, poll_s: float = 0.2) -> Any:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return self._find_device()
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(poll_s)

    def _reset_busy_device(self, device: Any) -> None:
        _free_v4l_devices("RealSense")
        if hasattr(device, "hardware_reset"):
            logger.warning("%s profile load still busy, resetting device", self)
            device.hardware_reset()
            time.sleep(1.0)
            self._wait_for_device(timeout_s=20.0)
        time.sleep(0.5)

    def _load_profile(self) -> None:
        profile_path = self.config.profile_path
        if not profile_path:
            return

        profile_text = Path(profile_path).read_text()
        for attempt in range(2):
            device = self._wait_for_device()
            advanced = rs.rs400_advanced_mode(device)
            if not advanced.is_enabled():
                advanced.toggle_advanced_mode(True)
                time.sleep(2.0)
                device = self._wait_for_device()
                advanced = rs.rs400_advanced_mode(device)
            try:
                advanced.load_json(profile_text)
                break
            except RuntimeError as exc:
                if attempt == 1 or "Device or resource busy" not in str(exc):
                    raise
                logger.warning("%s profile load busy, freeing RealSense video nodes and retrying", self)
                self._reset_busy_device(device)
        logger.info("Loaded RealSense profile from %s", profile_path)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        frame = self.latest_color_frame
        if (
            self.ready
            and frame is not None
            and time.monotonic() - self.latest_frame_time <= timeout_ms / 1000.0
        ):
            self.last_frame = frame
            return frame

        timeout_s = timeout_ms / 1000.0
        if self.new_frame_event.wait(timeout=timeout_s):
            frame = self.latest_color_frame
            if frame is not None:
                self.ready = True
                self.latest_frame_time = time.monotonic()
                self.last_frame = frame
                return frame

        raise TimeoutError(
            f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
            f"Read thread alive: {self.thread.is_alive()}."
        )

    def connect(self, warmup: bool = True) -> None:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self._load_profile()
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
