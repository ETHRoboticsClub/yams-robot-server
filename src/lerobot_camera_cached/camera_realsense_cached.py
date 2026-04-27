import json
import logging
import re
import threading
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
UNSUPPORTED_PROFILE_KEY = re.compile(r"([A-Za-z0-9_-]+) key is not supported")


def _remove_unsupported_key(profile_text: str, error: Exception) -> str | None:
    match = UNSUPPORTED_PROFILE_KEY.search(str(error))
    if not match:
        return None

    key = match.group(1)
    profile = json.loads(profile_text)
    params = profile.get("parameters", {})
    if key not in params:
        return None

    params.pop(key)
    logger.warning("Skipping unsupported RealSense profile key: %s", key)
    return json.dumps(profile)


class RealSenseCameraCached(RealSenseCamera):
    def __init__(self, config: RealSenseCameraCachedConfig):
        super().__init__(config)
        self.config = config
        self.ready = False
        self.latest_frame_time = 0.0
        self.last_frame = np.zeros([self.config.height, self.config.width, 3], np.uint8)
        # Depth snapshot captured atomically with the color frame returned by
        # async_read(). Callers (e.g. the record-with-depth sidecar writer)
        # pop it after each get_observation() so color and depth in the saved
        # dataset come from the same read-loop iteration.
        self._depth_snapshot_lock = threading.Lock()
        self._last_depth_snapshot: NDArray[Any] | None = None

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
        busy_retry = True
        for _ in range(20):
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
                cleaned = _remove_unsupported_key(profile_text, exc)
                if cleaned is not None:
                    profile_text = cleaned
                    continue
                if busy_retry and "Device or resource busy" in str(exc):
                    busy_retry = False
                    logger.warning("%s profile load busy, freeing RealSense video nodes and retrying", self)
                    self._reset_busy_device(device)
                    continue
                raise
        else:
            raise RuntimeError(f"{self} profile load failed after dropping unsupported keys.")
        logger.info("Loaded RealSense profile from %s", profile_path)

    def _snapshot_pair_locked(self) -> NDArray[Any] | None:
        """Atomically grab the current (color, depth) pair under frame_lock.

        Returns the color frame and stashes the matching depth frame in
        _last_depth_snapshot. The parent class's _read_loop updates color
        and depth together under the same lock, so pairs snapshotted here
        come from one read-loop iteration — no drift between the channels
        that get persisted as the "same frame" of the dataset.
        """
        with self.frame_lock:
            color = self.latest_color_frame
            depth = self.latest_depth_frame if self.use_depth else None
        if depth is not None:
            depth_copy = depth.copy()
            with self._depth_snapshot_lock:
                self._last_depth_snapshot = depth_copy
        return color

    def pop_depth_snapshot(self) -> NDArray[Any] | None:
        """Return and clear the depth snapshot stashed by the last async_read()."""
        with self._depth_snapshot_lock:
            snap = self._last_depth_snapshot
            self._last_depth_snapshot = None
        return snap

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if (
            self.ready
            and self.latest_color_frame is not None
            and time.monotonic() - self.latest_frame_time <= timeout_ms / 1000.0
        ):
            frame = self._snapshot_pair_locked()
            if frame is not None:
                self.last_frame = frame
                return frame

        timeout_s = timeout_ms / 1000.0
        if self.new_frame_event.wait(timeout=timeout_s):
            frame = self._snapshot_pair_locked()
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
