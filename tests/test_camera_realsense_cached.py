import sys
import threading
import time
import types
import unittest

import numpy as np


def _install_camera_stubs() -> None:
    if "lerobot" not in sys.modules:
        sys.modules["lerobot"] = types.ModuleType("lerobot")

    if "lerobot.cameras.realsense.camera_realsense" not in sys.modules:
        cam_mod = types.ModuleType("lerobot.cameras.realsense.camera_realsense")

        class RealSenseCamera:
            def __init__(self, config):
                self.config = config
                self.thread = None
                self.latest_color_frame = None
                self.new_frame_event = threading.Event()

        cam_mod.RealSenseCamera = RealSenseCamera
        sys.modules["lerobot.cameras.realsense.camera_realsense"] = cam_mod

    if "lerobot_camera_cached.realsense_cached_config" not in sys.modules:
        cfg_mod = types.ModuleType("lerobot_camera_cached.realsense_cached_config")

        class RealSenseCameraCachedConfig:
            def __init__(self, width=2, height=2):
                self.width = width
                self.height = height

        cfg_mod.RealSenseCameraCachedConfig = RealSenseCameraCachedConfig
        sys.modules["lerobot_camera_cached.realsense_cached_config"] = cfg_mod


_install_camera_stubs()

from lerobot_camera_cached.camera_realsense_cached import RealSenseCameraCached


class _AliveThread:
    def is_alive(self):
        return True


class TestRealSenseCameraCached(unittest.TestCase):
    def test_returns_cached_frame_immediately(self):
        cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
        cam.thread = _AliveThread()
        cam.new_frame_event = threading.Event()
        cam.latest_color_frame = np.ones((2, 2, 3), dtype=np.uint8)
        cam.ready = True
        cam.latest_frame_time = time.monotonic()
        cam.last_frame = np.zeros((2, 2, 3), dtype=np.uint8)

        start = time.perf_counter()
        frame = cam.async_read(timeout_ms=200)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05)
        self.assertTrue(np.array_equal(frame, cam.latest_color_frame))

    def test_wait_is_bounded_to_single_timeout(self):
        cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
        cam.thread = _AliveThread()
        cam.new_frame_event = threading.Event()
        cam.latest_color_frame = None
        cam.ready = False
        cam.latest_frame_time = 0.0
        cam.last_frame = np.zeros((2, 2, 3), dtype=np.uint8)

        start = time.perf_counter()
        with self.assertRaises(TimeoutError):
            cam.async_read(timeout_ms=50)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.12)

    def test_stale_cached_frame_does_not_bypass_wait(self):
        cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
        cam.thread = _AliveThread()
        cam.new_frame_event = threading.Event()
        cam.latest_color_frame = np.ones((2, 2, 3), dtype=np.uint8)
        cam.ready = True
        cam.latest_frame_time = time.monotonic() - 1.0
        cam.last_frame = np.zeros((2, 2, 3), dtype=np.uint8)

        start = time.perf_counter()
        with self.assertRaises(TimeoutError):
            cam.async_read(timeout_ms=50)
        elapsed = time.perf_counter() - start

        self.assertGreater(elapsed, 0.04)


if __name__ == "__main__":
    unittest.main()
