import unittest
from unittest.mock import patch

import numpy as np

from utils.camera_auto_exposure import CameraAutoExposure, frame_brightness, next_exposure


class TestCameraAutoExposure(unittest.TestCase):
    def test_frame_brightness_uses_center_crop(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[2:6, 2:6] = 100
        self.assertEqual(frame_brightness(frame), 100.0)

    def test_next_exposure_increases_when_too_dark(self):
        self.assertEqual(next_exposure(50, 50, 100, 5, 0.5, 5, 200), 62)

    def test_next_exposure_decreases_when_too_bright(self):
        self.assertEqual(next_exposure(50, 150, 100, 5, 0.5, 5, 200), 38)

    def test_next_exposure_respects_deadband(self):
        self.assertEqual(next_exposure(50, 97, 100, 5, 0.5, 5, 200), 50)

    @patch("utils.camera_auto_exposure._set_exposure")
    @patch("utils.camera_auto_exposure.time.monotonic", side_effect=[10.0, 10.1, 10.7])
    def test_tick_rate_limits_updates(self, _, set_exposure):
        controller = CameraAutoExposure(
            device="/dev/video0",
            exposure=50,
            target=100,
            deadband=5,
            speed=0.5,
            min_exposure=5,
            max_exposure=200,
            period_s=0.5,
        )
        frame = np.full((8, 8, 3), 50, dtype=np.uint8)

        self.assertEqual(controller.tick(frame), 62)
        self.assertIsNone(controller.tick(frame))
        self.assertEqual(controller.tick(frame), 78)
        self.assertEqual(set_exposure.call_count, 2)


if __name__ == "__main__":
    unittest.main()
