import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_stubs() -> None:
    if "lerobot" not in sys.modules:
        sys.modules["lerobot"] = types.ModuleType("lerobot")

    if "lerobot.cameras" not in sys.modules:
        sys.modules["lerobot.cameras"] = types.ModuleType("lerobot.cameras")

    if "lerobot.cameras.opencv" not in sys.modules:
        sys.modules["lerobot.cameras.opencv"] = types.ModuleType("lerobot.cameras.opencv")

    if "lerobot.cameras.opencv.configuration_opencv" not in sys.modules:
        opencv_cfg_mod = types.ModuleType("lerobot.cameras.opencv.configuration_opencv")

        class OpenCVCameraConfig:
            pass

        opencv_cfg_mod.OpenCVCameraConfig = OpenCVCameraConfig
        sys.modules["lerobot.cameras.opencv.configuration_opencv"] = opencv_cfg_mod

    if "lerobot.cameras.realsense.camera_realsense" not in sys.modules:
        cam_mod = types.ModuleType("lerobot.cameras.realsense.camera_realsense")

        class RealSenseCamera:
            def __init__(self, config):
                self.config = config
                self.serial_number = config.serial_number_or_name

        cam_mod.RealSenseCamera = RealSenseCamera
        sys.modules["lerobot.cameras.realsense.camera_realsense"] = cam_mod

    if "lerobot.cameras.realsense.configuration_realsense" not in sys.modules:
        cfg_mod = types.ModuleType("lerobot.cameras.realsense.configuration_realsense")

        class RealSenseCameraConfig:
            pass

        cfg_mod.RealSenseCameraConfig = RealSenseCameraConfig
        sys.modules["lerobot.cameras.realsense.configuration_realsense"] = cfg_mod

    if "lerobot.cameras.configs" not in sys.modules:
        cfgs_mod = types.ModuleType("lerobot.cameras.configs")

        class _Registry:
            @classmethod
            def register_subclass(cls, _name):
                def deco(subcls):
                    return subcls

                return deco

        cfgs_mod.CameraConfig = _Registry
        sys.modules["lerobot.cameras.configs"] = cfgs_mod

    if "pyrealsense2" not in sys.modules:
        rs = types.ModuleType("pyrealsense2")
        rs.camera_info = types.SimpleNamespace(serial_number="serial_number")
        rs.context = lambda: None
        rs.rs400_advanced_mode = lambda device: None
        sys.modules["pyrealsense2"] = rs


_install_stubs()

from lerobot_camera_cached.camera_realsense_cached import RealSenseCameraCached


class _Device:
    def __init__(self, serial):
        self.serial = serial
        self.reset_calls = 0

    def get_info(self, _key):
        return self.serial

    def hardware_reset(self):
        self.reset_calls += 1


class _Advanced:
    def __init__(self, enabled=False, errors=None):
        self.enabled = enabled
        self.loaded = None
        self.toggled = []
        self.errors = list(errors or [])

    def is_enabled(self):
        return self.enabled

    def toggle_advanced_mode(self, value):
        self.toggled.append(value)
        self.enabled = value

    def load_json(self, text):
        if self.errors:
            raise self.errors.pop(0)
        self.loaded = text


class TestRealSenseProfileLoad(unittest.TestCase):
    def test_wait_for_device_handles_temporary_disconnect(self):
        import lerobot_camera_cached.camera_realsense_cached as mod

        device = _Device("abc")
        calls = iter([[], [], [device]])
        mod.rs = types.SimpleNamespace(
            camera_info=types.SimpleNamespace(serial_number="serial_number"),
            context=lambda: types.SimpleNamespace(query_devices=lambda: next(calls, [device])),
            rs400_advanced_mode=lambda _device: None,
        )

        cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
        cam.serial_number = "abc"

        self.assertIs(cam._wait_for_device(timeout_s=1.0, poll_s=0.0), device)

    def test_load_profile_reads_json_and_applies_it(self):
        import lerobot_camera_cached.camera_realsense_cached as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realsense.json"
            path.write_text('{"x":1}')

            advanced = _Advanced(enabled=True)
            mod.rs = types.SimpleNamespace(
                camera_info=types.SimpleNamespace(serial_number="serial_number"),
                context=lambda: types.SimpleNamespace(query_devices=lambda: [_Device("abc")]),
                rs400_advanced_mode=lambda device: advanced,
            )

            cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
            cam.serial_number = "abc"
            cam.config = types.SimpleNamespace(profile_path=path)
            cam._load_profile()

            self.assertEqual(advanced.loaded, '{"x":1}')

    def test_busy_profile_load_frees_realsense_nodes_and_retries(self):
        import lerobot_camera_cached.camera_realsense_cached as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realsense.json"
            path.write_text('{"x":1}')

            device = _Device("abc")
            advanced = _Advanced(errors=[RuntimeError("Device or resource busy")])
            freed = []
            mod.rs = types.SimpleNamespace(
                camera_info=types.SimpleNamespace(serial_number="serial_number"),
                context=lambda: types.SimpleNamespace(query_devices=lambda: [device]),
                rs400_advanced_mode=lambda device: advanced,
            )
            mod._free_v4l_devices = freed.append

            cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
            cam.serial_number = "abc"
            cam.config = types.SimpleNamespace(profile_path=path)
            cam._load_profile()

            self.assertEqual(freed, ["RealSense"])
            self.assertEqual(device.reset_calls, 1)
            self.assertEqual(advanced.loaded, '{"x":1}')

    def test_load_profile_skips_unsupported_keys(self):
        import lerobot_camera_cached.camera_realsense_cached as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realsense.json"
            path.write_text('{"parameters":{"bad-key":"1","good-key":"2"}}')

            advanced = _Advanced(errors=[RuntimeError("bad-key key is not supported by the connected device!")])
            mod.rs = types.SimpleNamespace(
                camera_info=types.SimpleNamespace(serial_number="serial_number"),
                context=lambda: types.SimpleNamespace(query_devices=lambda: [_Device("abc")]),
                rs400_advanced_mode=lambda device: advanced,
            )

            cam = RealSenseCameraCached.__new__(RealSenseCameraCached)
            cam.serial_number = "abc"
            cam.config = types.SimpleNamespace(profile_path=path)
            cam._load_profile()

            self.assertIn('"good-key": "2"', advanced.loaded)
            self.assertNotIn("bad-key", advanced.loaded)


if __name__ == "__main__":
    unittest.main()
