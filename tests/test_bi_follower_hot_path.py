import sys
import time
import types
import unittest


def _install_lerobot_stubs() -> None:
    if "lerobot" in sys.modules:
        return

    lerobot = types.ModuleType("lerobot")
    cameras = types.ModuleType("lerobot.cameras")
    cameras_utils = types.ModuleType("lerobot.cameras.utils")
    robots = types.ModuleType("lerobot.robots")

    class CameraConfig:
        pass

    def make_cameras_from_configs(configs):
        return configs

    class Robot:
        def __init__(self, config):
            self.config = config

    class RobotConfig:
        @classmethod
        def register_subclass(cls, _name):
            def decorator(subcls):
                return subcls

            return decorator

    cameras.CameraConfig = CameraConfig
    cameras_utils.make_cameras_from_configs = make_cameras_from_configs
    robots.Robot = Robot
    robots.RobotConfig = RobotConfig

    sys.modules["lerobot"] = lerobot
    sys.modules["lerobot.cameras"] = cameras
    sys.modules["lerobot.cameras.utils"] = cameras_utils
    sys.modules["lerobot.robots"] = robots


def _install_yams_follower_stub() -> None:
    module_name = "lerobot_robot_yams.follower"
    if module_name in sys.modules:
        return

    follower = types.ModuleType(module_name)

    class YamsFollowerConfig:
        def __init__(self, can_port, server_port, side):
            self.can_port = can_port
            self.server_port = server_port
            self.side = side
            self.joint_names = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper",
            ]

    class YamsFollower:
        def __init__(self, config):
            self.config = config

    follower.YamsFollower = YamsFollower
    follower.YamsFollowerConfig = YamsFollowerConfig
    sys.modules[module_name] = follower


def _install_check_action_stub() -> None:
    module_name = "lerobot_robot_yams.forward_kinematics"
    if module_name in sys.modules:
        return

    fk = types.ModuleType(module_name)

    def check_action(*args, **kwargs):
        return False, ""

    fk.check_action = check_action
    sys.modules[module_name] = fk


_install_lerobot_stubs()
_install_yams_follower_stub()
_install_check_action_stub()

from lerobot_robot_yams.bi_follower import BiYamsFollower


class _FakeArm:
    def __init__(self, name, delay):
        self.name = name
        self.delay = delay
        self.config = types.SimpleNamespace(
            joint_names=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper",
            ]
        )

    def get_observation(self):
        time.sleep(self.delay)
        return {"joint_1.pos": self.delay}

    def disconnect(self):
        return None


class _FakeCamera:
    def __init__(self, name, delay):
        self.name = name
        self.delay = delay
        self.is_connected = True

    def async_read(self):
        time.sleep(self.delay)
        return self.name

    def disconnect(self):
        return None


class TestBiFollowerHotPath(unittest.TestCase):
    def test_arm_reads_run_in_parallel(self):
        follower = BiYamsFollower.__new__(BiYamsFollower)
        from concurrent.futures import ThreadPoolExecutor

        follower._obs_pool = ThreadPoolExecutor(max_workers=4)
        follower.left_arm = _FakeArm("left", 0.12)
        follower.right_arm = _FakeArm("right", 0.12)
        follower.cameras = {}

        start = time.perf_counter()
        obs = follower.get_observation(with_cameras=False)
        elapsed = time.perf_counter() - start
        follower._obs_pool.shutdown(wait=True)

        self.assertLess(elapsed, 0.2)
        self.assertEqual(obs["left_joint_1.pos"], 0.12)
        self.assertEqual(obs["right_joint_1.pos"], 0.12)

    def test_camera_reads_run_in_parallel(self):
        follower = BiYamsFollower.__new__(BiYamsFollower)
        from concurrent.futures import ThreadPoolExecutor

        follower._obs_pool = ThreadPoolExecutor(max_workers=6)
        follower.left_arm = _FakeArm("left", 0.01)
        follower.right_arm = _FakeArm("right", 0.01)
        follower.cameras = {
            "cam_a": _FakeCamera("A", 0.12),
            "cam_b": _FakeCamera("B", 0.12),
            "cam_c": _FakeCamera("C", 0.12),
        }

        start = time.perf_counter()
        obs = follower.get_observation(with_cameras=True)
        elapsed = time.perf_counter() - start
        follower._obs_pool.shutdown(wait=True)

        self.assertLess(elapsed, 0.22)
        self.assertEqual(obs["cam_a"], "A")
        self.assertEqual(obs["cam_b"], "B")
        self.assertEqual(obs["cam_c"], "C")


if __name__ == "__main__":
    unittest.main()
