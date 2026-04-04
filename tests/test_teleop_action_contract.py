import sys
import types
import unittest
from unittest.mock import Mock


def _install_lerobot_stubs() -> None:
    if "lerobot" in sys.modules:
        return

    lerobot = types.ModuleType("lerobot")
    motors = types.ModuleType("lerobot.motors")
    dynamixel = types.ModuleType("lerobot.motors.dynamixel")
    teleoperators = types.ModuleType("lerobot.teleoperators")
    teleoperator = types.ModuleType("lerobot.teleoperators.teleoperator")
    utils = types.ModuleType("lerobot.utils")
    errors = types.ModuleType("lerobot.utils.errors")

    class Motor:
        def __init__(self, *args, **kwargs):
            pass

    class _MotorNormMode(dict):
        def __getitem__(self, key):
            return key

    class DynamixelMotorsBus:
        model_ctrl_table = {"xm430-w350": {}}
        model_baudrate_table = {"xm430-w350": {}}
        model_encoding_table = {"xm430-w350": {}}
        model_resolution_table = {"xm430-w350": {}}
        model_number_table = {"xm430-w350": 1}

        def __init__(self, *args, **kwargs):
            self.is_connected = False
            self.motors = {}

    class OperatingMode:
        CURRENT_POSITION = types.SimpleNamespace(value=0)

    class Teleoperator:
        def __init__(self, config):
            self.config = config

    class TeleoperatorConfig:
        @classmethod
        def register_subclass(cls, _name):
            def decorator(subcls):
                return subcls

            return decorator

    class DeviceAlreadyConnectedError(Exception):
        pass

    class DeviceNotConnectedError(Exception):
        pass

    motors.Motor = Motor
    motors.MotorNormMode = _MotorNormMode()
    dynamixel.DynamixelMotorsBus = DynamixelMotorsBus
    dynamixel.OperatingMode = OperatingMode
    teleoperator.Teleoperator = Teleoperator
    teleoperator.TeleoperatorConfig = TeleoperatorConfig
    errors.DeviceAlreadyConnectedError = DeviceAlreadyConnectedError
    errors.DeviceNotConnectedError = DeviceNotConnectedError

    sys.modules["lerobot"] = lerobot
    sys.modules["lerobot.motors"] = motors
    sys.modules["lerobot.motors.dynamixel"] = dynamixel
    sys.modules["lerobot.teleoperators"] = teleoperators
    sys.modules["lerobot.teleoperators.teleoperator"] = teleoperator
    sys.modules["lerobot.utils"] = utils
    sys.modules["lerobot.utils.errors"] = errors


_install_lerobot_stubs()

from lerobot_teleoperator_gello.bi_leader import BiYamsLeader
from lerobot_teleoperator_gello.leader import YamsLeader


class TestLeaderActionContract(unittest.TestCase):
    def test_yams_leader_raises_on_sync_read_failure(self):
        leader = YamsLeader.__new__(YamsLeader)
        leader.bus = Mock(is_connected=True)
        leader.bus.sync_read.side_effect = Exception("bad packet")
        leader.calibration = {"offsets": {}, "scales": {}}

        with self.assertRaisesRegex(RuntimeError, "Failed to read leader action"):
            leader.get_action()

    def test_bi_leader_raises_if_arm_returns_none(self):
        leader = BiYamsLeader.__new__(BiYamsLeader)
        leader._pool = Mock()

        left_future = Mock()
        left_future.result.return_value = None
        right_future = Mock()
        right_future.result.return_value = {"joint.pos": 1.0}
        leader._pool.submit.side_effect = [left_future, right_future]
        leader.left_arm = Mock(get_action=Mock())
        leader.right_arm = Mock(get_action=Mock())

        with self.assertRaisesRegex(RuntimeError, "Leader returned no action"):
            leader.get_action()

    def test_bi_leader_merges_actions(self):
        leader = BiYamsLeader.__new__(BiYamsLeader)
        leader._pool = Mock()

        left_future = Mock()
        left_future.result.return_value = {"joint.pos": 1.0}
        right_future = Mock()
        right_future.result.return_value = {"joint.pos": 2.0}
        leader._pool.submit.side_effect = [left_future, right_future]
        leader.left_arm = Mock(get_action=Mock())
        leader.right_arm = Mock(get_action=Mock())

        action = leader.get_action()

        self.assertEqual(action, {"left_joint.pos": 1.0, "right_joint.pos": 2.0})


if __name__ == "__main__":
    unittest.main()
