import unittest

from utils.lerobot_record_wrapper import (
    _is_leader_action_failure,
    run_with_graceful_stop,
)


class TestLerobotRecordWrapper(unittest.TestCase):
    def test_detects_direct_leader_failure(self):
        exc = RuntimeError("Failed to read leader action from leader")
        self.assertTrue(_is_leader_action_failure(exc))

    def test_detects_nested_leader_failure(self):
        inner = RuntimeError("Failed to read leader action from leader")
        outer = RuntimeError("record loop failed")
        outer.__cause__ = inner
        self.assertTrue(_is_leader_action_failure(outer))

    def test_ignores_unrelated_runtime_error(self):
        exc = RuntimeError("camera blew up")
        self.assertFalse(_is_leader_action_failure(exc))

    def test_returns_zero_for_leader_failure(self):
        def fail():
            raise RuntimeError("Failed to read leader action from leader")

        self.assertEqual(run_with_graceful_stop(fail), 0)

    def test_reraises_unrelated_runtime_error(self):
        def fail():
            raise RuntimeError("camera blew up")

        with self.assertRaisesRegex(RuntimeError, "camera blew up"):
            run_with_graceful_stop(fail)


if __name__ == "__main__":
    unittest.main()
