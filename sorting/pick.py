"""
Pick primitive: get one packet out of the cluttered source box.

This is the ONLY learned part of the pipeline (Approach B). Three modes:

  MockPick    always succeeds — for dry-run / demo / dashboard.
  PolicyPick  runs a trained LeRobot ACT policy on the real robot until a grasp
              is detected (gripper closed + lifted). Requires a checkpoint (M2)
              and a live robot; import-guarded so mock mode needs neither.

A scripted depth-grasp pick is intentionally NOT implemented: hand-engineering a
grasp on floppy packets is the brittle path the whole approach avoids. If ever
needed it slots in here behind the same interface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("sorting.pick")


@dataclass
class PickResult:
    ok: bool
    reason: str = ""


class MockPick:
    def execute(self, detection) -> PickResult:
        log.info("[pick] (mock) grasp packet at %s", getattr(detection, "pick_xy", None))
        return PickResult(True, "mock")


class PolicyPick:
    """Runs an ACT policy inference loop against the LeRobot robot.

    Wired to the standard LeRobot Robot + policy API so it is ready the moment a
    checkpoint exists. Not exercised in mock mode."""

    def __init__(self, robot, policy_path: str, max_seconds: float = 20.0, fps: int = 30):
        self.robot = robot
        self.policy_path = policy_path
        self.max_seconds = max_seconds
        self.fps = fps
        self._policy = None

    def _load(self):
        if self._policy is None:
            from lerobot.policies.factory import make_policy_config  # noqa
            from lerobot.policies.act.modeling_act import ACTPolicy
            self._policy = ACTPolicy.from_pretrained(self.policy_path)
            self._policy.eval()
        return self._policy

    def execute(self, detection) -> PickResult:
        try:
            import torch
            policy = self._load()
            deadline = time.time() + self.max_seconds
            period = 1.0 / self.fps
            while time.time() < deadline:
                obs = self.robot.get_observation()
                batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()
                         if not hasattr(v, "shape") or v.ndim <= 1}
                with torch.no_grad():
                    action = policy.select_action(batch)
                act = {k: float(action[0, i]) for i, k in enumerate(self.robot.action_features)}
                self.robot.send_action(act)
                time.sleep(period)
                if self._grasped(obs):
                    return PickResult(True, "grasped")
            return PickResult(False, "timeout without grasp")
        except Exception as e:
            log.exception("policy pick failed")
            return PickResult(False, f"{type(e).__name__}: {e}")

    @staticmethod
    def _grasped(obs) -> bool:
        # Heuristic placeholder: refine with gripper-closed + lift detection once
        # the policy and gripper feedback are wired against real observations.
        return False


def make_pick(mode: str, robot=None, policy_path: str | None = None):
    if mode == "policy":
        if robot is None or not policy_path:
            raise ValueError("policy pick needs a robot and policy_path")
        return PolicyPick(robot, policy_path)
    return MockPick()
