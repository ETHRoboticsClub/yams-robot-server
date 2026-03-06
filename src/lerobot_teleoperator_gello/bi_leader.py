#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("bi_yams_leader")
@dataclass
class BiYamsLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670


class BiYamsLeader(Teleoperator):
    config_class = BiYamsLeaderConfig
    name = "bi_yams_leader"

    def __init__(self, config: BiYamsLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = YamsLeaderConfig(
            port=self.config.left_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
            side="left",
        )
        right_arm_config = YamsLeaderConfig(
            port=self.config.right_arm_port,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_closed_pos=self.config.gripper_closed_pos,
            side="right",
        )

        self.left_arm = YamsLeader(left_arm_config)
        self.right_arm = YamsLeader(right_arm_config)
        self._pool = ThreadPoolExecutor(max_workers=2)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }  # type: ignore

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = False) -> None:
        self.left_arm.connect()
        self.right_arm.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        left_f = self._pool.submit(self.left_arm.get_action)
        right_f = self._pool.submit(self.right_arm.get_action)
        left_action = left_f.result()
        right_action = right_f.result()
        if left_action is None or right_action is None:
            return None
        return {
            **{f"left_{k}": v for k, v in left_action.items()},
            **{f"right_{k}": v for k, v in right_action.items()},
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(self.left_arm.disconnect)
            ex.submit(self.right_arm.disconnect)
        self._pool.shutdown(wait=True)
