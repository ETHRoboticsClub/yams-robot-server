import time

import numpy as np

from yams_robot_server.follower import YamsFollower, YamsFollowerConfig
from yams_robot_server.leader import YamsLeader, YamsLeaderConfig
from yams_robot_server.utils.utils import slow_move

follower_config_left = YamsFollowerConfig(
    port="can0",
)

follower_config_right = YamsFollowerConfig(
    port="can1",
)
leader_config_left = YamsLeaderConfig(port="/dev/ttyACM1", side="left")
leader_config_right = YamsLeaderConfig(port="/dev/ttyACM2", side="right")

leader_left = YamsLeader(leader_config_left)
leader_right = YamsLeader(leader_config_right)
leader_left.connect()
leader_right.connect()

follower_left = YamsFollower(follower_config_left)
follower_right = YamsFollower(follower_config_right)
follower_left.connect()
follower_right.connect()

freq = 200  # Hz

leader_action_left = leader_left.get_action()
leader_action_right = leader_right.get_action()
slow_move(follower_left, leader_action_left)
slow_move(follower_right, leader_action_right)

try:
    while True:
        leader_action_left = leader_left.get_action()
        leader_action_right = leader_right.get_action()
        print({key: f"{value:.2f}" for key, value in leader_action_left.items()})
        print({key: f"{value:.2f}" for key, value in leader_action_right.items()})
        follower_left.send_action(leader_action_left)
        follower_right.send_action(leader_action_right)
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    slow_move(
        follower_left, {f"{name}.pos": 0.0 for name in follower_left.config.joint_names}
    )
    slow_move(
        follower_right,
        {f"{name}.pos": 0.0 for name in follower_right.config.joint_names},
    )
    leader_left.disconnect()
    leader_right.disconnect()
    follower_left.disconnect()
    follower_right.disconnect()
