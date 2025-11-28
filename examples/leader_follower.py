import time

import numpy as np

from yams_robot_server.follower import YamsFollower, YamsFollowerConfig
from yams_robot_server.leader import YamsLeader, YamsLeaderConfig
from yams_robot_server.utils.utils import slow_move

follower_config = YamsFollowerConfig(
    port="can0",
)

leader_config = YamsLeaderConfig(port="/dev/ttyACM1", side="left")

leader = YamsLeader(leader_config)
leader.connect()

follower = YamsFollower(follower_config)
follower.connect()

freq = 50  # Hz

leader_action = leader.get_action()
slow_move(follower, leader_action)

try:
    while True:
        leader_action = leader.get_action()
        print({key: f"{value:.2f}" for key, value in leader_action.items()})
        follower.send_action(leader_action)
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    slow_move(follower, {f"{name}.pos": 0.0 for name in follower.config.joint_names})
    leader.disconnect()
    follower.disconnect()
