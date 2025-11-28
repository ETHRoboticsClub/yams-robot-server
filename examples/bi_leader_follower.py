import time

from yams_robot_server.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from yams_robot_server.bi_leader import BiYamsLeader, BiYamsLeaderConfig
from yams_robot_server.utils.utils import slow_move, split_arm_action

bi_follower_config = BiYamsFollowerConfig(
    left_arm_port="can0",
    right_arm_port="can1",
)

bi_leader_config = BiYamsLeaderConfig(
    left_arm_port="/dev/ttyACM1",
    right_arm_port="/dev/ttyACM2",
)

bi_leader = BiYamsLeader(bi_leader_config)
bi_leader.connect()

bi_follower = BiYamsFollower(bi_follower_config)
bi_follower.connect()

freq = 100  # Hz

bi_leader_action = bi_leader.get_action()

slow_move(bi_follower.left_arm, split_arm_action(bi_leader_action, "left_"))
slow_move(bi_follower.right_arm, split_arm_action(bi_leader_action, "right_"))

try:
    while True:
        bi_leader_action = bi_leader.get_action()
        print({key: f"{value:.2f}" for key, value in bi_leader_action.items()})
        bi_follower.send_action(bi_leader_action)
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    for arm in [bi_follower.left_arm, bi_follower.right_arm]:
        slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
    bi_leader.disconnect()
    bi_follower.disconnect()
