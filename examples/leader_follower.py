import time

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move
from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig


def main():
    follower_config = YamsFollowerConfig(
        can_port="can_follower_r",
        server_port=11333,
    )

    leader_config = YamsLeaderConfig(port="/dev/ttyACM0", side="right")

    leader = YamsLeader(leader_config)
    leader.connect()

    follower = YamsFollower(follower_config)
    follower.connect()

    freq = 100  # Hz

    leader_action = leader.get_action()
    slow_move(follower, leader_action)

    try:
        while True:
            leader_action = leader.get_action()
            if leader_action is None:
                continue
            # print({key: f"{value:.2f}" for key, value in leader_action.items()})
            follower.send_action(leader_action)
            time.sleep(1 / freq)
    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        slow_move(
            follower, {f"{name}.pos": 0.0 for name in follower.config.joint_names}
        )
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
