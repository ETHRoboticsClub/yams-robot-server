import logging
import time

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move
from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

# Create a handler that prints to console
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# # Add handler to the specific package's logger
# logger = logging.getLogger("can")
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)


def main():
    follower_config = YamsFollowerConfig(
        can_port="can0",
        server_port=11333,
    )

    leader_config = YamsLeaderConfig(port="/dev/ttyACM1", side="left")

    leader = YamsLeader(leader_config)
    leader.connect()

    follower = YamsFollower(follower_config)
    follower.connect()

    freq = 100  # Hz

    leader_action = leader.get_action()
    slow_move(follower, leader_action)

    try:
        start_time = time.time()
        count = 0
        while True:
            leader_action = leader.get_action()
            if leader_action is None:
                continue
            # print({key: f"{value:.2f}" for key, value in leader_action.items()})
            follower.send_action(leader_action)
            time.sleep(1 / freq)
            time_elapsed = time.time() - start_time
            if count % 400 == 0:
                print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
            if time_elapsed >= 0.05:
                print(f"Max elapsed time larger then 50ms: {time_elapsed:.2f} seconds")
            start_time = time.time()
            count += 1
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
