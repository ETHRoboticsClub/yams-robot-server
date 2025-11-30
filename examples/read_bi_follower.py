import time

from yams_robot_server.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from yams_robot_server.utils.utils import slow_move


def main():
    bi_follower_config = BiYamsFollowerConfig()

    bi_follower = BiYamsFollower(bi_follower_config)
    bi_follower.connect()

    freq = 200  # Hz

    for arm in [bi_follower.left_arm, bi_follower.right_arm]:
        slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})

    zero_action = {
        f"left_{name}.pos": 0.0 for name in bi_follower.left_arm.config.joint_names
    } | {f"right_{name}.pos": 0.0 for name in bi_follower.right_arm.config.joint_names}

    start_time = time.time()
    count = 0
    try:
        while True:
            count += 1

            obs = bi_follower.get_observation()
            bi_follower.send_action(zero_action)
            time.sleep(1 / freq)
            time_elapsed = time.time() - start_time
            if count % 400 == 0:
                print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
            if time_elapsed >= 0.1:
                print(f"Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds")
            start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        for arm in [bi_follower.left_arm, bi_follower.right_arm]:
            slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
        bi_follower.disconnect()


if __name__ == "__main__":
    main()
