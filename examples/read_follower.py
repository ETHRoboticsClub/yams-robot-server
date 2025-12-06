import time

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig


def main():
    follower_config = YamsFollowerConfig(
        can_port="can_follower_r",
        server_port=11334,
    )
    follower = YamsFollower(follower_config)

    follower.connect()

    hz = 100  # Hz

    try:
        while True:
            print({key: f"{value:.2f}" for key, value in follower.get_observation().items()})
            time.sleep(1 / hz)

    except KeyboardInterrupt:
        print("\nStopping read position...")
        follower.disconnect()


if __name__ == "__main__":
    main()
