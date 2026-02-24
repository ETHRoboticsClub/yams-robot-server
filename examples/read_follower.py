import time
import argparse

from lerobot_robot_yams.follower import YamsFollower, YamsFollowerConfig


def main(side):
    if side == "right":
        can_port = "can_follower_r"
        server_port = 11334

    elif side == "left":
        can_port = "can_follower_l"
        server_port = 11333

    follower_config = YamsFollowerConfig(
        can_port=can_port,
        server_port=server_port,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("side", help="Arm to read from (left/right)", choices=["left", "right"])
    args = parser.parse_args()
    print(f"Reading from {args.side} arm.")
    
    main(args.side)
