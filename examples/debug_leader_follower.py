import argparse
import logging
import time
import signal
import sys
from pathlib import Path

import yaml

from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_camera_zed.zed_camera import ZEDCamera, ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

from utils import _free_port

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
ARMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Bimanual leader-follower teleoperation")
    parser.add_argument(
        "--left-leader-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the left leader arm (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--right-leader-port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for the right leader arm (default: /dev/ttyACM1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(ARMS_CONFIG_PATH, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]

    # Free from old subprocesses
    _free_port(left_follower_server_port)
    _free_port(right_follower_server_port)

    available_zed_cameras = ZEDCamera.find_cameras()
    if not available_zed_cameras:
        print("No ZED cameras found.")

    # get first camera for now - generalise later
    zed_cam_id = available_zed_cameras[0]["id"]

    bi_follower_config = BiYamsFollowerConfig(
        left_arm_server_port=left_follower_server_port,
        right_arm_server_port=right_follower_server_port,
        cameras={
            "topdown": ZEDCameraConfig(
                camera_id=zed_cam_id,
                width=640,
                height=480,
                fps=30,
            ),
            "left_wrist": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
        },
    )

    bi_leader_config = BiYamsLeaderConfig(
        left_arm_port=args.left_leader_port,
        right_arm_port=args.right_leader_port,
    )

    bi_leader = BiYamsLeader(bi_leader_config)
    bi_leader.connect()

    bi_follower = BiYamsFollower(bi_follower_config)
    bi_follower.connect()

    cleaned_up = False

    def cleanup():
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        print("Cleaning up arm connections")
        bi_follower.disconnect()
        bi_leader.disconnect()

    def handle_sigint(signum, frame):
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        return
    finally:
        cleanup()

    # freq = 200  # Hz

    # bi_leader_action = bi_leader.get_action()

    # slow_move(bi_follower.left_arm, split_arm_action(bi_leader_action, "left_"))
    # slow_move(bi_follower.right_arm, split_arm_action(bi_leader_action, "right_"))

    # start_time = time.time()
    # count = 0
    # try:
    #     while True:
    #         count += 1
    #         bi_leader_action = bi_leader.get_action()
    #         if bi_leader_action is None:
    #             continue
    #         bi_follower.send_action(bi_leader_action)
    #         time.sleep(1 / freq)
    #         time_elapsed = time.time() - start_time
    #         if count % 400 == 0:
    #             print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
    #         if time_elapsed >= 0.05:
    #             print(f"Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds")
    #         start_time = time.time()

    # except KeyboardInterrupt:
    #     print("\nStopping teleop...")
    # finally:
    #     for arm in [bi_follower.left_arm, bi_follower.right_arm]:
    #         slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
    #     bi_leader.disconnect()
    #     bi_follower.disconnect()


if __name__ == "__main__":
    main()
