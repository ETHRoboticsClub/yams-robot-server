import argparse
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_camera_zed.zed_camera import ZEDCamera, ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

from utils.connection import _free_port
from utils.live_joint_plot import start_joint_plotter

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
ARMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"


def _build_joint_label_map(section_config: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    joint_labels = section_config.get("joint_labels", {})
    for side in ("left", "right"):
        for joint, label in joint_labels.get(side, {}).items():
            out[f"{side}_{joint}.pos"] = label
    return out


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

cleaned_up = False
bi_leader = None
bi_follower = None
plotter = None

def cleanup():
    global cleaned_up, plotter
    if cleaned_up:
        return
    cleaned_up = True
    print("Cleaning up arm connections")
    if bi_follower is not None:
        bi_follower.disconnect()
    if bi_leader is not None:
        bi_leader.disconnect()
    if plotter is not None:
        plotter.close()

def handle_sigint(signum, frame):
    cleanup()
    raise SystemExit(0)

# def monitor_arm_obs(bi_follower, bi_leader):
#     global plotter
#     obs = bi_follower.get_observation(with_cameras=True)
#     act = bi_leader.get_action()
#     plotter.push(obs, act)
    # def fmt(value):
    #     arr = np.asarray(value)
    #     if arr.ndim == 0:
    #         return f"{float(arr):.2f}"
    #     return f"array(shape={arr.shape}, dtype={arr.dtype})"

    # arm_obs = {
    #     key: fmt(value)
    #     for key, value in obs.items()
    #     if key.startswith(("left_", "right_"))
    # }
    # print(arm_obs)

HZ = 200

def main():
    global bi_leader, bi_follower, plotter
    subprocess.run(["sh", str(Path(__file__).resolve().parents[1] / "third_party/i2rt/scripts/reset_all_can.sh")], check=True)

    args = parse_args()
    with open(ARMS_CONFIG_PATH, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    follower_joint_label_map = _build_joint_label_map(follower_config)
    leader_joint_label_map = _build_joint_label_map(arms_config.get("leader", {}))
    camera_label_map = arms_config.get("cameras", {}).get("labels", {})
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]

    # Free from old subprocesses
    _free_port(left_follower_server_port)
    _free_port(right_follower_server_port)

    available_zed_cameras = ZEDCamera.find_cameras()
    if available_zed_cameras:
        zed_cam_id = available_zed_cameras[0]["id"]
    else:
        print("No ZED cameras found.")

    # get first camera for now - generalise later
    cameras = {
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
    }
    
    if zed_cam_id:
        cameras["topdown"] = ZEDCameraConfig(
            camera_id=zed_cam_id,
            width=640,
            height=480,
            fps=30,
        )

    bi_follower_config = BiYamsFollowerConfig(
        left_arm_server_port=left_follower_server_port,
        right_arm_server_port=right_follower_server_port,
        cameras=cameras
    )

    bi_leader_config = BiYamsLeaderConfig(
        left_arm_port=args.left_leader_port,
        right_arm_port=args.right_leader_port,
    )

    try:
        bi_leader = BiYamsLeader(bi_leader_config)
        bi_leader.connect()

        bi_follower = BiYamsFollower(bi_follower_config)
        bi_follower.connect()

        signal.signal(signal.SIGINT, handle_sigint)
        

        plotter = start_joint_plotter(
            bi_follower,
            hz=60,
            history_s=10,
            backend="web",
            web_port=8988,
            camera_hz=5,
            follower_joint_label_map=follower_joint_label_map,
            leader_joint_label_map=leader_joint_label_map,
            camera_label_map=camera_label_map,
        )

        while True:
            # monitor_arm_obs(bi_follower, bi_leader)
            obs = bi_follower.get_observation(with_cameras=True)
            act = bi_leader.get_action()
            plotter.push(obs, act)

            time.sleep(1 / HZ)
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
