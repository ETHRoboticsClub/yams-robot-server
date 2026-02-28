from pathlib import Path

import cv2
import yaml
from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

from utils.lifecycle import run_pre_setup
from utils.live_joint_plot import LiveJointPlotter
from utils.teleop_data import build_joint_label_map


def can_read_camera(index_or_path) -> bool:
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        return False
    try:
        ok, _ = cap.read()
        return ok
    finally:
        cap.release()


def setup_arms_cameras_plotter(args, arms_config_path: Path, logger):
    with open(arms_config_path, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    leader_config = arms_config["leader"]
    follower_joint_label_map = build_joint_label_map(follower_config)
    leader_joint_label_map = build_joint_label_map(leader_config)
    cameras_config = arms_config.get("cameras", {})
    camera_label_map = cameras_config.get("labels", {})
    camera_devices = cameras_config.get("devices", {})
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]
    left_leader_port = leader_config["left_arm"]["port"]
    right_leader_port = leader_config["right_arm"]["port"]
    run_pre_setup(left_follower_server_port, right_follower_server_port)

    configured_cameras = {}
    for name, cam in camera_devices.items():
        path = cam["path"]
        width = int(cam.get("width", 640))
        height = int(cam.get("height", 480))
        fps = int(cam.get("fps", 30))
        fourcc = cam.get("fourcc")
        if not can_read_camera(path):
            if args.allow_no_cams:
                logger.warning("%s camera (%s) not readable, skipping", name, path)
                continue
            raise Exception(f"{name} camera ({path}) not readable (use --allow-no-cams to continue)")
        fourcc_msg = f", fourcc={fourcc}" if fourcc else ""
        logger.info("Using %s camera %s at %sx%s@%s%s", name, path, width, height, fps, fourcc_msg)
        configured_cameras[name] = OpenCVCameraConfig(
            index_or_path=path,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )

    if not configured_cameras and not args.allow_no_cams:
        raise Exception("No cameras found (use --allow-no-cams to continue)")

    bi_follower = BiYamsFollower(
        BiYamsFollowerConfig(
            left_arm_server_port=left_follower_server_port,
            right_arm_server_port=right_follower_server_port,
            cameras=configured_cameras,
        )
    )
    bi_leader = BiYamsLeader(
        BiYamsLeaderConfig(left_arm_port=left_leader_port, right_arm_port=right_leader_port)
    )
    bi_leader.connect()
    bi_follower.connect()

    obs = bi_follower.get_observation(with_cameras=False)
    joint_keys = sorted(k for k in obs if k.endswith(".pos") and k.startswith(("left_", "right_")))
    if not joint_keys:
        raise ValueError("No joint position keys found in follower observation.")
    plotter = LiveJointPlotter(
        joint_keys,
        hz=60,
        history_s=10,
        backend="web",
        web_port=8988,
        camera_hz=5,
        follower_joint_label_map=follower_joint_label_map,
        leader_joint_label_map=leader_joint_label_map,
        camera_label_map=camera_label_map,
    )
    return bi_leader, bi_follower, plotter
