from pathlib import Path

import yaml
from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_camera_zed.zed_camera import ZEDCamera, ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig
from lerobot.cameras.configs import Cv2Rotation

from utils.lifecycle import run_pre_setup
from utils.live_joint_plot import LiveJointPlotter
from utils.teleop_data import build_joint_label_map


def setup_arms_cameras_plotter(args, arms_config_path: Path, logger):
    with open(arms_config_path, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    leader_config = arms_config["leader"]
    follower_joint_label_map = build_joint_label_map(follower_config)
    leader_joint_label_map = build_joint_label_map(leader_config)
    camera_label_map = arms_config.get("cameras", {}).get("labels", {})
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]
    left_leader_port = leader_config["left_arm"]["port"]
    right_leader_port = leader_config["right_arm"]["port"]
    run_pre_setup(left_follower_server_port, right_follower_server_port, usb_ports=[left_leader_port, right_leader_port])

    cameras = {}
    if args.skip_cams:
        logger.info("Skipping camera setup (--skip-cams enabled)")
    else:
        available_zed_cameras = ZEDCamera.find_cameras()
        logger.info("Detected ZED cameras: %s", available_zed_cameras)
        if not available_zed_cameras:
            raise Exception("Zed camera not found (use --skip-cams to continue without cameras)")
        zed_cam_id = available_zed_cameras[0]["id"]
        logger.info("Using ZED camera id=%s", zed_cam_id)
        cameras = {
            "left_wrist": OpenCVCameraConfig(index_or_path=4, fps=30, width=640, height=480, fourcc="MJPG"),
            "right_wrist": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480, fourcc="MJPG"),
            "topdown": ZEDCameraConfig(camera_id=zed_cam_id, width=640, height=480, fps=30, rotation=Cv2Rotation.NO_ROTATION),
        }

    bi_follower = BiYamsFollower(
        BiYamsFollowerConfig(
            left_arm_server_port=left_follower_server_port,
            right_arm_server_port=right_follower_server_port,
            cameras=cameras,
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
