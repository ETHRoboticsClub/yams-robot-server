from pathlib import Path

import yaml
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Rotation

from lerobot_camera_cached.cached_config import OpenCVCameraCachedConfig
from lerobot_camera_cached.realsense_cached_config import RealSenseCameraCachedConfig
from lerobot_camera_zed.zed_config import ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

from utils.lifecycle import run_pre_setup
from utils.live_joint_plot import LiveJointPlotter
from utils.camera_memo import resolve_camera_configs
from utils.teleop_data import build_joint_label_map


def setup_arms_cameras_plotter(args, arms_config_path: Path, logger):
    with open(arms_config_path, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    leader_config = arms_config["leader"]
    follower_joint_label_map = build_joint_label_map(follower_config)
    leader_joint_label_map = build_joint_label_map(leader_config)
    camera_label_map: dict[str, str] = arms_config.get("cameras", {}).get("labels", {})
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]
    left_follower_can_port = follower_config["left_arm"]["can_port"]
    right_follower_can_port = follower_config["right_arm"]["can_port"]
    left_leader_port = leader_config["left_arm"]["port"]
    right_leader_port = leader_config["right_arm"]["port"]
    run_pre_setup(left_follower_server_port, right_follower_server_port, usb_ports=[left_leader_port, right_leader_port])

    cameras = {}
    had_camera_config = False
    if args.skip_cams:
        logger.info("Skipping camera setup (--skip-cams enabled)")
    else:
        raw_camera_configs = arms_config.get("cameras", {}).get("configs", {})
        if not raw_camera_configs:
            logger.warning("No cameras configured in mapping yaml; continuing without cameras.")
        else:
            had_camera_config = True
            try:
                resolved_camera_configs = resolve_camera_configs(raw_camera_configs, logger)

                zed_cam_id = None
                for name, cfg in resolved_camera_configs.items():
                    cfg = dict(cfg)
                    camera_type = cfg.pop("type", "zed")
                    if camera_type == "opencv":
                        cameras[name] = OpenCVCameraConfig(**cfg)
                    elif camera_type == "opencv-cached":
                        cameras[name] = OpenCVCameraCachedConfig(**cfg)
                    elif camera_type == "intelrealsense-cached":
                        cameras[name] = RealSenseCameraCachedConfig(**cfg)
                    else:
                        if zed_cam_id is None:
                            from lerobot_camera_zed.zed_camera import ZEDCamera

                            available_zed_cameras = ZEDCamera.find_cameras()
                            logger.info("Detected ZED cameras: %s", available_zed_cameras)
                            if not available_zed_cameras:
                                raise RuntimeError("Zed camera not found")
                            zed_cam_id = available_zed_cameras[0]["id"]
                            logger.info("Using ZED camera id=%s", zed_cam_id)
                        cfg["rotation"] = Cv2Rotation[cfg["rotation"]]
                        cameras[name] = ZEDCameraConfig(camera_id=zed_cam_id, **cfg)
            except Exception as exc:
                logger.warning("Camera setup failed (%s). Continuing without cameras.", exc)
                cameras = {}
                camera_label_map = {}

    bi_follower = BiYamsFollower(
        BiYamsFollowerConfig(
            left_arm_can_port=left_follower_can_port,
            left_arm_server_port=left_follower_server_port,
            right_arm_can_port=right_follower_can_port,
            right_arm_server_port=right_follower_server_port,
            cameras=cameras,
        )
    )
    bi_leader = BiYamsLeader(
        BiYamsLeaderConfig(left_arm_port=left_leader_port, right_arm_port=right_leader_port)
    )
    bi_leader.connect()
    try:
        bi_follower.connect()
    except Exception as exc:
        if had_camera_config and not args.skip_cams:
            logger.warning("Failed to connect cameras (%s). Retrying without cameras.", exc)
            try:
                bi_follower.disconnect()
            except Exception:
                pass
            bi_follower = BiYamsFollower(
                BiYamsFollowerConfig(
                    left_arm_can_port=left_follower_can_port,
                    left_arm_server_port=left_follower_server_port,
                    right_arm_can_port=right_follower_can_port,
                    right_arm_server_port=right_follower_server_port,
                    cameras={},
                )
            )
            try:
                bi_follower.connect()
                camera_label_map = {}
            except Exception:
                raise
        else:
            raise

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
