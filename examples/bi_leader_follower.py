import time
import cv2

from lerobot.cameras import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

from yams_robot_server.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from yams_robot_server.bi_leader import BiYamsLeader, BiYamsLeaderConfig
from yams_robot_server.camera import ZEDCameraConfig, ZEDCamera, find_opencv_cameras
from yams_robot_server.utils.utils import slow_move, split_arm_action

available_zed_cameras = ZEDCamera.find_cameras()
if not available_zed_cameras:
    print("No ZED cameras found.")

# get first camera for now - generalise later
zed_cam_id = available_zed_cameras[0]["id"]

bi_follower_config = BiYamsFollowerConfig(
    left_arm_port="can1",
    right_arm_port="can0",
    cameras={
        "topdown": ZEDCameraConfig(
            camera_id=zed_cam_id,
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            color_mode=ColorMode.RGB,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=4,
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
    left_arm_port="/dev/ttyACM0",
    right_arm_port="/dev/ttyACM1",
)

bi_leader = BiYamsLeader(bi_leader_config)
bi_leader.connect()

bi_follower = BiYamsFollower(bi_follower_config)
bi_follower.connect()

freq = 100  # Hz

bi_leader_action = bi_leader.get_action()

slow_move(bi_follower.left_arm, split_arm_action(bi_leader_action, "left_"))
slow_move(bi_follower.right_arm, split_arm_action(bi_leader_action, "right_"))

try:
    while True:
        bi_leader_action = bi_leader.get_action()
        print({key: f"{value:.2f}" for key, value in bi_leader_action.items()})
        bi_follower.send_action(bi_leader_action)
        observation = bi_follower.get_observation()
        zed_camera_image = observation["topdown"]
        print(
            f"Camera image shape: {zed_camera_image.shape}, dtype: {zed_camera_image.dtype}"
        )
        cv2.imshow("ZED Camera", zed_camera_image)
        cv2.waitKey(1)

        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    for arm in [bi_follower.left_arm, bi_follower.right_arm]:
        slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
    bi_leader.disconnect()
    bi_follower.disconnect()
