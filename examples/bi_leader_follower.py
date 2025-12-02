import gc
import logging
import time

from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_camera_zed.zed_camera import ZEDCamera, ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

gc.disable()

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    available_zed_cameras = ZEDCamera.find_cameras()
    if not available_zed_cameras:
        print("No ZED cameras found.")

    # get first camera for now - generalise later
    zed_cam_id = available_zed_cameras[0]["id"]

    bi_follower_config = BiYamsFollowerConfig(
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
        left_arm_port="/dev/ttyACM1",
        right_arm_port="/dev/ttyACM0",
    )

    bi_leader = BiYamsLeader(bi_leader_config)
    bi_leader.connect()

    bi_follower = BiYamsFollower(bi_follower_config)
    bi_follower.connect()

    freq = 200  # Hz

    bi_leader_action = bi_leader.get_action()

    slow_move(bi_follower.left_arm, split_arm_action(bi_leader_action, "left_"))
    slow_move(bi_follower.right_arm, split_arm_action(bi_leader_action, "right_"))

    start_time = time.time()
    count = 0
    try:
        while True:
            count += 1
            bi_leader_action = bi_leader.get_action()
            if bi_leader_action is None:
                continue
            bi_follower.send_action(bi_leader_action)
            time.sleep(1 / freq)
            time_elapsed = time.time() - start_time
            if count % 400 == 0:
                print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
            if time_elapsed >= 0.05:
                print(f"Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds")
            start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        for arm in [bi_follower.left_arm, bi_follower.right_arm]:
            slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
        bi_leader.disconnect()
        bi_follower.disconnect()


if __name__ == "__main__":
    main()
