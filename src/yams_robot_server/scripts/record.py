from pynput import keyboard
import time
import parquet as pq
from lerobot.utils.control_utils import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH, write_info

from lerobot.cameras import ColorMode, Cv2Rotation
from yams_robot_server.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from yams_robot_server.bi_leader import BiYamsLeader, BiYamsLeaderConfig
from yams_robot_server.camera import ZEDCamera, ZEDCameraConfig

repo_id = "ETHRC/yams"
root = "../../../data"

available_zed_cameras = ZEDCamera.find_cameras()
if not available_zed_cameras:
    print("No ZED cameras found.")

# get first camera for now - generalise later
zed_cam_id = available_zed_cameras[0]["id"]

bi_follower_config = BiYamsFollowerConfig(
    left_arm_port="can_follower_l",
    right_arm_port="can_follower_r",
    cameras={
        "topdown": ZEDCameraConfig(
            camera_id=zed_cam_id,
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            color_mode=ColorMode.RGB,
        ),
    },
)

bi_leader_config = BiYamsLeaderConfig(
    left_arm_port="/dev/ttyACM0",
    right_arm_port="/dev/ttyACM1",
)


class RecordingController:
    def __init__(self):
        self.recording = False
        self.discard = False
        self.exit = False
        self.episode_count = 0
        self.frame_count = 0
        self.start_time = None

    def on_press(self, key):
        try:
            if key == keyboard.Key.right and not self.recording:
                self.recording = True
                self.discard = False
                self.frame_count = 0
                self.start_time = time.monotonic()
                print(
                    "\n[INFO] Press ↓ (Down Arrow) to save and ← (Left Arrow) to discard"
                )

            elif key == keyboard.Key.left and self.recording:
                self.recording = False
                self.discard = True

            elif key == keyboard.Key.down and self.recording:
                self.recording = False
                self.discard = False

            elif key == keyboard.Key.up:
                print("\n[INFO] Exiting...")
                self.exit = True

        except AttributeError:
            pass

    def convert_image_dataset_to_video(dataset: LeRobotDataset):
        """Convert a dataset recorded with image frames into the canonical video layout."""
        if dataset.num_episodes == 0:
            return
        import shutil

        image_keys = list(dataset.meta.image_keys)
        if not image_keys:
            return

        for key in image_keys:
            dataset.meta.info["features"][key]["dtype"] = "video"
        dataset.meta.info["video_path"] = DEFAULT_VIDEO_PATH
        write_info(dataset.meta.info, dataset.meta.root)
        dataset.meta.load_metadata()

        for episode_index in range(dataset.num_episodes):
            dataset.encode_episode_videos(episode_index)

        for episode_index in range(dataset.num_episodes):
            parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_index)

            table = pq.read_table(parquet_path)

            columns_to_keep = [
                col
                for col in table.schema.names
                if not col.startswith("observation.images.")
            ]
            table_without_images = table.select(columns_to_keep)

            pq.write_table(table_without_images, parquet_path)

        dataset.meta.info["total_videos"] = dataset.num_episodes * len(
            dataset.meta.video_keys
        )
        write_info(dataset.meta.info, dataset.meta.root)
        dataset.meta.load_metadata()

        # Step 4: Clean up images directory
        images_dir = dataset.root / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)


def main():
    follower = BiYamsFollower(config=bi_follower_config)
    leader = BiYamsLeader(config=bi_leader_config)

    follower.connect()
    leader.connect()

    # fix: create dataset if not there

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=30,
        use_videos=False,
        image_writer_processes=4,
        image_writer_threads=8,
    )

    print("\n[INFO] Recording Control:")
    print("  → (Right Arrow)    - Start a new episode")
    print("  ← (Left Arrow)     - Discard the current episode")
    print("  ↓ (Down Arrow)     - Save the current episode")
    print("  ↑ (Up Arrow)       - Quit and finalize the recording")

    controller = RecordingController()

    listener = keyboard.Listener(on_press=controller.on_press)
    listener.start()

    was_recording = False
    try:

        while not controller.exit:
            if was_recording and not controller.recording:
                try:
                    if controller.discard:
                        print(
                            f"[INFO] Discarding episode {controller.episode_count + 1}..."
                        )
                        dataset.clear_episode_buffer()
                    else:
                        print(
                            f"[INFO] Saving episode {controller.episode_count + 1} ({controller.frame_count} frames)..."
                        )
                        dataset.save_episode()
                        controller.episode_count += 1

                except Exception as e:
                    print(f"[ERROR] Failed to save episode: {e}")

            elif controller.recording:
                try:
                    action = leader.get_action()
                    follower.send_action(action)
                    observation = follower.get_observation()
                    dataset.add_frame({"observation": observation, "action": action})
                    frame_count += 1
                    time_elapsed = time.time() - controller.start_time
                    if time_elapsed >= 0.1:
                        print(
                            f"[INFO] Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds"
                        )

                except Exception as e:
                    print(f"[ERROR] Failed to capture frame: {e}")

            was_recording = controller.recording

    except KeyboardInterrupt:
        print(f"\n[INFO] Keyboard interrupt. Saving episode")
        if controller.recording:
            dataset.save_episode()
    except Exception as e:
        print(f"[ERROR] Failed to record: {e}")
    finally:
        print(f"[INFO] Shutting down...")
        listener.stop()

        dataset.stop_image_writer()

        controller.convert_image_dataset_to_video(dataset)

        dataset.finalize()
        dataset.push_to_hub()

        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
