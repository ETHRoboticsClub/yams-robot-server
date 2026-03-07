YAML=configs/arms.yaml
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

sh third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

# uv run lerobot-teleoperate \
#     --robot.type=bi_yams_follower \
#     --teleop.type=bi_yams_leader \
#     --robot.cameras="$cameras" \ 
#     --teleop.left_arm_port="$LEFT_PORT" \
#     --teleop.right_arm_port="$RIGHT_PORT" \
#     --display_data=false \
#     --fps=250 \

uv run lerobot-record \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --robot.cameras="$cameras" \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --display_data=true \
    --fps=250 \
    --dataset.repo_id=ETHRC/fake_dataset \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=2 \
    --dataset.single_task="Fold the towel."

    # --dataset.streaming_encoding=true \
    # --dataset.encoder_threads=2
