YAML=configs/arms.yaml
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.devices' "$YAML")

uv run lerobot-teleoperate \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --display_data=false \
    --fps=250
    # --robot.cameras="$cameras" \
# uv run lerobot-teleoperate \
#     --robot.type=bi_yams_follower \
#     --teleop.type=bi_yams_leader \
#     --teleop.left_arm_port=/dev/ttyUSB1 \
#     --teleop.right_arm_port=/dev/ttyUSB0 \
#     --display_data=false \
#     --fps=250

# uv run lerobot-teleoperate     --robot.type=bi_yams_follower   --teleop.type=bi_yams_leader     --teleop.left_arm_port=/dev/ttyUSB1     --teleop.right_arm_port=/dev/ttyUSB0     --display_data=false --fps=120
