#!/usr/bin/env bash
set -e
CONFIG="$(dirname "$0")/configs/arms.yaml"

left_port=$(yq -r '.leader.left_arm.port' "$CONFIG")
right_port=$(yq -r '.leader.right_arm.port' "$CONFIG")
cameras=$(yq -c '.cameras.devices' "$CONFIG")

exec uv run lerobot-teleoperate \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$left_port" \
    --teleop.right_arm_port="$right_port" \
    --display_data=true \
    "$@"

# --robot.cameras="$cameras" \