#!/usr/bin/env bash
set -euo pipefail

YAML=${YAML:-configs/arms.yaml}
REPO=${REPO:?REPO env var required (e.g. ETHRC/my-dataset)}
EPISODE=${EPISODE:-0}

LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")

bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

set -x
PYTHONPATH=src uv run lerobot-replay \
    --robot.type=bi_yams_follower \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --dataset.repo_id="$REPO" \
    --dataset.episode="$EPISODE"
set +x
