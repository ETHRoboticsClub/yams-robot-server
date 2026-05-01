#!/usr/bin/env bash
# r=start recording  space=save  d=discard  esc=exit
set -euo pipefail

YAML=configs/arms.yaml
REPO=${REPO:-ETHRC/closed-carton-box-to-migros-box-go2}
TASK=${TASK:-Pick & Place and Closing a Box}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
DATASET_TAGS=${DATASET_TAGS:-yams,bimanual}
DISPLAY_DATA=${DISPLAY_DATA:-false}
DATASET_FPS=${DATASET_FPS:-15}
VCODEC=${VCODEC:-auto}

LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

pgrep -f "lerobot-record|lerobot-teleoperate|yams_server.py" | grep -vx "$$" | xargs -r kill || true
PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_ROOT="$SCRIPT_DIR/../data/$REPO"

if [ -f "$DATASET_ROOT/meta/tasks.parquet" ]; then
    RESUME=true
else
    rm -rf "$DATASET_ROOT"
    RESUME=false
fi

set +o pipefail
PYTHONPATH=src uv run python src/utils/patched_record.py \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --display_data="$DISPLAY_DATA" \
    --play_sounds=true \
    --dataset.fps="$DATASET_FPS" \
    --dataset.num_episodes=10000 \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s=86400 \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --dataset.tags="[$DATASET_TAGS]" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true \
    2>&1 | grep -vE "^\[Server\]|^\[Client\]|Corrupt JPEG|Invalid SOS|premature end"
RC=${PIPESTATUS[0]}
set -o pipefail

echo "=== Session ended ==="
