[ -z "$BASH" ] && exec bash "$0" "$@"
echo "working directory $PWD"

LOG=false
for arg in "$@"; do [ "$arg" = "--log" ] && LOG=true; done
if $LOG; then
    mkdir -p logs
    LOGFILE="logs/$(date +%Y%m%d_%H%M%S).out"
    echo "Logging to $LOGFILE"
    exec > >(tee "$LOGFILE") 2>&1
fi

pgrep -f "lerobot-record|lerobot-teleoperate|yams_server.py" | grep -vx "$$" | xargs -r kill

YAML=configs/arms.yaml
REPO=ETHRC/towelspring26_TESTTSTSTSTS
RESUME=${RESUME:-false}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
MIN_CAMERA_FPS=$(yq '[.cameras.configs[].fps] | min' "$YAML")
DATASET_FPS=${DATASET_FPS:-$MIN_CAMERA_FPS}
NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-0}
TASK=${TASK:-Fold the towel.}
VCODEC=${VCODEC:-auto}
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

# if [ -d "$HOME/.cache/huggingface/lerobot/$REPO" ] && [ ! -f "$HOME/.cache/huggingface/lerobot/$REPO/meta/info.json" ]; then
#     mv "$HOME/.cache/huggingface/lerobot/$REPO" "$HOME/.cache/huggingface/lerobot/$REPO.stale.$(date +%s)"
# fi
# rm -rf /home/ethrc/.cache/huggingface/lerobot/$REPO

if [ "$RESUME" != "true" ] && [ -d "$HOME/.cache/huggingface/lerobot/$REPO" ]; then
    rm -rf "$HOME/.cache/huggingface/lerobot/$REPO"
fi

PYTHONPATH=src uv run python -m utils.lerobot_record_wrapper \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --display_data=false \
    --dataset.fps="$DATASET_FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/$REPO" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true
    # --dataset.push_to_hub=true \
    # --dataset.encoder_queue_maxsize=1000
    # --dataset.encoder_threads=2
    # --dataset.vcodec=libx264 \
