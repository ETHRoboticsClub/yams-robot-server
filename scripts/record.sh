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

YAML=configs/arms.yaml
REPO=ETHRC/act
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT')"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

# if [ -d "$HOME/.cache/huggingface/lerobot/$REPO" ] && [ ! -f "$HOME/.cache/huggingface/lerobot/$REPO/meta/info.json" ]; then
#     mv "$HOME/.cache/huggingface/lerobot/$REPO" "$HOME/.cache/huggingface/lerobot/$REPO.stale.$(date +%s)"
# fi
rm -rf /home/ethrc/.cache/huggingface/lerobot/ETHRC/act

uv run lerobot-record \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --display_data=false \
    --dataset.fps=60 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Fold the towel." \
    --dataset.repo_id="$REPO" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/$REPO" \
    --dataset.push_to_hub=false \
    --resume=false \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --dataset.encoder_threads=2


    # --dataset.streaming_encoding=true \
    # --dataset.encoder_threads=2
