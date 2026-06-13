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

pgrep -f "lerobot-record|lerobot-teleoperate|yams_server.py|run_record.py" | grep -vx "$$" | xargs -r kill

YAML=configs/arms.yaml
# DATA_DIR is resolved by the cosmos dataloading config (${oc.env:DATA_DIR});
# only the training dataloader uses it, but config composition resolves it at
# worker startup, so it must be set. Exported so mimic_adapter forwards it to
# the cosmos subprocess (and so it survives `sudo`).
# export DATA_DIR=${DATA_DIR:-/home/ethrc/.cache/mimic_lerobot_t5}
# Pin the T5 embedding cache to an absolute path so the cosmos worker finds it
# regardless of HOME (e.g. under `sudo`, where HOME=/root). Exported so
# mimic_adapter forwards it to the cosmos subprocess.
export MIMIC_T5_CACHE_DIR=${MIMIC_T5_CACHE_DIR:-/home/ethrc/.cache/mimic-yams/t5_embeddings}
RESUME=${RESUME:-false}
RECORD=${RECORD:-false}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
NEW_REPO=${NEW_REPO:-false}
MIN_CAMERA_FPS=$(yq '[.cameras.configs[].fps] | min' "$YAML")
DATASET_FPS=${DATASET_FPS:-$MIN_CAMERA_FPS}
NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-0}
VCODEC=${VCODEC:-auto}
STOP_VIDEO_DENOISING_STEP=${STOP_VIDEO_DENOISING_STEP:?must be set explicitly (mimic-video uses partial denoising; operator picks the value)}
# Debug: if FUTURE_VIDEO_DEBUG_DIR is set, cosmos dumps its predicted future
# video (full denoising + VAE decode) as MP4 per inference batch. Roughly
# doubles per-batch inference time on dump ticks — throttle with
# FUTURE_VIDEO_DUMP_EVERY_N (default 1 = every batch). Set DUMP_VIDEO=true
# to auto-pick a timestamped dir under logs/future_video/.
DUMP_VIDEO=${DUMP_VIDEO:-false}
FUTURE_VIDEO_DEBUG_DIR=${FUTURE_VIDEO_DEBUG_DIR:-}
FUTURE_VIDEO_DUMP_EVERY_N=${FUTURE_VIDEO_DUMP_EVERY_N:-1}
if [ "$DUMP_VIDEO" = "true" ] && [ -z "$FUTURE_VIDEO_DEBUG_DIR" ]; then
    FUTURE_VIDEO_DEBUG_DIR="logs/future_video/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$FUTURE_VIDEO_DEBUG_DIR"
    echo "DUMP_VIDEO=true: predicted-future MP4s will land in $FUTURE_VIDEO_DEBUG_DIR"
fi
ACTION_STRIDE=${ACTION_STRIDE:-1}
# Set WITH_TELEOP=true to also connect the leader arms (e.g., for emergency
# takeover or to sanity-check the leader chain). Off by default — the
# mimic_video policy drives the follower directly and the leader's actions
# are discarded.
WITH_TELEOP=${WITH_TELEOP:-false}

# =============================================================================
# TASK: CARTON BOX CLOSING (mimic-video / Cosmos VAM)
# Dataset: ETHRC/yams-carton-box-closing-mon-tom-mat  |  EPISODE_TIME_S=240  |  RESET_TIME_S=10
# Checkpoints loaded by MimicVideoConfig defaults from ./checkpoints/.
# =============================================================================
REPO=${REPO:-ETHRC/eval_carton_box_test}
TASK=${TASK:-push the box to the right with the right arm}
EPISODE_TIME_S=${EPISODE_TIME_S:-240}
RESET_TIME_S=${RESET_TIME_S:-10}

if [ "$WITH_TELEOP" = "true" ]; then
    LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
    RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
fi
LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq '.cameras.configs | pick(["topdown"])' "$YAML")
CAMERA_PATHS=$(yq -r '.cameras.configs[] | select(has("index_or_path")) | .index_or_path' "$YAML")
INTERRUPTED=false
_TMPDIR=""

if [ "$RECORD" = "false" ]; then
    PUSH_TO_HUB=false
    _TMPDIR=$(mktemp -d)
    DATASET_ROOT=${DATASET_ROOT:-"$_TMPDIR/eval"}
    REPO=${REPO:-"local/eval"}
    echo "RECORD=false: dataset will not be saved (using $_TMPDIR)"
else
    if [ "$NEW_REPO" = "true" ]; then
        RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
        REPO="${REPO}_$RUN_ID"
        echo "NEW_REPO=true: writing this eval to $REPO"
    fi
    DATASET_BASE_DIR=${DATASET_BASE_DIR:-"$HOME/.cache/huggingface/lerobot"}
    DATASET_ROOT=${DATASET_ROOT:-"$DATASET_BASE_DIR/$REPO"}
fi

cleanup_zero() {
    echo "Signal received: moving follower arms to zero"
    pgrep -f "lerobot-record|lerobot-teleoperate|run_record.py" | grep -vx "$$" | xargs -r kill
    PYTHONPATH=src uv run python -m utils.move_arms_zero
    [ -n "$_TMPDIR" ] && rm -rf "$_TMPDIR"
}

trap 'INTERRUPTED=true; [ -n "${LEROBOT_PID:-}" ] && kill -INT "$LEROBOT_PID" 2>/dev/null || true' INT TERM

if [ "$WITH_TELEOP" = "true" ]; then
    PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
else
    PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
fi
bash third_party/i2rt/scripts/reset_all_can.sh
if [ "$WITH_TELEOP" = "true" ]; then
    echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
    echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
fi

for camera in $CAMERA_PATHS; do
    dev="$(readlink -f "$camera")"
    current_ae=$(v4l2-ctl -d "$dev" --get-ctrl=auto_exposure 2>/dev/null | awk -F: '{print $2}' | tr -d ' ')
    if [ "$current_ae" != "1" ]; then
        ./scripts/set_camera_profile.sh "$dev"
    else
        echo "Camera profile already applied to $dev, skipping"
    fi
done

if [ "$RECORD" != "false" ] && [ "$RESUME" != "true" ] && [ -d "$DATASET_ROOT" ]; then
    read -r -p "ATTENTION: You set resume to false. DELETE YOUR ENTIRE DATASET at $DATASET_ROOT?? [y/N] " confirm
    [ "$confirm" = "y" ] || [ "$confirm" = "Y" ] || exit 1
    rm -rf "$DATASET_ROOT"
fi

export PYNPUT_BACKEND_KEYBOARD=uinput
export PYNPUT_BACKEND_MOUSE=dummy
TELEOP_ARGS=()
if [ "$WITH_TELEOP" = "true" ]; then
    TELEOP_ARGS=(
        --teleop.type=bi_yams_leader
        --teleop.left_arm_port="$LEFT_PORT"
        --teleop.right_arm_port="$RIGHT_PORT"
    )
fi
uv run python run_record.py \
    --robot.type=bi_yams_follower \
    "${TELEOP_ARGS[@]}" \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --display_data=false \
    --dataset.fps="$DATASET_FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true \
    --policy.type=mimic_video \
    --policy.task_prompt="$TASK" \
    --policy.stop_video_denoising_step="$STOP_VIDEO_DENOISING_STEP" \
    --policy.action_stride="$ACTION_STRIDE" \
    --policy.future_video_debug_dir="$FUTURE_VIDEO_DEBUG_DIR" \
    --policy.future_video_dump_every_n="$FUTURE_VIDEO_DUMP_EVERY_N" \
    --play_sounds=false &
LEROBOT_PID=$!
wait "$LEROBOT_PID"
status=$?
trap - INT TERM

if $INTERRUPTED || [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then
    trap '' INT TERM
    cleanup_zero
else
    [ -n "$_TMPDIR" ] && rm -rf "$_TMPDIR"
fi

exit "$status"
