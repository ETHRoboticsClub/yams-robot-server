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
RESUME=${RESUME:-false}
PUSH_TO_HUB=${PUSH_TO_HUB:-true}
NEW_REPO=${NEW_REPO:-false}
MIN_CAMERA_FPS=$(yq '[.cameras.configs[].fps] | min' "$YAML")
DATASET_FPS=${DATASET_FPS:-$MIN_CAMERA_FPS}
NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-0}
VCODEC=${VCODEC:-auto}

# =============================================================================
# TASK: TOWEL FOLDING
# Dataset: ETHRC/towelspring26_2
# Recommended: EPISODE_TIME_S=20  RESET_TIME_S=10
# To activate: uncomment the REPO, TASK and one POLICY_PATH below,
#              comment out the CARTON BOX section.
# =============================================================================
# REPO=${REPO:-ETHRC/eval_towelspring26_test}
# TASK=${TASK:-Fold the towel.}
#
# -- Baraq / previous runs --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2/checkpoints/last} # WORKS WELL at night with light ON, NOT in daylight
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/realsense_1/checkpoints/last} # trained on towelspring26_3-trimmed
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/realsense_1_notrim/checkpoints/last}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/run1/pretrained_model}
#
# -- April 17/18 augmentation batch (towelspring26_2) --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_noise_20260417_224504_74152/checkpoints/last}   # WORKS WELL - daylight tested
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_shadow_20260417_224504_74152/checkpoints/last}  # WORKS WELL - daylight tested
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_blur_20260417_224504_74152/checkpoints/last}    # NOT WORKING
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_no_aug_20260417_224504_74152/checkpoints/last}       # NOT WORKING
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_augmented_20260417_224504_74152/checkpoints/last}    # NOT WORKING

# =============================================================================
# TASK: CARTON BOX CLOSING
# Dataset: ETHRC/yams-carton-box-closing-mon-tom-mat  |  EPISODE_TIME_S=120  |  RESET_TIME_S=10
# To activate: uncomment the REPO, TASK, EPISODE_TIME_S, RESET_TIME_S and one POLICY_PATH below,
#              comment out the TOWEL FOLDING section.
# =============================================================================
REPO=${REPO:-ETHRC/eval_carton_box_test}
TASK=${TASK:-Pick & Place and Closing a Box}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-10}
#
# -- April 20 carton box runs --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_no_aug/checkpoints/last}  # NOT WORKING - doesn't pick up object, lifts too low
POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_no_aug_full_20260420_221222_217313/checkpoints/last} # Doesn't work (but seems the best so far)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_noise_strong_full_20260420_221222_217313/checkpoints/last} # Doesn't work - way to jerky (worse than the two previous ones)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_noise_full_20260420_221222_217313/checkpoints/last} # Doesn't work - same as last one
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_shadow_full_20260420_221222_217313/checkpoints/last} # Doesn't work

LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")
CAMERA_PATHS=$(yq -r '.cameras.configs[] | select(has("index_or_path")) | .index_or_path' "$YAML")
INTERRUPTED=false

if [ "$NEW_REPO" = "true" ]; then
    RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
    REPO="${REPO}_$RUN_ID"
    echo "NEW_REPO=true: writing this eval to $REPO"
fi
DATASET_BASE_DIR=${DATASET_BASE_DIR:-"$HOME/.cache/huggingface/lerobot"}
DATASET_ROOT=${DATASET_ROOT:-"$DATASET_BASE_DIR/$REPO"}

cleanup_zero() {
    echo "Signal received: moving follower arms to zero"
    pgrep -f "lerobot-record|lerobot-teleoperate" | grep -vx "$$" | xargs -r kill
    PYTHONPATH=src uv run python -m utils.move_arms_zero
}

trap 'INTERRUPTED=true; [ -n "${LEROBOT_PID:-}" ] && kill -INT "$LEROBOT_PID" 2>/dev/null || true' INT TERM

[ -d "$POLICY_PATH/pretrained_model" ] && POLICY_PATH="$POLICY_PATH/pretrained_model"

PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

for camera in $CAMERA_PATHS; do
    dev="$(readlink -f "$camera")"
    current_ae=$(v4l2-ctl -d "$dev" --get-ctrl=auto_exposure 2>/dev/null | awk -F: '{print $2}' | tr -d ' ')
    if [ "$current_ae" != "1" ]; then
        ./scripts/set_camera_profile.sh "$dev"
    else
        echo "Camera profile already applied to $dev, skipping"
    fi
done

if [ "$RESUME" != "true" ] && [ -d "$DATASET_ROOT" ]; then
    read -r -p "ATTENTION: You set resume to false. DELETE YOUR ENTIRE DATASET at $DATASET_ROOT?? [y/N] " confirm
    [ "$confirm" = "y" ] || [ "$confirm" = "Y" ] || exit 1
    rm -rf "$DATASET_ROOT"
fi

export PYNPUT_BACKEND_KEYBOARD=uinput
export PYNPUT_BACKEND_MOUSE=dummy
uv run lerobot-record \
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
    --dataset.root="$DATASET_ROOT" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true \
    --policy.path="$POLICY_PATH" \
    --play_sounds=false &
LEROBOT_PID=$!
wait "$LEROBOT_PID"
status=$?
trap - INT TERM

if $INTERRUPTED || [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then
    trap '' INT TERM
    cleanup_zero
fi

exit "$status"
