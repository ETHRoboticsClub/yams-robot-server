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
if pgrep -f "realsense-viewer" >/dev/null; then
    echo "Close realsense-viewer before recording; it holds the RealSense device."
    exit 1
fi

YAML=configs/arms.yaml
REPO=ETHRC/yams-closed-carton-box-to-migros-basket-go2
RESUME=${RESUME:-true}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
# RECORD_DEPTH=true → also capture topdown RealSense depth to a PNG-16
# sidecar at <dataset_root>/depth/. Harmless to leave off; RGB is always
# saved regardless.
RECORD_DEPTH=${RECORD_DEPTH:-false}
# Integer divisor applied to depth before PNG encoding. 1 = native 640x480
# (~900 MB/episode). 2 = 320x240, ~4x less disk — the default because a 30 FPS
# 320x240 depth stream is still plenty for manipulation policies, which
# typically downsample image inputs to ~224x224 in the backbone anyway. Set
# to 1 if you need full resolution, or 4 for 160x120 (~16x smaller).
DEPTH_DOWNSAMPLE=${DEPTH_DOWNSAMPLE:-2}
# Any depth pixel > DEPTH_CLIP_MM is set to 0 ("invalid") BEFORE downsampling.
# 0 disables the clip. D455 depth past ~3000 mm indoors is noisy enough to
# hurt policies more than help; 3000 mm is a safe starting point for a
# tabletop topdown view, but default is 0 so existing behavior is preserved.
DEPTH_CLIP_MM=${DEPTH_CLIP_MM:-1500}
export DEPTH_DOWNSAMPLE DEPTH_CLIP_MM
MIN_CAMERA_FPS=$(yq '[.cameras.configs[].fps] | min' "$YAML")
DATASET_FPS=${DATASET_FPS:-$MIN_CAMERA_FPS}
NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_TIME_S=${EPISODE_TIME_S:-45}
RESET_TIME_S=${RESET_TIME_S:-10}
# TASK=${TASK:-Fold the towel.}
TASK=${TASK:-Push the closed box off the table onto a Migros basket on a Go2}
VCODEC=${VCODEC:-auto}
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")
if [ "$RECORD_DEPTH" = "true" ]; then
    # Enable use_depth on every RealSense-backed camera so RealSenseCamera
    # actually opens a depth stream. Round-trip through proper JSON (yq -c
    # emits compact YAML, which jq can't parse) before the jq patch, then
    # back to YAML-or-JSON-both-fine for draccus.
    cameras=$(yq -o=json -I=0 '.cameras.configs' "$YAML" | jq -c '
        with_entries(
            if (.value.type? | tostring | startswith("intelrealsense"))
            then .value.use_depth = true
            else .
            end
        )
    ')
fi
echo "Dataset repo: $REPO"
echo "Dataset root: data/$REPO"
echo "Task: $TASK"
echo "Push to Hub: $PUSH_TO_HUB"
echo "Record depth: $RECORD_DEPTH"
if [ "$RECORD_DEPTH" = "true" ]; then
    echo "Depth downsample: ${DEPTH_DOWNSAMPLE}x min-pool (→ $((640 / DEPTH_DOWNSAMPLE))x$((480 / DEPTH_DOWNSAMPLE)))"
    if [ "$DEPTH_CLIP_MM" -gt 0 ]; then
        echo "Depth clip: pixels > ${DEPTH_CLIP_MM} mm → 0 (invalid)"
    else
        echo "Depth clip: disabled"
    fi
fi

PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

# if [ -d "data/$REPO" ] && [ ! -f "data/$REPO/meta/info.json" ]; then
#     mv "data/$REPO" "data/$REPO.stale.$(date +%s)"
# fi
# rm -rf data/$REPO

if [ "$RESUME" = "true" ] && [ ! -f "data/$REPO/meta/tasks.parquet" ]; then
    RESUME=false
fi
if [ "$RESUME" != "true" ] && [ -d "data/$REPO" ]; then
    rm -rf "data/$REPO"
fi

PYTHONPATH=src uv run python scripts/check_setup.py || exit 1

DATASET_ROOT="data/$REPO"
PYTHONPATH=src uv run python scripts/watch_pose.py --repo-root "$DATASET_ROOT" &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT INT TERM

export PYNPUT_BACKEND_KEYBOARD=xorg
if [ "$RECORD_DEPTH" = "true" ]; then
    RECORD_BIN=(uv run python scripts/record_with_depth.py)
else
    RECORD_BIN=(uv run lerobot-record)
fi
highlight() {
    while IFS= read -r line; do
        # Drop known high-frequency noise
        echo "$line" | grep -qE \
            "Record loop is running slower|No policy or teleoperator provided|frame timeout, returning last frame" \
            && continue
        if echo "$line" | grep -qE "Recording episode|Reset the environment|Stop recording|Re-record episode"; then
            echo ""
            echo "================================================================"
            echo "  >>> $line"
            echo "================================================================"
            echo ""
        else
            echo "$line"
        fi
    done
}

PYTHONPATH=src "${RECORD_BIN[@]}" \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --display_data=false \
    --play_sounds=false \
    --dataset.fps="$DATASET_FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO" \
    --dataset.root="data/$REPO" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true 2>&1 | highlight
    # --dataset.push_to_hub=true \
    # --dataset.encoder_queue_maxsize=1000
    # --dataset.encoder_threads=2
    # --dataset.vcodec=libx264 \
