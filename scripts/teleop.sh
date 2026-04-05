
pgrep -f /home/ethrc/Code/yams-robot-server | grep -vx "$$" | xargs -r kill

YAML=configs/arms.yaml
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")
PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT')"
sh third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

set -x
uv run lerobot-teleoperate \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --display_data=false \
    --fps=250 \
    --robot.cameras="$cameras" 
set +x