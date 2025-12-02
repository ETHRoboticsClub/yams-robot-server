## Quick Start

```bash
git clone --recursive https://github.com/ETHRoboticsClub/yams-robot-server.git
cd yams-robot-server
```

## Installation

### Virtual Environment (Recommended)

First, install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Now install the dependencies:
```bash
uv sync
```

## Usage

- **Follower arms:** Check CAN port mapping by plugging cables one at a time and monitoring `ip link show`
- **Leader arms:** Check USB port mapping by plugging cables one at a time and monitoring `ls /dev/ttyACM*`

### Reset CAN Devices
```bash
sh third_party/i2rt/scripts/reset_all_can.sh
```
### Run Bimanual Teleoperation
**Important:** Put the leader arms into a nominal position before starting the script. The follower arms will move to the initial leader arm position. 

Run standalone example script:
```bash
uv run examples/bi_leader_follower.py
```

Run inside lerobot:

```bash
lerobot-teleoperate \
    --robot.type=bi_yams_follower \
    --robot.cameras="{ 
        left_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, 
        right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},
        topdown: {type: zed, camera_id: 0, width: 640, height: 480, fps: 30}
      }" \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port=/dev/ttyACM1 \
    --teleop.right_arm_port=/dev/ttyACM0 \
    --display_data=true
```

### Record Bimanual Teleoperation

Run inside lerobot:

```bash
lerobot-record \
    --robot.type=bi_yams_follower \
    --robot.cameras="{ 
        left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, 
        right_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        topdown: {type: zed, camera_id: 0, width: 640, height: 480, fps: 30}
      }" \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port=/dev/ttyACM1 \
    --teleop.right_arm_port=/dev/ttyACM0 \
    --display_data=true \
    --dataset.repo_id=ETHRC/my_dataset \
    --dataset.push_to_hub=true \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Fold the towel."
```