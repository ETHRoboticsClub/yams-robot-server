## Quick Start

```bash
git clone --recursive https://github.com/ETHRoboticsClub/yams-robot-server.git
cd yams-robot-server
```

## Installation

### Virtual Environment Setup (Recommended)

1. **Install `uv` (if you don't have it):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install the project dependencies:**
   ```bash
   uv sync
   ```

3. **(Optional) ZED Camera Support:**  
   If you have the ZED SDK installed, copy the .whl (Python wheel) file into `src/lerobot_camera_zed/pyzed/`, then run:
   ```bash
   uv sync --extra zed
   ```

## Usage

### Setup Guidance

- **Follower Arms:**  
  Ensure both CAN cables are connected. Verify their presence by running:  
  ```bash
  ip link show
  ```
  Look for both `can_follower_r` and `can_follower_l`.

- **Leader Arms:**  
  Check USB port mapping by plugging in cables one at a time and monitoring:  
  ```bash
  ls /dev/ttyACM*
  ```
  > **Note:**  
  > Ensure the mapping for left and right arm ports is correct in your example script or when supplying arguments to `lerobot`.

- **Reset CAN Devices**

  To reset all CAN devices, run:
  ```bash
  sh third_party/i2rt/scripts/reset_all_can.sh
  ```
---

### Before You Start (Teleoperation & Recording)

> **Important:** Read this before running any teleoperation or recording commands.

- Place the leader arms in a nominal (safe) position. The follower arms will move to match the leader's initial positions.
- Ensure correct mapping of each leader arm to its USB port.
- Identify the correct camera ids by running `python scripts/find_camera.py`

---

### Run Bimanual Teleoperation

**Standalone example script:**
```bash
uv run examples/bi_leader_follower.py --left-leader-port /dev/ttyACM1 --right-leader-port /dev/ttyACM0
```

**Run inside LeRobot:**

```bash
lerobot-teleoperate \
    --robot.type=bi_yams_follower \
    --robot.cameras="{ 
        left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, 
        right_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        topdown: {type: zed, camera_id: 0, width: 640, height: 480, fps: 30}
      }" \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --display_data=true
```

---

### Record Bimanual Teleoperation

```bash
lerobot-record \
    --robot.type=bi_yams_follower \
    --robot.cameras="{ 
        left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, 
        right_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        topdown: {type: zed, camera_id: 0, width: 640, height: 480, fps: 30}
      }" \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --display_data=true \
    --dataset.repo_id=ETHRC/my_dataset \
    --dataset.push_to_hub=true \
    --dataset.num_episodes=1000 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=2 \
    --dataset.single_task="Fold the towel."
```

---