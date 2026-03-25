## Quick Start

```bash
git clone --recursive https://github.com/ETHRoboticsClub/yams-robot-server.git
git submodule update --init --recursive
cd yams-robot-server
```

## Important teleop info

## Installation

1. **Install `uv` (if you don't have it):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install the project dependencies:**
   ```bash
   uv sync
   ```

3. **ZED Camera Support: (instructions may be outdated)**  
   1. Install the ZED SDK (Download script from https://www.stereolabs.com/en-fr/developers and run).
   Only version 5.1 works I think (not the most recent 5.2)
   2. Copy the .whl (Python wheel) file into `src/lerobot_camera_zed/pyzed/`, then run:
   ```bash
   uv sync
   ```

<!-- ## Usage

Set up USBs with `sudo .venv/bin/python scripts/setup_leader_ports.py` -->

<!-- ### Setup Guidance

- **Follower Arms:**  
  Ensure both CAN cables are connected. Verify their presence by running:  
  ```bash
  ip link show
  ```
  Look for both `can_follower_r` and `can_follower_l`.

  For first time setup of the arms, please see `third_party/i2rt/doc/set_persist_id_socket_can.md`
  **Reset CAN buses:** `sudo sh third_party/i2rt/scripts/reset_all_can.sh`
  ****

- **Leader Arms:**  
  Check USB port mapping by plugging in cables one at a time and monitoring:  
  ```bash
  ls /dev/ttyUSB*
  ```
  Increase hz by changing the latency timer to 1ms (some boards default to 16ms capping us at 50hz):
  ```
  echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
  echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
  ```
  > **Note:**  
  > Ensure the mapping for left and right arm ports is correct in your example script or when supplying arguments to `lerobot`.

- **Reset CAN Devices**

  To reset all CAN devices, run:
  ```bash
  sh third_party/i2rt/scripts/reset_all_can.sh
  ```
--- -->

## Setup (Important)

> **Important:** Read this before running any teleoperation or recording commands.
- List followers (CAN) with `ip link show`. Make sure both are connected.
  - Reset CAN busses: `sudo sh third_party/i2rt/scripts/reset_all_can.sh`.
- Set up USBs with `sudo .venv/bin/python scripts/setup_leader_ports.py`. Check that t
- Precisely place the leader arms in the zero position for calibration.
- Precisely place the follower arms in the zero position for calibration.
- Turn on power and calibrate follower arms with `uv run scripts/compute_offsets.py`.
- Identify the correct camera ids by running `uv run lerobot-find-cameras`. Make sure mapping is correct in `arms.yaml` in the `index_or_path` field. You can find their images in `outputs/captured_images/`.
<!-- - Place the leader arms in a nominal (safe) position. The follower arms will move to match the leader's initial positions. -->
<!-- - Ensure correct mapping of each leader arm to its USB port. **If they are swapped arms behave erratically and damage themselves. They may switch when swapping around USB ports.** -->
  <!-- - Check port mapping with ls `/dev/ttyACM*` -->
  <!-- - Make sure it's correct in (arms.yaml) -->
---

### Run Bimanual

```bash
./scripts/teleop.sh
```
```bash
./scripts/record.sh
```
<!-- **Standalone example script:**
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
``` -->

---

<!-- ### Record Bimanual Teleoperation

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

--- -->