## Quick Start

```bash
git submodule update --init --recursive
```

## Important teleop info

## Installation

1. **Install `uv` (if you don't have it):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Install the project dependencies:**
   ```bash
   uv sync
   ```

## Setup (Important)
** Before running use the god setup script to guide you through all problems **
```bash
uv run python scripts/check_setup.py
```
## IF WORKS, YOU CAN SKIP THE NEXT PART.

> **Important:** Read this before running any teleoperation or recording commands.
- Turn on both power strips. Follower fans should start making noise.
- List followers (CAN) with `ip link show`. Make sure both are connected.
  - If you need, reset CAN busses with this: `sudo sh third_party/i2rt/scripts/reset_all_can.sh`.
- Set up leader USBs with `sudo .venv/bin/python scripts/setup_leader_ports.py`.
- Precisely place the leader arms in the zero position for calibration.
- Calibrate leader arms with `uv run python scripts/compute_offsets.py`.
- Follower zeros are stored on the DM motors, not in the leader offset YAMLs. If a follower joint jumps when teleop starts, place that follower joint in mechanical zero and reset that motor zero.
  - Example right joint 3: `uv run python third_party/i2rt/i2rt/motor_config_tool/set_zero.py --channel <right_arm.can_port> --motor_id 3`.
- Identify the correct camera ids by running `uv run lerobot-find-cameras`. Make sure mapping is correct in `arms.yaml` in the `index_or_path` field. You can find their images in `outputs/captured_images/`.
- Make sure the output images of the wrist cameras look properly exposed. If needed, tweak the fixed baseline in `scripts/set_camera_profile.sh` or the runtime auto-exposure knobs in `configs/arms.yaml`.
- Type `realsense-viewer`, load the config from configs/realsense.json, and make sure it looks good. If it looks bad, overwrite the configs/realsense.json with better settings.
- DO THIS FOR WRIST CAMERAS, NOT ZED CAMERA: `./scripts/set_camera_profile.sh /dev/video<ID>`
- Run `uv run lerobot-find-cameras` again, check outputs to make sure they look normal.
- Make sure the cameras are focused.
- Wrist cameras can also self-adjust manual exposure at runtime with the `auto_exposure_*` fields under each `opencv-cached` camera in `configs/arms.yaml`.
- Check the setup without resetting CAN or starting teleop with `uv run python scripts/check_setup.py`.
---

### Run Bimanual

```bash
./scripts/teleop.sh
```
```bash
sudo -i
cd /home/ethrc/Desktop/yams-robot-server
./scripts/record.sh
```

---
