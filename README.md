## Quick Start (on-site machine)

Before teleop/recording, run the setup checker:
```bash
uv run python scripts/check_setup.py
```

If this passes, you can proceed with the following teleop/recording commands. If something doesn't work, walk through the [Troubleshooting checklist](#teleop-troubleshooting-checklist).

```bash
./scripts/teleop.sh
```

```bash
sudo -i
cd /home/ethrc/Desktop/yams-robot-server
./scripts/record.sh
```
#command to record episodes 
NUM_EPISODES=50 EPISODE_TIME_S=50 PUSH_TO_HUB=true bash scripts/record.sh --log

### Hotkeys during recording

lerobot's record loop listens for these keys — no extra setup needed.

| Key | Action |
|-----|--------|
| `→` Right arrow | End the current episode early and save it (advance to next) |
| `←` Left arrow | Discard the current episode and re-record it |
| `Esc` | Stop the entire recording session and finalize the dataset |

---

## First-time setup (new devs, read this before cloning repo on personal machine)apte

### 1. Request access to the private `i2rt` fork

This repo depends on `third_party/i2rt`, which is a submodule pointing at the private fork [`ETHRoboticsClub/i2rt`](https://github.com/ETHRoboticsClub/i2rt). Ping an ETHRC admin to add your GitHub account to the org / grant read access before cloning. `uv sync` will fail without it.

### 2. Clone with submodules

```bash
git clone <this-repo-url>
cd yams-robot-server
git submodule update --init --recursive
```

If you hit `upload-pack: not our ref <sha>`, the pinned submodule commit apteisn't reachable on the fork (force-pushed or never pushed). Fall back to the branch tip:

```bash
git -C third_party/i2rt fetch --all
git submodule update --init --remote third_party/i2rt
git -C third_party/i2rt checkout ethrc-fork
```

Do **not** `git add third_party/i2rt` after this — it would bump the submodule pin for everyone.

### 3. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 4. Install dependenciesfrom a separate barrel jack (usually plugged into th

```bash
uv sync
```

Use `uv run ...` for all Python entry points.

---

## Teleop troubleshooting checklist

Only needed if teleop or recording isn't working. Run through these on the robot machine:

- Turn on both power strips. Follower fans should start making noise.
- List followers (CAN) with `ip link show`. Make sure both are connected.
  - If needed, reset CAN busses: `sudo sh third_party/i2rt/scripts/reset_all_can.sh`.
- Set up leader USBs: `sudo .venv/bin/python scripts/setup_leader_ports.py`.
- Precisely place the leader arms in the zero position for calibration.
- Calibrate leader arms with `uv run python scripts/compute_offsets.py`.
- Follower zeros are stored on the DM motors, not in the leader offset YAMLs. If a follower joint jumps when teleop starts, place that follower joint in mechanical zero and reset that motor zero.
  - Example right joint 3: `uv run python third_party/i2rt/i2rt/motor_config_tool/set_zero.py --channel <right_arm.can_port> --motor_id 3`.
- Identify the correct camera ids by running `uv run lerobot-find-cameras`. Make sure mapping is correct in `configs/arms.yaml` in the `index_or_path` field. You can find their images in `outputs/captured_images/`.
- Make sure the output images of the wrist cameras look properly exposed. If needed, tweak the fixed baseline in `scripts/set_camera_profile.sh` or the runtime auto-exposure knobs in `configs/arms.yaml`.
- Type `realsense-viewer`, load the config from `configs/realsense.json`, and make sure it looks good. If it looks bad, overwrite `configs/realsense.json` with better settings.
- DO THIS FOR WRIST CAMERAS, NOT ZED CAMERA: `./scripts/set_camera_profile.sh /dev/video<ID>`
- Run `uv run lerobot-find-cameras` again, check outputs to make sure they look normal.
- Make sure the cameras are focused.
- Wrist cameras can also self-adjust manual exposure at runtime with the `auto_exposure_*` fields under each `opencv-cached` camera in `configs/arms.yaml`.
