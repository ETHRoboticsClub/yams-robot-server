## Quick Start (on-site machine)

```bash
./scripts/teleop.sh
```

```bash
sudo -i
cd /home/ethrc/Desktop/yams-robot-server
./scripts/record.sh
```

If something doesn't work, walk through the [Troubleshooting / pre-flight checklist](#troubleshooting--pre-flight-checklist).

---

## First-time setup (new devs, read this before cloning repo on personal machine)

### 1. Request access to the private `i2rt` fork

This repo depends on `third_party/i2rt`, which is a submodule pointing at the private fork [`ETHRoboticsClub/i2rt`](https://github.com/ETHRoboticsClub/i2rt). Ping an ETHRC admin to add your GitHub account to the org / grant read access before cloning. `uv sync` will fail without it.

### 2. Clone with submodules

```bash
git clone <this-repo-url>
cd yams-robot-server
git submodule update --init --recursive
```

If you hit `upload-pack: not our ref <sha>`, the pinned submodule commit isn't reachable on the fork (force-pushed or never pushed). Fall back to the branch tip:

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

### 4. Install dependencies

```bash
uv sync
```

Use `uv run ...` for all Python entry points.

---

## Teleop troubleshooting / pre-flight checklist

Only needed if teleop or recording isn't working. Run through these on the robot machine:

- Turn on both power strips. Follower fans should start making noise.
- List followers (CAN) with `ip link show`. Make sure both are connected.
  - If needed, reset CAN busses: `sudo sh third_party/i2rt/scripts/reset_all_can.sh`.
- Set up leader USBs: `sudo .venv/bin/python scripts/setup_leader_ports.py`.
- Precisely place the leader arms in the zero position for calibration.
- Precisely place the follower arms in the zero position for calibration.
- Calibrate follower arms: `uv run scripts/compute_offsets.py`.
- Identify the correct camera ids: `uv run lerobot-find-cameras`. Make sure mapping is correct in `configs/arms.yaml` under `index_or_path`. Images land in `outputs/captured_images/`.
- Make sure the output images of the wrist cameras look properly exposed. If needed, tweak the fixed baseline in `scripts/set_camera_profile.sh` or the runtime auto-exposure knobs in `configs/arms.yaml`.
- Type `realsense-viewer`, load the config from `configs/realsense.json`, and make sure it looks good. If it looks bad, overwrite `configs/realsense.json` with better settings.
- DO THIS FOR WRIST CAMERAS, NOT ZED CAMERA: `./scripts/set_camera_profile.sh /dev/video<ID>`
- Run `uv run lerobot-find-cameras` again, check outputs to make sure they look normal.
- Make sure the cameras are focused.
- Wrist cameras can also self-adjust manual exposure at runtime with the `auto_exposure_*` fields under each `opencv-cached` camera in `configs/arms.yaml`.
