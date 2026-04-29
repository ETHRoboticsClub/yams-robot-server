# Copy & Paste Commands

Commonly-used commands for this project. Copy, paste, tweak as needed.
Setup troubleshooting lives at the bottom.

---

## Inference

To switch which policy runs, edit `scripts/inference.sh` and comment /
uncomment a `POLICY_PATH=...` line before running any of the below.

**One-off test — fresh eval repo, no HF push (safest default):**

```bash
NEW_REPO=true \
REPO=ETHRC/eval_box_test \
NUM_EPISODES=1 \
EPISODE_TIME_S=100 \
RESET_TIME_S=10 \
PUSH_TO_HUB=false \
./scripts/inference.sh --log
```

**Quick single-episode run — reuses existing eval repo:**

```bash
REPO=ETHRC/eval_box_test NUM_EPISODES=1 EPISODE_TIME_S=50 PUSH_TO_HUB=false ./scripts/inference.sh --log
```

---

## Recording

```bash
sudo -i
cd /home/ethrc/Desktop/yams-robot-server
hf auth whoami
NUM_EPISODES=20 EPISODE_TIME_S=10 RESET_TIME_S=0 PUSH_TO_HUB=true ./scripts/record.sh
hf datasets info ETHRC/yams-carton-box-closing
```

---

## Grant `tommaso` SSH access (run as `ethrc`)

```bash
sudo install -d -o tommaso -g tommaso -m 700 /home/tommaso/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINzss9V+etthB7s6Um1kXzkE99lhPHEdHv52cwt5L4eI tom.gazzini@gmail.com' | sudo tee /home/tommaso/.ssh/authorized_keys > /dev/null
sudo chown tommaso:tommaso /home/tommaso/.ssh/authorized_keys
sudo chmod 600 /home/tommaso/.ssh/authorized_keys
```

---

# Setup Problems

## Leader port exists, but some Dynamixel motors are missing

If `/dev/leader-left` or `/dev/leader-right` exists but teleop fails with missing motor IDs, this is probably not a port-mapping problem.

Check the physical leader arm first:

- Make sure the leader cable is correctly plugged in.
- Make sure both LED lights have power supplied.
- Make sure the Dynamixel chain is connected all the way through the arm.

The key clue is that the port is found and some motors respond, but the rest of the expected motors are missing.

Exact error:

```text
Traceback (most recent call last):
  File "/home/ethrc/Desktop/yams-robot-server/.venv/bin/lerobot-teleoperate", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/scripts/lerobot_teleoperate.py", line 250, in main
    teleoperate()
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/configs/parser.py", line 233, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/scripts/lerobot_teleoperate.py", line 224, in teleoperate
    teleop.connect()
  File "/home/ethrc/Desktop/yams-robot-server/src/lerobot_teleoperator_gello/bi_leader.py", line 75, in connect
    self.left_arm.connect()
  File "/home/ethrc/Desktop/yams-robot-server/src/lerobot_teleoperator_gello/leader.py", line 103, in connect
    self.bus.connect()
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/utils/decorators.py", line 39, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py", line 513, in connect
    self._connect(handshake)
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py", line 522, in _connect
    self._handshake()
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/motors/dynamixel/dynamixel.py", line 140, in _handshake
    self._assert_motors_exist()
  File "/home/ethrc/Desktop/yams-robot-server/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py", line 489, in _assert_motors_exist
    raise RuntimeError("\n".join(error_lines))
RuntimeError: DynamixelMotorsBus motor check failed on port '/dev/leader-left':

Missing motor IDs:
  - 4 (expected model: 1190)
  - 5 (expected model: 1190)
  - 6 (expected model: 1190)
  - 7 (expected model: 1190)

Full expected motor list (id: model_number):
{1: 1020, 2: 1020, 3: 1020, 4: 1190, 5: 1190, 6: 1190, 7: 1190}

Full found motor list (id: model_number):
{1: 1020, 2: 1020, 3: 1020}
```

## Innomaker wrist camera resets mid-inference (second run onwards)

**Symptom:** `OpenCVCameraCached(/dev/videoN) exceeded maximum consecutive read failures` during inference, always on the second run. First run is fine.

**Root cause:** The Innomaker U20CAM firmware triggers a USB-level self-reset ~25 seconds after `set_camera_profile.sh` applies `auto_exposure=1` / `exposure_dynamic_framerate=1`. On run 1 the camera is in a fresh state and the reset either doesn't fire or fires while idle. On run 2+ the reset fires mid-stream, the kernel drops the USB device (`uvcvideo: Failed to resubmit video URB (-19)`), and OpenCV reads return `False`.

Confirmed via `journalctl -k`: `usb 1-3: USB disconnect` at exactly the timestamp of the read failures, followed by immediate reconnect under a new device number. The stable `/dev/by-path` symlink survives the reset.

**Fix applied (Option A) — skip profile if already set (`inference.sh`):**
Before calling `set_camera_profile.sh`, check if `auto_exposure` is already `1` (Manual Mode). If so, skip the profile application entirely — no trigger, no reset. This handles the common case where `check_setup.py` already applied the profile before inference.

**Remaining fix (Option B) — reconnect recovery in `OpenCVCameraCached`:**
If the reset fires anyway (e.g. on the very first run, or if `check_setup` wasn't run), the read thread hits 10+ consecutive failures and crashes, killing inference. The fix is to catch this in `_read_loop` or `connect()`: on consecutive failures, call `disconnect()` then `connect()` (with retries) instead of raising. The camera comes back on the same stable symlink path within ~2 seconds. This mirrors what `RealSenseCameraCached` already does via `_reset_busy_device()` + `hardware_reset()`. Without this, Option A is a best-effort guard but not a hard guarantee.

## Right leader: cut cable into motor 4 → flaky `sync_read`

**Symptom:** `check_setup.py` fails intermittently on the right leader with either
`Missing motor IDs: 4` (sometimes 5/6/7) at connect, or
`Failed to sync read 'Present_Position' on ids=[8..14] ... [TxRxResult] There is no status packet!`.
Single pings to each motor mostly succeed; `sync_read` of the full bus fails most of the time. Failure rate climbs as the bus warms up / the arm flexes.

**Root cause:** the 3-pin daisy-chain cable going **into motor 4 on the right leader arm** is physically cut/damaged. The bus partially conducts, so individual pings get through, but back-to-back `sync_read` traffic across the whole chain breaks.

**Fix:**

1. Power off both leader power strips before unplugging anything.
2. On the **right** leader arm (USB cable goes to `/dev/leader-right`, FTDI serial `FT94EW3S`), find the cable that runs into motor 4 — it's the first **small** XL330 motor right after the three big black XM430 motors (counting from the base). It's the cable crossing the visible "step" where the arm necks down.
3. Replace that cable with a known-good 3-pin Dynamixel daisy-chain cable. While you're in there, also reseat the cable on the other side of motor 4 (4 → 5).
4. Power back on, then verify:

```bash
uv run python scripts/check_setup.py
```

It should pass the leaders step on the first try; rerun 3–4 times to confirm no intermittent drops.

If a spare cable isn't available immediately, runtime teleop (`get_action`) uses `num_retry=10` and tolerates the flakiness — but `check_setup.py` uses 0 retries, so it will keep failing until the cable is replaced.
