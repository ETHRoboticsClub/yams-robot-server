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

---

REPO=ETHRC/eval_box_test NUM_EPISODES=1 EPISODE_TIME_S=50 PUSH_TO_HUB=false ./scripts/inference.sh --log

 NEW_REPO=true \
REPO=ETHRC/eval_box_test \
NUM_EPISODES=1 \
EPISODE_TIME_S=20 \
RESET_TIME_S=10 \
PUSH_TO_HUB=false \
./scripts/inference.sh --log


# RECORD
sudo -i
cd /home/ethrc/Desktop/yams-robot-server
hf auth whoami
NUM_EPISODES=20 EPISODE_TIME_S=10 RESET_TIME_S=0 PUSH_TO_HUB=true ./scripts/record.sh
hf datasets info ETHRC/yams-carton-box-closing
