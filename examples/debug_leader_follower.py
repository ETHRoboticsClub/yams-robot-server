import argparse
from collections import defaultdict
from functools import wraps
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lerobot.cameras.opencv import OpenCVCameraConfig

from lerobot_camera_zed.zed_camera import ZEDCamera, ZEDCameraConfig
from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action
from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

from utils.connection import _free_port
from utils.live_joint_plot import start_joint_plotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
ARMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
RUN_HISTORY_DIR = Path(__file__).resolve().parents[1] / "run_history"


def _build_joint_label_map(section_config: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    joint_labels = section_config.get("joint_labels", {})
    for side in ("left", "right"):
        for joint, label in joint_labels.get(side, {}).items():
            out[f"{side}_{joint}.pos"] = label
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Bimanual leader-follower teleoperation")
    parser.add_argument(
        "--left-leader-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the left leader arm (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--right-leader-port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for the right leader arm (default: /dev/ttyACM1)",
    )
    return parser.parse_args()

cleaned_up = False
bi_leader = None
bi_follower = None
plotter = None
run_started_at = time.time()
trajectory: list[dict[str, Any]] = []


def _joint_only(data: dict[str, Any] | None) -> dict[str, float]:
    if not data:
        return {}
    return {
        k: float(v)
        for k, v in data.items()
        if k.endswith(".pos") and k.startswith(("left_", "right_"))
    }


def _save_run_history() -> None:
    if not trajectory:
        return
    RUN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(run_started_at))
    out_path = RUN_HISTORY_DIR / f"trajectory_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in trajectory:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    logger.info("Saved trajectory: %s", out_path)

def cleanup():
    global cleaned_up, plotter
    if cleaned_up:
        return
    cleaned_up = True
    logger.info("Cleaning up arm connections")
    _save_run_history()
    if bi_follower is not None:
        bi_follower.disconnect()
    if bi_leader is not None:
        bi_leader.disconnect()
    if plotter is not None:
        plotter.close()

def handle_sigint(signum, frame):
    cleanup()
    raise SystemExit(0)

# def monitor_arm_obs(bi_follower, bi_leader):
#     global plotter
#     obs = bi_follower.get_observation(with_cameras=True)
#     act = bi_leader.get_action()
#     plotter.push(obs, act)
    # def fmt(value):
    #     arr = np.asarray(value)
    #     if arr.ndim == 0:
    #         return f"{float(arr):.2f}"
    #     return f"array(shape={arr.shape}, dtype={arr.dtype})"

    # arm_obs = {
    #     key: fmt(value)
    #     for key, value in obs.items()
    #     if key.startswith(("left_", "right_"))
    # }
    # print(arm_obs)

HZ = 200


def _new_timing_stats():
    return defaultdict(lambda: {"n": 0, "sum": 0.0, "min": float("inf"), "max": 0.0})


def _record_timing(stats, name: str, dt_s: float) -> None:
    s = stats[name]
    s["n"] += 1
    s["sum"] += dt_s
    s["min"] = min(s["min"], dt_s)
    s["max"] = max(s["max"], dt_s)


def _format_timing(stats) -> str:
    parts = []
    for name, s in stats.items():
        if not s["n"]:
            continue
        parts.append(
            f"{name}: avg={s['sum']/s['n']*1e3:.1f}ms min={s['min']*1e3:.1f}ms max={s['max']*1e3:.1f}ms"
        )
    return " | ".join(parts)


def time_each_line(fn):
    src_lines, start = inspect.getsourcelines(fn)
    labels = {
        start + i: (line.strip() or "<blank>")[:40]
        for i, line in enumerate(src_lines)
        if line.strip() and not line.strip().startswith("#")
    }

    @wraps(fn)
    def wrapped(*args, **kwargs):
        line_dt = defaultdict(float)
        prev_line = None
        prev_t = time.perf_counter()

        def tracer(frame, event, arg):
            nonlocal prev_line, prev_t
            if frame.f_code is fn.__code__ and event == "line":
                now = time.perf_counter()
                if prev_line is not None:
                    line_dt[prev_line] += now - prev_t
                prev_line = frame.f_lineno
                prev_t = now
            return tracer

        prev_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            out = fn(*args, **kwargs)
        finally:
            now = time.perf_counter()
            if prev_line is not None:
                line_dt[prev_line] += now - prev_t
            sys.settrace(prev_trace)

        return out, {labels.get(n, f"L{n}"): dt for n, dt in line_dt.items()}

    return wrapped


# @time_each_line
def run_loop_iteration(bi_follower, bi_leader, plotter):
    obs = bi_follower.get_observation(with_cameras=False)
    bi_leader_action = bi_leader.get_action()
    if bi_leader_action is None:
        return

    plotter.push(obs, bi_leader_action)
    for msg in plotter.pop_control_messages():
        logger.info("UI control message: %s", msg)
    trajectory.append({"t": time.time(), "obs": _joint_only(obs), "act": _joint_only(bi_leader_action)})

    bi_follower.send_action(bi_leader_action)

    time.sleep(1 / HZ)


def main():
    global bi_leader, bi_follower, plotter
    subprocess.run(["sh", str(Path(__file__).resolve().parents[1] / "third_party/i2rt/scripts/reset_all_can.sh")], check=True)

    args = parse_args()
    with open(ARMS_CONFIG_PATH, "r") as f:
        arms_config = yaml.safe_load(f)

    follower_config = arms_config["follower"]
    follower_joint_label_map = _build_joint_label_map(follower_config)
    leader_joint_label_map = _build_joint_label_map(arms_config.get("leader", {}))
    camera_label_map = arms_config.get("cameras", {}).get("labels", {})
    left_follower_server_port = follower_config["left_arm"]["server_port"]
    right_follower_server_port = follower_config["right_arm"]["server_port"]

    # Free from old subprocesses
    _free_port(left_follower_server_port)
    _free_port(right_follower_server_port)

    zed_cam_id = None
    available_zed_cameras = ZEDCamera.find_cameras()
    if available_zed_cameras:
        zed_cam_id = available_zed_cameras[0]["id"]
    else:
        logger.warning("No ZED cameras found.")

    # get first camera for now - generalise later
    cameras = {
        "left_wrist": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=2,
            fps=30,
            width=640,
            height=480,
        ),
    }
    
    if zed_cam_id:
        cameras["topdown"] = ZEDCameraConfig(
            camera_id=zed_cam_id,
            width=640,
            height=480,
            fps=30,
        )

    bi_follower_config = BiYamsFollowerConfig(
        left_arm_server_port=left_follower_server_port,
        right_arm_server_port=right_follower_server_port,
        cameras=cameras
    )

    bi_leader_config = BiYamsLeaderConfig(
        left_arm_port=args.left_leader_port,
        right_arm_port=args.right_leader_port,
    )

    try:
        bi_leader = BiYamsLeader(bi_leader_config)
        bi_leader.connect()

        bi_follower = BiYamsFollower(bi_follower_config)
        bi_follower.connect()

        signal.signal(signal.SIGINT, handle_sigint)
        

        plotter = start_joint_plotter(
            bi_follower,
            hz=60,
            history_s=10,
            backend="web",
            web_port=8988,
            camera_hz=5,
            follower_joint_label_map=follower_joint_label_map,
            leader_joint_label_map=leader_joint_label_map,
            camera_label_map=camera_label_map,
        )

        while True:
            run_loop_iteration(bi_follower, bi_leader, plotter)

        # reset_after_seconds = 10
        # i = 0
        # timing = _new_timing_stats()
        # timing_window_start = time.perf_counter()
        # timing_window_iters = 0
        # while True:
        #     _, line_timing = run_loop_iteration(bi_follower, bi_leader, plotter)
        #     for name, dt_s in line_timing.items():
        #         _record_timing(timing, name, dt_s)

        #     i += 1
        #     timing_window_iters += 1
        #     if i % HZ == 0:
        #         wall_s = time.perf_counter() - timing_window_start
        #         logger.info(
        #             "timings over %.2fs (%d loops): %s",
        #             wall_s,
        #             timing_window_iters,
        #             _format_timing(timing),
        #         )
        #         logger.info("%s loop-seconds passed", i / HZ)
        #         timing = _new_timing_stats()
        #         timing_window_start = time.perf_counter()
        #         timing_window_iters = 0
        #     if i == reset_after_seconds * HZ:
        #         logger.info("Resetting arms")
        #         for arm in [bi_follower.left_arm, bi_follower.right_arm]:
        #             slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})

        return
    finally:
        cleanup()

    # freq = 200  # Hz

    # bi_leader_action = bi_leader.get_action()

    # slow_move(bi_follower.left_arm, split_arm_action(bi_leader_action, "left_"))
    # slow_move(bi_follower.right_arm, split_arm_action(bi_leader_action, "right_"))

    # start_time = time.time()
    # count = 0
    # try:
    #     while True:
    #         count += 1
    #         bi_leader_action = bi_leader.get_action()
    #         if bi_leader_action is None:
    #             continue
    #         bi_follower.send_action(bi_leader_action)
    #         time.sleep(1 / freq)
    #         time_elapsed = time.time() - start_time
    #         if count % 400 == 0:
    #             print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
    #         if time_elapsed >= 0.05:
    #             print(f"Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds")
    #         start_time = time.time()

    # except KeyboardInterrupt:
    #     print("\nStopping teleop...")
    # finally:
    #     for arm in [bi_follower.left_arm, bi_follower.right_arm]:
    #         slow_move(arm, {f"{name}.pos": 0.0 for name in arm.config.joint_names})
    #     bi_leader.disconnect()
    #     bi_follower.disconnect()


if __name__ == "__main__":
    main()
