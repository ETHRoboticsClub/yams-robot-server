import argparse
import cProfile
import io
import logging
import os
import pstats
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action

from utils.lifecycle import build_cleanup_and_sigint
from utils.teleop_data import TRAJECTORIES_DIR, cameras_only, joint_only, load_task_names, load_task_config, save_run_history
from utils.teleop_setup import setup_arms_cameras_plotter
from utils.time_each_line import format_timing, new_timing_stats, record_timing, time_each_line
from plotting.live_joint_plot import LiveJointPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
ARMS_CONFIG_PATH = PROJECT_ROOT / "configs" / "arms.yaml"
RUN_HISTORY_DIR = PROJECT_ROOT / "run_history"
HZ = 200
CAM_HZ = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Bimanual leader-follower teleoperation")
    parser.add_argument(
        "--skip-cams",
        "--skip_cams",
        dest="skip_cams",
        action="store_true",
        help="Skip camera configuration",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the teleop loop with cProfile (including threads)",
    )
    return parser.parse_args()


def camera_loop(bi_follower, latest_obs, obs_lock, stop_event, plotter: LiveJointPlotter, trajectory, collecting):
    deadline = time.monotonic()
    with_cameras = True

    while not stop_event.is_set():
        plotter.process_trajectory_controls(trajectory, collecting)

        try:
            obs = bi_follower.get_observation(with_cameras=with_cameras)
        except Exception as exc:
            if with_cameras:
                logger.warning("Camera error at runtime (%s). Continuing without cameras.", str(exc))
                with_cameras = False
                continue
            raise
        with obs_lock:
            latest_obs.update(obs)
        
        deadline += 1 / CAM_HZ
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        
        bi_leader_action = None
        plotter.push(obs, bi_leader_action)
        if collecting.is_set():
            trajectory.append({"t": time.time(), "obs": joint_only(obs), "act": joint_only(bi_leader_action), "cams": cameras_only(obs)})


def run_loop(bi_follower, bi_leader, plotter, trajectory, collecting, report_hz=False):
    stop_event = threading.Event()
    latest_obs: dict[str, Any] = {}
    obs_lock = threading.Lock()
    cam_thread = threading.Thread(target=camera_loop, args=(bi_follower, latest_obs, obs_lock, stop_event, plotter, trajectory, collecting), daemon=True)
    cam_thread.start()

    deadline = time.monotonic()
    t0 = time.monotonic()
    iters = 0
    try:
        while True:
            bi_leader_action = bi_leader.get_action()
            if bi_leader_action is None:
                return

            with obs_lock:
                obs = dict(latest_obs)
            
            bi_follower.send_action(bi_leader_action)

            iters += 1
            deadline += 1 / HZ
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        elapsed = time.monotonic() - t0
        stop_event.set()
        cam_thread.join()
        if report_hz and iters > 0:
            logger.info("Teleop loop: %.1f Hz over %.1f s (%d iters)", iters / elapsed, elapsed, iters)


def main():
    args = parse_args()
    bi_leader, bi_follower, plotter = setup_arms_cameras_plotter(args, ARMS_CONFIG_PATH, logger)

    run_started_at = time.time()
    trajectory: list[dict[str, Any]] = []
    collecting = threading.Event()
    task_names = load_task_names()
    task_config = load_task_config()

    plotter.trajectory_dir = TRAJECTORIES_DIR
    plotter.task_names = task_names
    plotter.task_goals = {t['name']: t.get('goal') for t in task_config}

    cleanup, handle_sigint = build_cleanup_and_sigint(
        logger,
        save_run_history,
        trajectory,
        run_started_at,
        RUN_HISTORY_DIR,
        bi_follower,
        bi_leader,
        plotter,
    )

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        plotter.start()

        if args.profile:
            _thread_profiles: list[cProfile.Profile] = []
            _lock = threading.Lock()
            _orig_submit = bi_leader._pool.submit

            def _profiled_submit(fn, *a, **kw):
                def _wrapped():
                    p = cProfile.Profile()
                    p.enable()
                    try:
                        return fn(*a, **kw)
                    finally:
                        p.disable()
                        with _lock:
                            _thread_profiles.append(p)
                return _orig_submit(_wrapped)

            bi_leader._pool.submit = _profiled_submit

            main_prof = cProfile.Profile()
            main_prof.enable()
            try:
                run_loop(bi_follower, bi_leader, plotter, trajectory, collecting, report_hz=True)
            finally:
                main_prof.disable()
                stats = pstats.Stats(main_prof, stream=(s := io.StringIO()))
                for p in _thread_profiles:
                    stats.add(p)
                stats.sort_stats("cumulative")
                stats.print_stats()
                print(s.getvalue())
                prof_path = PROJECT_ROOT / "teleop.prof"
                stats.dump_stats(prof_path)
                logger.info("Profile saved to %s", prof_path)
        else:
            run_loop(bi_follower, bi_leader, plotter, trajectory, collecting, report_hz=True)

        return
    finally:
        cleanup()

if __name__ == "__main__":
    main()
