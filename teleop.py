import argparse
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
from lerobot_robot_yams.utils.utils import slow_move, split_arm_action

from utils.lifecycle import build_cleanup_and_sigint
from utils.teleop_data import joint_only, save_run_history
from utils.teleop_setup import setup_arms_cameras_plotter
from utils.time_each_line import format_timing, new_timing_stats, record_timing, time_each_line

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

def parse_args():
    parser = argparse.ArgumentParser(description="Bimanual leader-follower teleoperation")
    parser.add_argument(
        "--skip-cams",
        "--skip_cams",
        dest="skip_cams",
        action="store_true",
        help="Skip camera configuration",
    )
    return parser.parse_args()


# @time_each_line
def run_loop(bi_follower, bi_leader, plotter, trajectory):
    while True:
        # obs = bi_follower.get_observation(with_cameras=False)
        bi_leader_action = bi_leader.get_action()
        if bi_leader_action is None:
            return

        # plotter.push(obs, bi_leader_action)
        # for msg in plotter.pop_control_messages():
        #     logger.info("UI control message: %s", msg)
        # trajectory.append({"t": time.time(), "obs": joint_only(obs), "act": joint_only(bi_leader_action)})

        bi_follower.send_action(bi_leader_action)

        time.sleep(1 / HZ)

# def run_loop_iteration(bi_follower, bi_leader, plotter, trajectory):
#     obs = bi_follower.get_observation(with_cameras=False)
#     bi_leader_action = bi_leader.get_action()
#     if bi_leader_action is None:
#         return

#     plotter.push(obs, bi_leader_action)
#     for msg in plotter.pop_control_messages():
#         logger.info("UI control message: %s", msg)
#     trajectory.append({"t": time.time(), "obs": joint_only(obs), "act": joint_only(bi_leader_action)})

#     bi_follower.send_action(bi_leader_action)

#     time.sleep(1 / HZ)


def main():
    args = parse_args()
    bi_leader, bi_follower, plotter = setup_arms_cameras_plotter(args, ARMS_CONFIG_PATH, logger)

    run_started_at = time.time()
    trajectory: list[dict[str, Any]] = []

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

        run_loop(bi_follower, bi_leader, plotter, trajectory)

        return
    finally:
        cleanup()

if __name__ == "__main__":
    main()
