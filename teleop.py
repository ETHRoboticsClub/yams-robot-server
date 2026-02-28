import argparse
import logging
import signal
import time
from pathlib import Path

from utils.lifecycle import build_cleanup_and_sigint
from utils.teleop_data import (
    append_recording_sample,
    joint_only,
    new_recording_state,
    save_run_history,
    start_recording,
    stop_recording,
)
from utils.teleop_setup import setup_arms_cameras_plotter
from utils.time_each_line import format_line_timing, time_each_line

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
ARMS_CONFIG_PATH = PROJECT_ROOT / "configs" / "arms.yaml"
RUN_HISTORY_DIR = PROJECT_ROOT / "run_history"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"
HZ = 200

def parse_args():
    parser = argparse.ArgumentParser(description="Bimanual leader-follower teleoperation")
    parser.add_argument(
        "--allow-no-cams",
        "--allow_no_cams",
        dest="allow_no_cams",
        action="store_true",
        help="Run teleop without configuring cameras",
    )
    return parser.parse_args()


def handle_ui_messages(plotter, recording_state) -> None:
    for msg in plotter.pop_control_messages():
        if msg.get("type") != "trajectory":
            logger.info("UI control message: %s", msg)
            continue
        command = msg.get("command")
        if command == "start":
            if recording_state["is_active"]:
                logger.info("Trajectory recording already active")
                continue
            start_recording(recording_state)
            logger.info("Started trajectory recording")
        elif command == "stop":
            if not recording_state["is_active"]:
                logger.info("Trajectory recording is not active")
                continue
            stop_recording(recording_state, TRAJECTORIES_DIR, logger)
            logger.info("Stopped trajectory recording")


# @time_each_line
def run_loop(bi_follower, bi_leader, plotter, run_history, recording_state):
    while True:
        obs = bi_follower.get_observation(with_cameras=False)
        bi_leader_action = bi_leader.get_action()
        if bi_leader_action is None:
            return

        plotter.push(obs, bi_leader_action)
        handle_ui_messages(plotter, recording_state)

        now = time.time()
        run_history.append({"t": now, "obs": joint_only(obs), "act": joint_only(bi_leader_action)})
        append_recording_sample(recording_state, now, obs, bi_leader_action)

        bi_follower.send_action(bi_leader_action)

        time.sleep(1 / HZ)


# def run_loop_iteration(bi_follower, bi_leader, plotter, run_history, recording_state):


def main():
    args = parse_args()
    bi_leader, bi_follower, plotter = setup_arms_cameras_plotter(args, ARMS_CONFIG_PATH, logger)

    run_started_at = time.time()
    run_history: list[dict[str, Any]] = []
    recording_state = new_recording_state()

    def save_histories(trajectory, run_started_at, run_history_dir, logger):
        save_run_history(trajectory, run_started_at, run_history_dir, logger)
        if recording_state["is_active"]:
            stop_recording(recording_state, TRAJECTORIES_DIR, logger)

    cleanup, handle_sigint = build_cleanup_and_sigint(
        logger,
        save_histories,
        run_history,
        run_started_at,
        RUN_HISTORY_DIR,
        bi_follower,
        bi_leader,
        plotter,
    )

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        plotter.start()

        _, line_timing = run_loop(bi_follower, bi_leader, plotter, run_history, recording_state)
        logger.info("run_loop line timing: %s", format_line_timing(line_timing))

        return
    finally:
        cleanup()

if __name__ == "__main__":
    main()
