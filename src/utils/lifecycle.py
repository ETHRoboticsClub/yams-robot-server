import subprocess
from pathlib import Path

from utils.connection import _free_port

RESET_ALL_CAN_SCRIPT = Path(__file__).resolve().parents[2] / "third_party/i2rt/scripts/reset_all_can.sh"


def build_cleanup_and_sigint(
    logger,
    save_run_history,
    trajectory,
    run_started_at,
    run_history_dir,
    bi_follower,
    bi_leader,
    plotter,
):
    cleaned_up = False

    def cleanup():
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        logger.info("Cleaning up arm connections")
        save_run_history(trajectory, run_started_at, run_history_dir, logger)
        if bi_follower is not None:
            bi_follower.disconnect()
        if bi_leader is not None:
            bi_leader.disconnect()
        if plotter is not None:
            plotter.close()

    def handle_sigint(signum, frame):
        cleanup()
        raise SystemExit(0)

    return cleanup, handle_sigint


def run_pre_setup(*ports: int):
    subprocess.run(["bash", str(RESET_ALL_CAN_SCRIPT)], check=True)
    for port in ports:
        _free_port(port)
