import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _kill_pids(pids: list[int], label: str) -> None:
    for pid in pids:
        try:
            if pid != os.getpid():
                logger.warning(f"Killing stale process {pid} on {label}")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pids_using(target: str) -> list[int]:
    try:
        out = subprocess.check_output(
            ["fuser", target],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [int(pid) for pid in out.split() if pid.isdigit()]


def _free_port(port: int | str) -> None:
    """Kill any process currently using *port*."""
    target = f"{port}/tcp" if isinstance(port, int) else port
    pids = _pids_using(target)
    if not pids:
        return
    _kill_pids(pids, f"port {port}")

    # Give the OS a moment to release the socket
    time.sleep(0.3)


def _free_v4l_devices(name: str) -> None:
    found = False
    for device_dir in Path("/sys/class/video4linux").glob("video*"):
        label_path = device_dir / "name"
        if not label_path.exists():
            continue
        if name.lower() not in label_path.read_text().lower():
            continue
        video_path = f"/dev/{device_dir.name}"
        pids = _pids_using(video_path)
        if not pids:
            continue
        found = True
        _kill_pids(pids, video_path)
    if found:
        time.sleep(0.3)


def _wait_for_server(port: int, timeout: float = 120.0, poll: float = 0.5) -> None:
    """Block until a TCP connection to *port* succeeds, or raise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return
        except OSError:
            time.sleep(poll)
    raise TimeoutError(f"Server on port {port} did not start within {timeout}s")
