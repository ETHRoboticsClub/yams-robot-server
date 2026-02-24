import logging
import os
import signal
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

def _free_port(port: int) -> None:
    """Kill any process currently listening on *port* (TCP)."""
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return  # nothing listening or fuser not installed

    for pid_str in out.split():
        try:
            pid = int(pid_str)
            if pid != os.getpid():
                logger.warning(f"Killing stale process {pid} on port {port}")
                os.kill(pid, signal.SIGKILL)
        except (ValueError, ProcessLookupError):
            pass

    # Give the OS a moment to release the socket
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
