"""Post-episode pose watcher.

Run in the background during recording. Polls the LeRobot dataset videos
dir every 2s; when a new episode_*.mp4 shows up, decodes the first 30
frames, averages them, measures pose vs the committed topdown reference,
and alerts on drift via terminal bell + colored stderr line. OK episodes
log to outputs/logs/pose-watch.out only.

Launched automatically by scripts/record.sh in the background. Self-exits
when its parent PID dies, so a SIGKILL of record.sh does not leave an
orphan.
"""
from __future__ import annotations
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.camera_pose import evaluate_pose, load_reference  # noqa: E402

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"
BELL = "\a"

POLL_INTERVAL_S = 2.0
FRAMES_TO_AVERAGE = 30
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"
LOG_PATH = LOG_DIR / "pose-watch.out"

_stop = False


def _handle_signal(*_args) -> None:
    global _stop
    _stop = True


def _find_videos_dir(repo_root: Path) -> Path | None:
    candidates = list((repo_root / "videos").glob("chunk-*/observation.images.topdown"))
    return candidates[0] if candidates else None


def _read_first_frames(mp4_path: Path, n: int) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        frames: list[np.ndarray] = []
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                return None
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _average(frames: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def _episode_index(mp4_path: Path) -> str:
    return mp4_path.stem.replace("episode_", "")


def _log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {message}\n"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, help="LeRobot dataset root dir")
    parser.add_argument("--parent-pid", type=int, default=os.getppid())
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    repo_root = Path(args.repo_root)
    reference = load_reference()
    print(f"pose-watch: reference loaded, polling {repo_root}/videos/", file=sys.stderr)
    _log(f"watcher started, repo_root={repo_root}")

    seen: set[str] = set()
    videos_dir: Path | None = None

    while not _stop:
        # Self-exit if parent died (SIGKILL of record.sh).
        try:
            os.kill(args.parent_pid, 0)
        except ProcessLookupError:
            _log("parent gone, exiting")
            break

        if videos_dir is None:
            videos_dir = _find_videos_dir(repo_root)
            if videos_dir is None:
                time.sleep(POLL_INTERVAL_S)
                continue

        for mp4 in sorted(videos_dir.glob("episode_*.mp4")):
            key = mp4.name
            if key in seen:
                continue
            frames = _read_first_frames(mp4, FRAMES_TO_AVERAGE)
            if frames is None:
                # Mid-write: mp4 has fewer than FRAMES_TO_AVERAGE decodable
                # frames right now. Skip — we'll retry next poll.
                continue
            seen.add(key)
            avg = _average(frames)
            pose, ok, msg = evaluate_pose(avg, reference)
            ep = _episode_index(mp4)
            if ok:
                _log(f"ep{ep}: {msg}")
                print(f"{GREEN}pose-watch ep{ep}: OK{RESET}", file=sys.stderr)
            else:
                _log(f"ep{ep}: {msg}")
                print(
                    f"{BELL}{RED}pose-watch ep{ep} DRIFT: "
                    f"roll={pose.roll_deg:+.2f}° tx={pose.tx_px:+.2f}px ty={pose.ty_px:+.2f}px"
                    f"{RESET}",
                    file=sys.stderr,
                )

        time.sleep(POLL_INTERVAL_S)

    _log("watcher stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
