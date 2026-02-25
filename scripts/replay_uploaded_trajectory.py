#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay one episode from a LeRobot dataset on YAMS follower arms")
    p.add_argument("repo_id", help="HF dataset repo id, e.g. JessieLoki/testDataSet")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--root", type=Path, default=None, help="Local dataset root override")
    p.add_argument("--revision", default="main")
    p.add_argument("--force-cache-sync", action="store_true")
    p.add_argument("--arms-config", type=Path, default=Path("configs/arms.yaml"))
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--dry-run", action="store_true", help="Print actions without sending to robot")
    return p.parse_args()


def make_follower(arms_config_path: Path):
    import yaml
    from lerobot_robot_yams.bi_follower import BiYamsFollower, BiYamsFollowerConfig
    from utils.lifecycle import run_pre_setup

    arms = yaml.safe_load(arms_config_path.read_text())
    left_port = arms["follower"]["left_arm"]["server_port"]
    right_port = arms["follower"]["right_arm"]["server_port"]
    run_pre_setup(left_port, right_port)
    follower = BiYamsFollower(
        BiYamsFollowerConfig(
            left_arm_server_port=left_port,
            right_arm_server_port=right_port,
            cameras={},
        )
    )
    follower.connect()
    return follower


def main() -> None:
    args = parse_args()
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=False,
    )
    action_names = ds.features["action"]["names"]

    episodes = ds.meta.episodes
    row = next(ep for ep in episodes if ep["episode_index"] == args.episode_index)
    start = row["dataset_from_index"]
    end = row["dataset_to_index"]
    dt = (1.0 / ds.fps) / args.speed

    follower = None
    if not args.dry_run:
        follower = make_follower(args.arms_config)

    try:
        for i in range(start, end):
            frame = ds[i]
            action = frame["action"].detach().cpu().numpy()
            cmd = {k: float(v) for k, v in zip(action_names, action)}
            if args.dry_run:
                if i == start or (i - start) % 100 == 0:
                    print(f"frame {i-start}/{end-start}: {cmd}")
            else:
                follower.send_action(cmd)
            time.sleep(dt)
    finally:
        if follower is not None:
            follower.disconnect()


if __name__ == "__main__":
    main()
