#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert run_history JSONL trajectories to a LeRobot dataset")
    p.add_argument("repo_id", help="HF dataset repo id, e.g. yourname/my_dataset")
    p.add_argument("--run-history-dir", type=Path, default=Path("run_history"))
    p.add_argument("--fps", type=int, default=200)
    p.add_argument("--robot-type", default="yams")
    p.add_argument("--task", default="teleop")
    p.add_argument("--root", type=Path, default=None, help="Local LeRobot dataset root (optional)")
    p.add_argument("--overwrite", action="store_true", help="Delete existing local dataset root before create")
    p.add_argument("--private", action="store_true", help="Push dataset privately (if push supports it)")
    p.add_argument("--no-push", action="store_true", help="Build dataset locally only")
    return p.parse_args()


def read_joint_keys(files: list[Path]) -> list[str]:
    keys: set[str] = set()
    for path in files:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                keys.update(row.get("obs", {}).keys())
                keys.update(row.get("act", {}).keys())
    return sorted(keys)


def is_parquet_file(path: Path) -> bool:
    b = path.read_bytes()
    return len(b) >= 8 and b[:4] == b"PAR1" and b[-4:] == b"PAR1"


def push_dataset(root: Path, repo_id: str, private: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    files = sorted(p for p in root.rglob("*") if p.is_file())
    parquet_rel = [str(p.relative_to(root)) for p in files if p.suffix == ".parquet"]
    for p in files:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=str(p.relative_to(root)),
            repo_id=repo_id,
            repo_type="dataset",
        )

    info = json.loads((root / "meta" / "info.json").read_text())
    try:
        api.create_tag(repo_id, tag=info["codebase_version"], repo_type="dataset")
    except Exception:
        pass

    bad = []
    for rel in parquet_rel:
        cached = Path(
            hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision="main", force_download=True)
        )
        if not is_parquet_file(cached):
            bad.append(rel)
    for rel in bad:
        api.upload_file(
            path_or_fileobj=str(root / rel),
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="dataset",
        )
    for rel in bad:
        cached = Path(
            hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision="main", force_download=True)
        )
        if not is_parquet_file(cached):
            raise RuntimeError(f"Remote parquet is still invalid after reupload: {rel}")


def main() -> None:
    args = parse_args()
    files = sorted(args.run_history_dir.glob("trajectory_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No trajectory_*.jsonl found in {args.run_history_dir}")

    joints = read_joint_keys(files)
    if not joints:
        raise ValueError("No obs/act joint keys found in run history")

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
        "action": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    }

    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "robot_type": args.robot_type,
        "features": features,
    }
    if args.root is not None:
        if args.overwrite and args.root.exists():
            shutil.rmtree(args.root)
        create_kwargs["root"] = args.root
    if "use_videos" in inspect.signature(LeRobotDataset.create).parameters:
        create_kwargs["use_videos"] = False
    dataset = LeRobotDataset.create(**create_kwargs)

    total_frames = 0
    for path in files:
        episode_frames = 0
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                obs = row.get("obs", {})
                act = row.get("act", {})
                frame = {
                    "observation.state": torch.from_numpy(
                        np.asarray([obs.get(k, 0.0) for k in joints], dtype=np.float32)
                    ),
                    "action": torch.from_numpy(np.asarray([act.get(k, 0.0) for k in joints], dtype=np.float32)),
                    "task": args.task,
                }
                dataset.add_frame(frame)
                episode_frames += 1
                total_frames += 1
        if episode_frames > 0:
            dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(
        f"Built LeRobot dataset from {len(files)} trajectories, "
        f"{total_frames} frames, {len(joints)} joints for {args.repo_id}"
    )

    if args.no_push:
        print("Skipping push (--no-push)")
        return

    for p in (Path(dataset.root) / "data").rglob("*.parquet"):
        if not is_parquet_file(p):
            raise RuntimeError(f"Local parquet is invalid: {p}")
    for p in (Path(dataset.root) / "meta").rglob("*.parquet"):
        if not is_parquet_file(p):
            raise RuntimeError(f"Local parquet is invalid: {p}")
    push_dataset(Path(dataset.root), args.repo_id, args.private)
    print(f"Pushed to hf.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
