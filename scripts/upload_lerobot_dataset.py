#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert trajectory jsonl files to LeRobot and upload to HF")
    p.add_argument("data_path", type=Path, help="Folder with trajectory*.jsonl files")
    p.add_argument("repo_id", help="HF dataset repo id, e.g. JessieLoki/testDataSet")
    p.add_argument("--fps", type=int, default=200)
    p.add_argument("--task", default="teleop")
    p.add_argument("--robot-type", default="yams")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--private", action="store_true")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


def find_files(data_path: Path) -> list[Path]:
    files = sorted(data_path.glob("trajectory*.jsonl"))
    return files or sorted(data_path.glob("*.jsonl"))


def read_joint_keys(files: list[Path]) -> list[str]:
    keys = set()
    for path in files:
        for line in path.read_text().splitlines():
            if not line:
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

    # Verify remote parquet integrity and auto-reupload any corrupted files once.
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
    files = find_files(args.data_path)
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {args.data_path}")

    joints = read_joint_keys(files)
    if not joints:
        raise ValueError("No obs/act joints found in input data")

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
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            obs = row.get("obs", {})
            act = row.get("act", {})
            dataset.add_frame(
                {
                    "observation.state": np.asarray([obs.get(k, 0.0) for k in joints], dtype=np.float32),
                    "action": np.asarray([act.get(k, 0.0) for k in joints], dtype=np.float32),
                    "task": args.task,
                }
            )
            episode_frames += 1
            total_frames += 1
        if episode_frames:
            dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(f"Built {args.repo_id}: {len(files)} episodes, {total_frames} frames, {len(joints)} joints")

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
