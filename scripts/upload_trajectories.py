#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload local trajectories to a Hugging Face dataset repo")
    p.add_argument("local_dir", type=Path, help="Local trajectory/dataset directory to upload")
    p.add_argument("repo_id", help="Dataset repo id, e.g. yourname/my_dataset")
    p.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", help="Create repo as private if it does not exist")
    p.add_argument("--revision", default="main")
    p.add_argument("--path-in-repo", default=".", help="Target subfolder in remote repo")
    p.add_argument("--message", default="Upload trajectories")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(args.local_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.message,
    )
    print(f"Uploaded {args.local_dir} -> hf.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
