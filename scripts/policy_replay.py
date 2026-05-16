"""Replay training-dataset frames through the mimic_video policy.

Standalone diagnostic — does not require the robot. Pulls observations from a
LeRobot HF dataset (default ETHRC/robot-learning-fs26 @ 30 Hz), runs them through
the cosmos worker at one or more `stop_video_denoising_step` values, and
compares the predicted action chunk to the dataset's ground-truth actions.

Frame alignment mirrors the training-time policy_io config
(model/cosmos_predict2/configs/dataloading/policy_io/bi_yams.yaml):

  obs.workspace_rgb:  horizon=5,  target_frequency=5 Hz  -> 5 frames at 30/5=6 stride
  obs.joint_state:    horizon=1,  target_frequency=10 Hz -> single frame at T
  action.joint_action:horizon=30, target_frequency=10 Hz -> predicted[i] vs ds.action[T + 3*i]

After the adapter's action_stride=2, strided[i] -> ds.action[T + 6*i].

Run from the yams venv:
  uv run python scripts/policy_replay.py --stop-steps 1 4 8 16 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mimic_adapter import MimicVideoConfig, _CosmosClient


DATASET_FPS = 30          # ETHRC/robot-learning-fs26 native rate
TRAIN_IMG_FPS = 5         # policy_io.obs.workspace_rgb target_frequency
TRAIN_ACTION_FPS = 10     # policy_io.action.joint_action target_frequency
IMG_STRIDE = DATASET_FPS // TRAIN_IMG_FPS      # 6
ACTION_STRIDE_DS = DATASET_FPS // TRAIN_ACTION_FPS  # 3


def build_image_history(ds: LeRobotDataset, indices: list[int], image_key: str) -> np.ndarray:
    frames = []
    for i in indices:
        img = ds[i][image_key]  # (C, H, W) float32 in [0, 1]
        x = (2.0 * img.float() - 1.0).numpy().astype(np.float32)
        frames.append(x[:, None, :, :])  # (C, 1, H, W)
    return np.concatenate(frames, axis=1)  # (C, 5, H, W)


def gt_action_chunk(ds: LeRobotDataset, T: int, action_key: str, horizon: int = 30) -> np.ndarray:
    out = []
    for i in range(horizon):
        idx = T + i * ACTION_STRIDE_DS
        if idx >= len(ds):
            break
        out.append(ds[idx][action_key].numpy().astype(np.float32))
    return np.stack(out, axis=0)  # (<=horizon, 14)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="ETHRC/robot-learning-fs26")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-test-frames", type=int, default=4)
    parser.add_argument("--stop-steps", type=int, nargs="+", required=True)
    parser.add_argument("--image-key", default="observation.images.topdown")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--warmup-step", type=int, default=None,
                        help="stop step for worker warmup; defaults to stop_steps[0]")
    args = parser.parse_args()

    print(f"[replay] loading {args.repo_id} (episode {args.episode}) ...", flush=True)
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode], download_videos=True, revision="main")
    print(f"[replay] dataset: {len(ds)} frames in episode {args.episode}", flush=True)
    print(f"[replay] features: {list(ds.features.keys())}", flush=True)

    # Task prompt from episode metadata.
    ep_meta = ds.meta.episodes[args.episode] if hasattr(ds.meta, "episodes") else None
    task = None
    if ep_meta is not None:
        for key in ("tasks", "task", "single_task"):
            v = ep_meta.get(key) if isinstance(ep_meta, dict) else None
            if v:
                task = v[0] if isinstance(v, (list, tuple)) else v
                break
    if task is None:
        task = "Pick up the item and place it in the box"
    print(f"[replay] task: {task!r}", flush=True)

    # Valid frame range: need (5-1)*6=24 frames before T for image history,
    # and 29*3=87 frames after T for the full 30-action ground-truth.
    valid_lo = 24
    valid_hi = len(ds) - 88
    if valid_hi <= valid_lo:
        raise SystemExit(f"episode too short: len={len(ds)}, need >= 113 frames")
    test_frames = np.linspace(valid_lo, valid_hi - 1, args.num_test_frames, dtype=int).tolist()
    print(f"[replay] testing local episode frames: {test_frames}", flush=True)

    warmup_step = args.warmup_step if args.warmup_step is not None else args.stop_steps[0]
    cfg = MimicVideoConfig(stop_video_denoising_step=warmup_step, task_prompt=task)
    print(f"[replay] spawning cosmos worker (warmup at stop_step={warmup_step}, ~1 min) ...", flush=True)
    client = _CosmosClient(cfg)
    print(f"[replay] worker ready", flush=True)

    try:
        # Precompute frame inputs and ground truth once per test frame.
        cases = []
        for T in test_frames:
            img_idx = [T - i * IMG_STRIDE for i in range(4, -1, -1)]  # [T-24,...,T]
            video = build_image_history(ds, img_idx, args.image_key)  # (C, 5, H, W)
            state = ds[T][args.state_key].numpy().astype(np.float32)  # (14,)
            gt = gt_action_chunk(ds, T, args.action_key, horizon=30)  # (30, 14)
            cases.append((T, video, state, gt))

        for stop in args.stop_steps:
            print(f"\n========== stop_step={stop} ==========", flush=True)
            mse_per_frame = []
            for T, video, state, gt in cases:
                predicted = client.infer(
                    video=video[None],            # (1, C, 5, H, W)
                    state=state[None, None],      # (1, 1, 14)
                    prompt=task,
                    num_sampling_step=cfg.num_sampling_steps,
                    stop_after_step=stop,
                    use_cuda_graphs=False,
                )
                pred = predicted[0]  # (30, 14)
                n = min(pred.shape[0], gt.shape[0])
                diff = pred[:n] - gt[:n]
                mse = float((diff ** 2).mean())
                mse_per_joint = (diff ** 2).mean(axis=0)
                # Baseline: "do nothing" — predicted action = current state, repeated.
                baseline = np.broadcast_to(state, gt[:n].shape)
                baseline_diff = baseline - gt[:n]
                baseline_mse = float((baseline_diff ** 2).mean())
                mse_per_frame.append(mse)
                print(f"  T={T:4d}  mse={mse:.5f}  baseline_mse={baseline_mse:.5f}  "
                      f"pred[0]={pred[0].round(3).tolist()}", flush=True)
                print(f"             gt[0]  ={gt[0].round(3).tolist()}", flush=True)
                print(f"             state  ={state.round(3).tolist()}", flush=True)
                print(f"             |Δpred[0]-gt[0]|={np.abs(pred[0] - gt[0]).round(3).tolist()}", flush=True)
                print(f"             per-joint mse: {mse_per_joint.round(4).tolist()}", flush=True)
            print(f"  mean mse across frames: {np.mean(mse_per_frame):.5f}", flush=True)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
