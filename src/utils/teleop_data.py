import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"
TRAJECTORIES_CONFIG_PATH = PROJECT_ROOT / "configs" / "trajectories.yaml"


def build_joint_label_map(section_config: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for side in ("left", "right"):
        arm_config = section_config.get(f"{side}_arm", {})
        for joint, cfg in arm_config.get("motors", {}).items():
            label = cfg.get("label") if isinstance(cfg, dict) else None
            if label:
                out[f"{side}_{joint}.pos"] = label
    return out


def joint_only(data: dict[str, Any] | None) -> dict[str, float]:
    if not data:
        return {}
    return {
        k: float(v)
        for k, v in data.items()
        if k.endswith(".pos") and k.startswith(("left_", "right_"))
    }


def cameras_only(data: dict[str, Any] | None) -> dict[str, np.ndarray]:
    if not data:
        return {}
    return {
        k: v for k, v in data.items()
        if isinstance(v, np.ndarray) and v.ndim == 3
    }


def save_run_history(
    trajectory: list[dict[str, Any]],
    run_started_at: float,
    run_history_dir: Path,
    logger,
) -> None:
    if not trajectory:
        return
    run_history_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(run_started_at))
    out_path = run_history_dir / f"trajectory_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in trajectory:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    logger.info("Saved trajectory: %s", out_path)


def load_task_config() -> list[dict]:
    with open(TRAJECTORIES_CONFIG_PATH) as f:
        traj_config = yaml.safe_load(f)
    tasks = traj_config.get('tasks', ['miscellaneous'])
    return [t if isinstance(t, dict) else {'name': t, 'goal': None} for t in tasks]


def load_task_names() -> list[str]:
    return [t['name'] for t in load_task_config()]


def save_trajectory(traj: list[dict[str, Any]], task: str, logger) -> Path:
    task_dir = TRAJECTORIES_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.isdigit())
    next_num = int(existing[-1].name) + 1 if existing else 0
    ep_dir = task_dir / str(next_num)
    ep_dir.mkdir()

    # Save camera frames as JPEGs, grouped by camera key
    cam_dirs: dict[str, Path] = {}
    for i, row in enumerate(traj):
        for cam_key, frame in row.pop("cams", {}).items():
            if cam_key not in cam_dirs:
                cam_dirs[cam_key] = ep_dir / cam_key
                cam_dirs[cam_key].mkdir()
            cv2.imwrite(str(cam_dirs[cam_key] / f"{i:06d}.jpg"), frame)

    save_run_history(traj, time.time(), ep_dir, logger)
    # Write initial metadata
    meta_path = ep_dir / "metadata.yaml"
    meta_path.write_text(yaml.dump({
        'marked_bad': False,
        'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }), encoding="utf-8")
    return ep_dir


def get_trajectory_metadata(ep_dir: Path) -> dict:
    meta_path = ep_dir / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def set_trajectory_marked_bad(task: str, episode: str, bad: bool) -> None:
    ep_dir = TRAJECTORIES_DIR / task / episode
    meta_path = ep_dir / "metadata.yaml"
    meta = get_trajectory_metadata(ep_dir)
    meta["marked_bad"] = bad
    meta_path.write_text(yaml.dump(meta), encoding="utf-8")
