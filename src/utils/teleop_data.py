import json
import time
from pathlib import Path
from typing import Any

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


def load_task_names() -> list[str]:
    with open(TRAJECTORIES_CONFIG_PATH) as f:
        traj_config = yaml.safe_load(f)
    return traj_config.get('tasks', ['miscellaneous'])


def save_trajectory(traj: list[dict[str, Any]], task: str, logger) -> None:
    task_dir = TRAJECTORIES_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.isdigit())
    next_num = int(existing[-1].name) + 1 if existing else 0
    ep_dir = task_dir / str(next_num)
    ep_dir.mkdir()
    save_run_history(traj, time.time(), ep_dir, logger)
