import json
import time
from pathlib import Path
from typing import Any


def build_joint_label_map(section_config: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    joint_labels = section_config.get("joint_labels", {})
    for side in ("left", "right"):
        for joint, label in joint_labels.get(side, {}).items():
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
