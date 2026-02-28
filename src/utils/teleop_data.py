import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def copy_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}


def new_recording_state() -> dict[str, Any]:
    return {"is_active": False, "started_at": None, "rows": []}


def start_recording(recording_state: dict[str, Any]) -> None:
    recording_state["is_active"] = True
    recording_state["started_at"] = time.time()
    recording_state["rows"] = []


def stop_recording(recording_state: dict[str, Any], trajectories_dir: Path, logger) -> None:
    if recording_state["rows"]:
        save_trajectory(
            recording_state["rows"],
            recording_state["started_at"] or time.time(),
            trajectories_dir,
            logger,
        )
    recording_state["is_active"] = False
    recording_state["started_at"] = None
    recording_state["rows"] = []


def append_recording_sample(
    recording_state: dict[str, Any],
    t: float,
    observation: dict[str, Any],
    action: dict[str, Any],
) -> None:
    if not recording_state["is_active"]:
        return
    recording_state["rows"].append({"t": t, "obs": copy_observation(observation), "act": dict(action)})


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


def camera_only(data: dict[str, Any] | None) -> dict[str, np.ndarray]:
    if not data:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 3}


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_trajectory(
    trajectory: list[dict[str, Any]],
    trajectory_started_at: float,
    trajectories_dir: Path,
    logger,
) -> None:
    if not trajectory:
        return

    trajectories_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(trajectory_started_at))
    ms = int((trajectory_started_at % 1) * 1000)
    out_dir = trajectories_dir / f"trajectory_{ts}_{ms:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    camera_dirs: dict[str, Path] = {}
    out_path = out_dir / "trajectory.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(trajectory):
            obs = row.get("obs")
            cams = {}
            for cam_name, frame in camera_only(obs).items():
                safe_name = _safe_name(cam_name)
                if safe_name not in camera_dirs:
                    camera_dirs[safe_name] = frames_dir / safe_name
                    camera_dirs[safe_name].mkdir(parents=True, exist_ok=True)
                rel_path = Path("frames") / safe_name / f"{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / rel_path), _to_uint8(frame))
                cams[cam_name] = rel_path.as_posix()

            output_row = {
                "t": row["t"],
                "obs": joint_only(obs),
                "act": joint_only(row.get("act")),
                "cams": cams,
            }
            f.write(json.dumps(output_row, separators=(",", ":")) + "\n")

    logger.info("Saved trajectory: %s", out_path)
