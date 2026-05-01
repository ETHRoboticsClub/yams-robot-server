"""Thin wrapper around lerobot's teleoperate that reads `configs/arms.yaml`.

Why this exists: lerobot-teleoperate requires ~10 long CLI flags (CAN ports,
leader ports, cameras JSON). This wrapper reads the YAML once and builds the
dataclasses directly — no sys.argv mutation required.

Usage:
    uv run ethrc-teleoperate                  # display off, fps=250
    uv run ethrc-teleoperate --display-data   # spawn Rerun viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from lerobot.cameras.configs import CameraConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_robot_yams.bi_follower import BiYamsFollowerConfig
from lerobot_teleoperator_gello.bi_leader import BiYamsLeaderConfig
from utils.lerobot_record_wrapper import run_with_graceful_stop

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arms.yaml"


def _parse_cameras(cameras_configs: dict) -> dict[str, CameraConfig]:
    result = {}
    for name, entry in cameras_configs.items():
        entry = dict(entry)
        cam_type = entry.pop("type")
        cls = CameraConfig.get_choice_class(cam_type)
        result[name] = cls(**entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETHRC teleop: lerobot teleoperate driven by configs/arms.yaml.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Spawn a Rerun viewer streaming camera frames + joint state live.",
    )
    parser.add_argument("--fps", type=int, default=250)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())

    cameras = _parse_cameras(raw["cameras"]["configs"])
    robot_cfg = BiYamsFollowerConfig(
        left_arm_can_port=raw["follower"]["left_arm"]["can_port"],
        right_arm_can_port=raw["follower"]["right_arm"]["can_port"],
        cameras=cameras,
    )
    teleop_cfg = BiYamsLeaderConfig(
        left_arm_port=raw["leader"]["left_arm"]["port"],
        right_arm_port=raw["leader"]["right_arm"]["port"],
    )
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=args.fps,
        display_data=args.display_data,
    )

    register_third_party_plugins()
    raise SystemExit(run_with_graceful_stop(lambda: teleoperate(cfg)))


if __name__ == "__main__":
    main()
