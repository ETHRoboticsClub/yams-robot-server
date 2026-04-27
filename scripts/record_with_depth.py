"""Wrapper entry point around lerobot-record that also saves depth.

Installs two monkey-patches before delegating to lerobot.scripts.lerobot_record.main():

  1. BiYamsFollower.get_observation: after the normal observation is built,
     pull the depth snapshot from each RealSenseCameraCached with
     use_depth=True and stash it under a private __depth__.<cam_name> key
     in the observation dict.

  2. LeRobotDataset.add_frame: strip and divert any __depth__.* keys to a
     PNG-16 sidecar writer before upstream validation runs. LeRobotDataset
     never sees the uint16 depth arrays, so its add_frame stays on the
     RGB-only happy path. The sidecar writes to
         <dataset_root>/depth/observation.depth.<cam_name>/episode_NNNNNN/frame_NNNNNN.png

Also hooks save_episode / clear_episode_buffer so episode drop / re-record
keeps the depth side in sync with the parquet side.

Invoked by scripts/record.sh when RECORD_DEPTH=true. With depth disabled,
the monkey-patches are no-ops and behavior is identical to lerobot-record.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.scripts import lerobot_record as _lerobot_record_mod  # noqa: E402
from lerobot.scripts.lerobot_record import main as lerobot_record_main  # noqa: E402

from lerobot_camera_cached.camera_realsense_cached import (  # noqa: E402
    RealSenseCameraCached,
)
from lerobot_camera_cached.depth_sidecar import DepthSidecar  # noqa: E402
from lerobot_robot_yams.bi_follower import BiYamsFollower  # noqa: E402

logger = logging.getLogger(__name__)

DEPTH_KEY_PREFIX = "__depth__."
DEPTH_FEATURE_PREFIX = "observation.depth."

# Keyed by dataset root — a single process only ever records into one root,
# but keep the mapping explicit so the patches stay thread-safe.
_SIDECARS: dict[str, DepthSidecar] = {}


def _positive_int_env(name: str, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _sidecar_for(dataset: LeRobotDataset) -> DepthSidecar:
    key = str(dataset.root)
    if key not in _SIDECARS:
        downsample = _positive_int_env("DEPTH_DOWNSAMPLE", default=1, min_value=1)
        clip_max_mm = _positive_int_env("DEPTH_CLIP_MM", default=0, min_value=0)
        _SIDECARS[key] = DepthSidecar(
            dataset.root, downsample=downsample, clip_max_mm=clip_max_mm
        )
        logger.info(
            "depth sidecar: root=%s downsample=%d clip_max_mm=%d",
            key,
            downsample,
            clip_max_mm,
        )
    return _SIDECARS[key]


def _install_get_observation_patch() -> None:
    original = BiYamsFollower.get_observation

    def patched(self: BiYamsFollower, with_cameras: bool = True) -> dict[str, Any]:
        obs = original(self, with_cameras=with_cameras)
        if not with_cameras:
            return obs
        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseCameraCached) and cam.use_depth:
                depth = cam.pop_depth_snapshot()
                if depth is not None:
                    obs[DEPTH_KEY_PREFIX + cam_key] = depth
        return obs

    BiYamsFollower.get_observation = patched
    logger.info("depth patch: BiYamsFollower.get_observation hooked")


def _install_add_frame_patch() -> None:
    original = LeRobotDataset.add_frame

    def patched(self: LeRobotDataset, frame: dict[str, Any]):
        # Extract private depth entries before upstream validation. Only
        # depth keys prefixed with __depth__ get diverted; everything else
        # passes through unchanged.
        depth_entries: dict[str, Any] = {}
        for k in list(frame.keys()):
            if k.startswith(DEPTH_KEY_PREFIX):
                depth_entries[k[len(DEPTH_KEY_PREFIX):]] = frame.pop(k)

        original(self, frame)

        if not depth_entries:
            return

        episode_buffer = self.episode_buffer
        if episode_buffer is None:
            return  # pragma: no cover — upstream would have raised.
        episode_index = int(episode_buffer["episode_index"])
        frame_index = int(episode_buffer["size"] - 1)

        sidecar = _sidecar_for(self)
        for cam_name, depth_arr in depth_entries.items():
            feature_name = DEPTH_FEATURE_PREFIX + cam_name
            sidecar.write_frame(feature_name, episode_index, frame_index, depth_arr)

    LeRobotDataset.add_frame = patched
    logger.info("depth patch: LeRobotDataset.add_frame hooked")


def _install_clear_episode_buffer_patch() -> None:
    """When the operator re-records an episode (clear_episode_buffer fires
    before save), drop the depth frames we queued for that episode so the
    sidecar stays aligned with the parquet."""
    original = LeRobotDataset.clear_episode_buffer

    def patched(self: LeRobotDataset) -> None:
        buf = self.episode_buffer
        if buf is not None:
            episode_index = int(buf.get("episode_index", -1))
            if episode_index >= 0 and str(self.root) in _SIDECARS:
                sidecar = _SIDECARS[str(self.root)]
                root_depth_dir = sidecar.root / "depth"
                if root_depth_dir.is_dir():
                    for feature_dir in root_depth_dir.iterdir():
                        sidecar.drop_episode(feature_dir.name, episode_index)
        original(self)

    LeRobotDataset.clear_episode_buffer = patched
    logger.info("depth patch: LeRobotDataset.clear_episode_buffer hooked")


def _install_build_dataset_frame_patch() -> None:
    """record_loop filters observations through build_dataset_frame before
    calling dataset.add_frame, and that function drops any key not declared
    in dataset.features. Wrap it so __depth__.* keys bleed through to the
    add_frame patch. Shadows the name in lerobot.scripts.lerobot_record,
    since that module imported the function at load time."""
    original = _lerobot_record_mod.build_dataset_frame

    def patched(ds_features, values, prefix):
        frame = original(ds_features, values, prefix)
        for k, v in values.items():
            if isinstance(k, str) and k.startswith(DEPTH_KEY_PREFIX):
                frame[k] = v
        return frame

    _lerobot_record_mod.build_dataset_frame = patched
    logger.info("depth patch: lerobot_record.build_dataset_frame hooked")


def _install_patches() -> None:
    _install_get_observation_patch()
    _install_build_dataset_frame_patch()
    _install_add_frame_patch()
    _install_clear_episode_buffer_patch()


if __name__ == "__main__":
    _install_patches()
    sys.exit(lerobot_record_main() or 0)
