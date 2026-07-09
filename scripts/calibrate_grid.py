#!/usr/bin/env python3
"""
Grid calibration (HUMAN-IN-THE-LOOP, needs the robot).

For each cell you teleop the placing arm to two poses and we save the joints:
  1) HOVER  — just above the cell
  2) PLACE  — lowered into the cell (where the gripper opens)

Run teleop in another terminal (./scripts/teleop.sh), move the arm by hand, then
press ENTER here to capture each pose. Writes configs/grid.yaml.

    python scripts/calibrate_grid.py --rows 2 --cols 3 --arm right

No IK, no cuRobo: placing later just replays these joint vectors via slow_move.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorting.config import load_config          # noqa: E402
from sorting.grid import GridMap                 # noqa: E402
from sorting.robot_backend import RealBackend    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--arm", choices=["left", "right"], default="right")
    args = ap.parse_args()

    cfg = load_config()
    grid = GridMap(cfg.repo_root / cfg.grid.file)
    grid.arm = args.arm

    backend = RealBackend()
    backend.connect()
    print(f"\nCalibrating a {args.rows}x{args.cols} grid with the {args.arm} arm.")
    print("Move the arm by teleop, then press ENTER to capture.\n")

    try:
        for r in range(args.rows):
            for c in range(args.cols):
                input(f"cell {r},{c}  → move to HOVER, press ENTER…")
                hover = backend.get_joints(args.arm).tolist()
                input(f"cell {r},{c}  → move to PLACE, press ENTER…")
                place = backend.get_joints(args.arm).tolist()
                grid.set_cell((r, c), hover, place)
                grid.save()
                print(f"  saved cell {r},{c}\n")
    finally:
        backend.disconnect()

    print(f"Done → {grid.path}")


if __name__ == "__main__":
    main()
