"""
Grid map: destination-box cell -> saved joint waypoints.

Because the rig is joint-space controlled and ships a slow_move that interpolates
joints, "calibrating the grid" means teleoperating the arm to each cell once and
saving the 14-DoF joint vector. Placing = replaying that vector. No IK, no cuRobo.

configs/grid.yaml schema:

    arm: right                    # which arm does the placing
    gripper_open: 1.0
    gripper_closed: 0.0
    cells:
      "0,0":
        hover: [ ...14 floats... ]  # above the cell
        place: [ ...14 floats... ]  # lowered into the cell
      "0,1": { ... }

Until calibrated, waypoints are null and PlacePrimitive runs only in mock mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CellWaypoints:
    cell: tuple[int, int]
    hover: list[float] | None
    place: list[float] | None

    @property
    def calibrated(self) -> bool:
        return self.hover is not None and self.place is not None


class GridMap:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.arm: str = "right"
        self.gripper_open: float = 1.0
        self.gripper_closed: float = 0.0
        self._cells: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = yaml.safe_load(self.path.read_text()) or {}
        self.arm = data.get("arm", self.arm)
        self.gripper_open = float(data.get("gripper_open", self.gripper_open))
        self.gripper_closed = float(data.get("gripper_closed", self.gripper_closed))
        self._cells = data.get("cells", {}) or {}

    @staticmethod
    def _key(cell: tuple[int, int]) -> str:
        return f"{cell[0]},{cell[1]}"

    def waypoints(self, cell: tuple[int, int]) -> CellWaypoints:
        raw = self._cells.get(self._key(cell), {}) or {}
        return CellWaypoints(cell=cell, hover=raw.get("hover"), place=raw.get("place"))

    def is_calibrated(self, cell: tuple[int, int]) -> bool:
        return self.waypoints(cell).calibrated

    def calibrated_cells(self) -> list[tuple[int, int]]:
        out = []
        for k, v in self._cells.items():
            if v and v.get("hover") and v.get("place"):
                r, c = k.split(",")
                out.append((int(r), int(c)))
        return sorted(out)

    # -- calibration write path ----------------------------------------

    def set_cell(self, cell: tuple[int, int], hover: list[float], place: list[float]) -> None:
        self._cells[self._key(cell)] = {
            "hover": [float(x) for x in hover],
            "place": [float(x) for x in place],
        }

    def save(self) -> None:
        data = {
            "arm": self.arm,
            "gripper_open": self.gripper_open,
            "gripper_closed": self.gripper_closed,
            "cells": self._cells,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(data, sort_keys=True))
