"""
Place primitive: drop the held packet into a grid cell by replaying the cell's
saved joint waypoints. hover -> place -> open gripper -> retract to hover.

No inverse kinematics: the waypoints were captured by teleop during calibration.
Works the instant a cell is calibrated; runs harmlessly in mock mode before that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .grid import GridMap

log = logging.getLogger("sorting.place")


@dataclass
class PlaceResult:
    ok: bool
    reason: str = ""


class PlacePrimitive:
    def __init__(self, grid: GridMap, backend, move_duration: float = 2.5):
        self.grid = grid
        self.backend = backend
        self.move_duration = move_duration

    def execute(self, cell: tuple[int, int]) -> PlaceResult:
        wp = self.grid.waypoints(cell)
        arm = self.grid.arm

        if getattr(self.backend, "name", "mock") == "mock":
            log.info("[place] (mock) arm=%s cell=%s hover→place→open→retract", arm, cell)
            return PlaceResult(True, "mock")

        if not wp.calibrated:
            return PlaceResult(False, f"cell {cell} not calibrated")

        try:
            self.backend.slow_move_to(arm, wp.hover, self.move_duration)
            self.backend.slow_move_to(arm, wp.place, self.move_duration)
            self.backend.set_gripper(arm, self.grid.gripper_open)
            self.backend.slow_move_to(arm, wp.hover, self.move_duration)
            return PlaceResult(True, "ok")
        except Exception as e:  # real-hardware faults surface here, not silently
            log.exception("place failed")
            return PlaceResult(False, f"{type(e).__name__}: {e}")
