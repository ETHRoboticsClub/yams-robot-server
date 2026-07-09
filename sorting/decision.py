"""
The "which cell" brain — a pure dict lookup with parts-list validation.

Deliberately NOT a neural net. Adding a SKU or remapping cells is a config edit,
never a retrain. A decoded identity that is not on the known parts list is
rejected (returns None with a reason) so a mis-decode can never silently place a
packet in the wrong cell.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    cell: tuple[int, int] | None
    identity: str
    ok: bool
    reason: str = ""


class DecisionTable:
    def __init__(self, parts_to_cell: dict[str, list[int]]):
        self._map: dict[str, tuple[int, int]] = {
            str(k): (int(v[0]), int(v[1])) for k, v in parts_to_cell.items()
        }

    @property
    def known_parts(self) -> set[str]:
        return set(self._map)

    def cells_in_use(self) -> list[tuple[int, int]]:
        return sorted(set(self._map.values()))

    def decide(self, identity: str | None) -> Decision:
        if not identity:
            return Decision(None, identity or "", False, "no identity read")
        cell = self._map.get(identity)
        if cell is None:
            return Decision(None, identity, False, f"unknown part '{identity}'")
        return Decision(cell, identity, True, "ok")
