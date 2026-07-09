"""
Assign scanned SKUs to box compartments.

The kit rule (from the operator): identical part numbers must land in the SAME
compartment; WHICH compartment is arbitrary. So this is a grouping + placement
problem, deterministic and explainable:

  - one compartment per unique SKU (never split a SKU across bins),
  - default strategy "count-size": the SKU with the most bags goes to the
    largest compartment (big bin for the big pile), ties broken by part number
    for stability,
  - abstained/unreadable packets are never assigned (they route to re-read).

Placement itself is downstream (needs the camera->robot transform); this module
only decides the target compartment per SKU and per packet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Assignment:
    part: str
    compartment: int          # compartment id (1..N)
    count: int                # bags of this SKU in the scene


class CompartmentAssigner:
    def __init__(self, compartments: list[dict]):
        # each: {"id", "mm_rect":[x,y,w,h], "img_poly", "img_center"}
        self.comps = list(compartments)

    @classmethod
    def from_file(cls, path: str | Path) -> "CompartmentAssigner":
        return cls(json.loads(Path(path).read_text())["compartments"])

    @staticmethod
    def _area(c: dict) -> float:
        _, _, w, h = c["mm_rect"]
        return float(w) * float(h)

    def assign(self, counts: dict[str, int], strategy: str = "count-size") -> list[Assignment]:
        skus = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if strategy == "count-size":
            comps = sorted(self.comps, key=self._area, reverse=True)
        else:  # "by-id": stable, ignore size
            comps = sorted(self.comps, key=lambda c: c["id"])
        if len(skus) > len(comps):
            raise ValueError(f"{len(skus)} SKUs but only {len(comps)} compartments — "
                             "kit does not fit this box")
        return [Assignment(part, comps[i]["id"], cnt) for i, (part, cnt) in enumerate(skus)]

    def mapping(self, counts: dict[str, int], **kw) -> dict[str, int]:
        """part -> compartment id."""
        return {a.part: a.compartment for a in self.assign(counts, **kw)}

    def comp_by_id(self, cid: int) -> dict | None:
        return next((c for c in self.comps if c["id"] == cid), None)
