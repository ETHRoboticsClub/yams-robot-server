"""
The sorting loop. Perception → pick → read → decide → place, repeat until the
source box is empty. Order does not matter (greedy topmost-first), per spec.

Every step prints a legible line and pushes best-effort telemetry so failures are
attributable (perception vs read vs decide vs place). Telemetry is optional and
never raises into the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .decision import DecisionTable
from .place import PlacePrimitive
from .reader import BarcodeReader
from .telemetry import Telemetry

log = logging.getLogger("sorting.loop")


@dataclass
class RunStats:
    picked: int = 0
    placed: int = 0
    read_ok: int = 0
    read_fail: int = 0
    unknown_part: int = 0
    pick_fail: int = 0
    place_fail: int = 0
    placements: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


class Orchestrator:
    def __init__(self, perception, pick, reader: BarcodeReader,
                 decision: DecisionTable, place: PlacePrimitive,
                 telemetry: Telemetry, max_picks: int = 100,
                 max_read_attempts: int = 3):
        self.perception = perception
        self.pick = pick
        self.reader = reader
        self.decision = decision
        self.place = place
        self.tel = telemetry
        self.max_picks = max_picks
        self.max_read_attempts = max_read_attempts

    def run(self) -> RunStats:
        stats = RunStats()
        self.tel.update(state="running")
        self.tel.event("run started")

        for i in range(self.max_picks):
            det = self.perception.next_detection()
            if not det.present:
                self.tel.event("source box empty")
                break

            self.tel.update(state="picking", current={"step": i + 1})
            self.tel.set_frame(self.perception.overview_frame())

            pr = self.pick.execute(det)
            if not pr.ok:
                stats.pick_fail += 1
                self.tel.bump("errors")
                self.tel.event(f"pick failed: {pr.reason}")
                log.warning("pick failed: %s", pr.reason)
                continue
            stats.picked += 1
            self.tel.bump("picked")

            # read the label (with active re-imaging retries)
            res = None
            for attempt in range(self.max_read_attempts):
                frame = self.perception.reading_frame(det)
                self.tel.set_frame(frame)
                res = self.reader.read_part_number(frame)
                if res.ok:
                    break
            identity = res.identity if res else None

            if not identity:
                stats.read_fail += 1
                self.tel.bump("read_fail")
                self.tel.event("read failed after retries — skipping")
                log.warning("read failed after %d attempts", self.max_read_attempts)
                continue
            stats.read_ok += 1
            self.tel.bump("read_ok")

            dec = self.decision.decide(identity)
            self.tel.update(current={"step": i + 1, "identity": identity,
                                     "cell": list(dec.cell) if dec.cell else None,
                                     "method": res.method})
            if not dec.ok:
                stats.unknown_part += 1
                self.tel.event(f"{identity}: {dec.reason}")
                log.warning("decision: %s", dec.reason)
                continue

            print(f"  {identity}  ->  cell {dec.cell}  [{res.method}]")

            self.tel.update(state="placing")
            plr = self.place.execute(dec.cell)
            if not plr.ok:
                stats.place_fail += 1
                self.tel.bump("errors")
                self.tel.event(f"place failed: {plr.reason}")
                log.warning("place failed: %s", plr.reason)
                continue

            stats.placed += 1
            stats.placements.append({"identity": identity, "cell": list(dec.cell)})
            self.tel.mark_placed(dec.cell, identity)
            self.tel.event(f"placed {identity} -> {dec.cell}")

        self.tel.update(state="done", stats=self.tel._state.get("stats"))
        self.tel.event("run finished")
        return stats
