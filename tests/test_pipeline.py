"""End-to-end dry-run: SimPerception (real barcodes) → real reader → decision →
mock place. Asserts every packet lands in its configured cell with no errors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorting.config import load_config            # noqa: E402
from sorting.decision import DecisionTable         # noqa: E402
from sorting.grid import GridMap                   # noqa: E402
from sorting.orchestrator import Orchestrator      # noqa: E402
from sorting.perception import SimPerception       # noqa: E402
from sorting.pick import MockPick                  # noqa: E402
from sorting.place import PlacePrimitive           # noqa: E402
from sorting.reader import BarcodeReader           # noqa: E402
from sorting.telemetry import Telemetry            # noqa: E402


class _MockHandle:
    name = "mock"


def test_demo_end_to_end(tmp_path=None):
    cfg = load_config()
    parts = list(cfg.parts_to_cell)
    names = cfg.part_names
    tel = Telemetry(Path("/tmp/sorting-test-tel"), enabled=True)

    per = SimPerception(parts, names, seed=3)
    reader = BarcodeReader(part_pattern=cfg.reading.part_pattern)
    decision = DecisionTable(cfg.parts_to_cell)
    grid = GridMap(cfg.repo_root / cfg.grid.file)
    place = PlacePrimitive(grid, _MockHandle())

    orch = Orchestrator(per, MockPick(), reader, decision, place, tel,
                        max_picks=50, max_read_attempts=cfg.reading.max_read_attempts)
    stats = orch.run()

    assert stats.placed == len(parts), (stats.placed, len(parts))
    assert stats.read_fail == 0 and stats.pick_fail == 0 and stats.place_fail == 0
    assert stats.unknown_part == 0
    # every placement matches the configured cell
    for pl in stats.placements:
        assert cfg.parts_to_cell[pl["identity"]] == pl["cell"], pl


if __name__ == "__main__":
    test_demo_end_to_end()
    print("pipeline test: OK")
