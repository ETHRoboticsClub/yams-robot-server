# Sorting pipeline — Bühler messy-box → sorted-grid kit

MVP for the Robot Learning project, built on this YAMS/LeRobot rig. Take Bühler
packets out of a cluttered source box, read each label, drop it in the right
compartment of a gridded destination box. Order does not matter — we prove the
packets get sorted to the right cells.

**Approach B (Hybrid), CEO-reviewed.** The learned policy does exactly one thing:
pick a floppy packet out of clutter. Everything else is classical and printable:

```
 RealSense topdown ─► segment topmost packet ─► ACT pick (learned) ─┐
                                                                     │  reading-station pose
   place at cell (saved joint waypoints) ◄─ dict lookup ◄─ zxing-cpp decode ◄─┘
```

The "which cell" decision is a plain `dict`, never a neural net. Identity comes
from a checksummed **barcode**, not OCR of wrinkled text. Place replays a
teleop-captured joint vector — **no IK, no cuRobo**.

## Quickstart (no robot, no trained policy)

```bash
cd ~/Desktop/yams-sorting-robot

# end-to-end demo with the live dashboard (synthetic packets, REAL barcode reader)
./scripts/sort.sh
# open http://localhost:8770

# prove the reader on the real Bühler label
PYTHONPATH=. uv run python scripts/perception_dryrun.py --image tests/sample-label.jpg

# tests
PYTHONPATH=. uv run python tests/test_reader.py
PYTHONPATH=. uv run python tests/test_pipeline.py
```

The demo generates real Code128 labels and decodes them with the real reader, so
`read → decide → place` is genuinely exercised. The dashboard shows the current
part, its destination cell, and a ✓ on each cell as it fills.

## Modes

| Mode | Perception | Pick | Robot | Use |
|---|---|---|---|---|
| `demo` | synthetic packets (real barcodes) | mock | none | show the concept + dashboard |
| `mock` | one label image | mock | none | test the reader/decision on a real photo |
| `real` | RealSense depth segmentation | ACT policy | live | the actual cell (needs M1+M2) |

```bash
MODE=demo ./scripts/sort.sh
MODE=mock ./scripts/sort.sh
MODE=real ./scripts/sort.sh      # after calibration + a trained policy
```

## The one config that matters

`configs/sorting.yaml` → `parts_to_cell`. This is the kit. Add a SKU or remap a
cell by editing this map. No retraining.

## What still needs the human + robot

The software runs today in demo/mock. To run for real:

1. **Reading camera ≥720p.** Measured decode floor: a Code128 will not decode
   below ~800 px across the label. Bump the reading camera in `configs/arms.yaml`
   to ≥1280×720 (prefer 1080p). `640×480` reads nothing.
2. **Re-aim the top-down camera** to frame source box + black table + grid, then
   `align_topdown.py` / `save_topdown_reference.py`.
3. **Calibrate the grid:** `python scripts/calibrate_grid.py` — teleop the arm to
   each cell's hover + place pose; joints are saved to `configs/grid.yaml`.
4. **Record ~50 pick demos + train ACT** (reuses `record.sh`), then set
   `pick_mode: policy` and `policy_path` in `configs/sorting.yaml`.

## Layout

```
sorting/
  config.py         one YAML → dataclasses
  perception.py     Mock / Sim(real barcodes) / RealSense(depth segmentation)
  pick.py           MockPick / PolicyPick (ACT)   ← the only learned part
  reader.py         zxing-cpp barcode + optional PaddleOCR fallback
  decision.py       identity → cell (dict + parts-list validation)
  grid.py           cell → saved joint waypoints
  place.py          replay hover→place→open→retract (no IK)
  robot_backend.py  Mock / Real (portal joint servers)
  telemetry.py      best-effort status writer (never raises into the loop)
  orchestrator.py   the loop
  run.py            CLI
dashboard/          stdlib-only web view (separate process, optional)
scripts/            sort.sh, perception_dryrun.py, calibrate_grid.py
tests/              reader + end-to-end dry-run
configs/            sorting.yaml, grid.yaml
```

## Why the dashboard can't break the pipeline

The sorter writes status to `outputs/sorting/status.json` (+ `frame.jpg`) inside a
swallow-all `try/except`, atomically. The dashboard is a separate process that
only reads those files. Turn it off, kill it, corrupt the file — the robot keeps
sorting. Visual feedback only.
