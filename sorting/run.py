"""
CLI entrypoint for the sorting pipeline.

    python -m sorting.run --mode demo          # synthetic packets, real reader, mock robot
    python -m sorting.run --mode mock --image tests/sample-label.jpg
    python -m sorting.run --mode real          # RealSense + robot + ACT policy (M2+)

--dashboard also launches the optional web dashboard in a background thread.
The pipeline runs identically with the dashboard off.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import load_config
from .decision import DecisionTable
from .grid import GridMap
from .orchestrator import Orchestrator
from .place import PlacePrimitive
from .reader import BarcodeReader
from .telemetry import Telemetry

DEMO_PARTS = {
    "MDDY14614001": "Auffangblech",
    "MDKR20871003": "Lagerdeckel",
    "MDGH11902005": "Spannhuelse",
    "MDPT30455002": "Dichtring",
    "MDVS64120007": "Fuehrungsbolzen",
    "MDZK88031004": "Klemmleiste",
}


def _demo_parts_to_cell(parts: list[str], cols: int = 3) -> dict[str, list[int]]:
    return {p: [i // cols, i % cols] for i, p in enumerate(parts)}


def _publish_plan(tel, cfg, p2c: dict, names: dict) -> None:
    """Push the kit plan to telemetry so the dashboard can label empty cells."""
    plan = {f"{v[0]},{v[1]}": k for k, v in p2c.items()}
    rows = max([v[0] for v in p2c.values()] + [cfg.grid.rows - 1]) + 1
    cols = max([v[1] for v in p2c.values()] + [cfg.grid.cols - 1]) + 1
    tel.update(plan=plan, part_names=names, grid={"rows": rows, "cols": cols})


def build(cfg, args):
    reader = BarcodeReader(part_pattern=cfg.reading.part_pattern,
                           upscale_retries=cfg.reading.upscale_retries,
                           use_ocr=("ocr" in cfg.reading.method))
    tel = Telemetry(cfg.repo_root / cfg.telemetry.dir, enabled=cfg.telemetry.enabled)

    if args.mode == "demo":
        from .perception import SimPerception
        from .pick import MockPick
        parts = list(cfg.parts_to_cell) or list(DEMO_PARTS)
        names = cfg.part_names or DEMO_PARTS
        p2c = cfg.parts_to_cell or _demo_parts_to_cell(parts, cfg.grid.cols)
        perception = SimPerception(parts, names)
        pick = MockPick()
        decision = DecisionTable(p2c)
        grid = GridMap(cfg.repo_root / cfg.grid.file)
        place = PlacePrimitive(grid, _MockBackendHandle())
        _publish_plan(tel, cfg, p2c, names)
        return perception, pick, reader, decision, place, tel

    if args.mode == "mock":
        import cv2
        from .perception import MockPerception, render_label
        from .pick import MockPick
        if args.image and Path(args.image).exists():
            img = cv2.imread(str(args.image))
        else:
            first = next(iter(cfg.parts_to_cell), "MDDY14614001")
            img = render_label(first, cfg.part_names.get(first, ""))
        perception = MockPerception(img, count=args.max_picks)
        pick = MockPick()
        p2c = cfg.parts_to_cell or _demo_parts_to_cell(list(DEMO_PARTS))
        decision = DecisionTable(p2c)
        grid = GridMap(cfg.repo_root / cfg.grid.file)
        place = PlacePrimitive(grid, _MockBackendHandle())
        _publish_plan(tel, cfg, p2c, cfg.part_names or DEMO_PARTS)
        return perception, pick, reader, decision, place, tel

    if args.mode == "real":
        from .perception import RealSensePerception
        from .pick import make_pick
        from .robot_backend import make_backend
        _warn_camera(cfg)
        perception = RealSensePerception()
        backend = make_backend("real")
        backend.connect()
        pick = make_pick(cfg.pick_mode, robot=None, policy_path=cfg.policy_path)
        decision = DecisionTable(cfg.parts_to_cell)
        grid = GridMap(cfg.repo_root / cfg.grid.file)
        place = PlacePrimitive(grid, backend)
        _publish_plan(tel, cfg, cfg.parts_to_cell, cfg.part_names)
        return perception, pick, reader, decision, place, tel

    raise SystemExit(f"unknown mode {args.mode}")


class _MockBackendHandle:
    name = "mock"


def _warn_camera(cfg) -> None:
    if cfg.camera.reading_width < cfg.camera.min_reading_width:
        logging.getLogger("sorting").warning(
            "reading camera width %d < measured decode floor %d — barcodes may "
            "not decode. Bump reading_width in configs/sorting.yaml.",
            cfg.camera.reading_width, cfg.camera.min_reading_width)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bühler sorting pipeline")
    ap.add_argument("--mode", choices=["demo", "mock", "real"], default="demo")
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None, help="label image for --mode mock")
    ap.add_argument("--max-picks", type=int, default=100)
    ap.add_argument("--step-delay", type=float, default=0.0,
                    help="seconds to pause between picks (nice for the dashboard)")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--dashboard-port", type=int, default=8770)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")
    cfg = load_config(args.config)

    if args.dashboard:
        _start_dashboard(cfg, args.dashboard_port)

    perception, pick, reader, decision, place, tel = build(cfg, args)

    # optional pacing so the demo is watchable on the dashboard
    if args.step_delay > 0:
        perception = _Paced(perception, args.step_delay)

    print(f"\n=== sorting run · mode={args.mode} · backend={cfg.backend} ===")
    orch = Orchestrator(perception, pick, reader, decision, place, tel,
                        max_picks=args.max_picks,
                        max_read_attempts=cfg.reading.max_read_attempts)
    stats = orch.run()

    print("\n--- summary ---")
    for k, v in stats.as_dict().items():
        if k != "placements":
            print(f"  {k:14s} {v}")
    print(f"  cells filled   {len(stats.placements)}")
    if args.dashboard:
        print(f"\nDashboard: http://localhost:{args.dashboard_port}  (Ctrl-C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


class _Paced:
    """Wrap a perception to sleep between detections (demo pacing only)."""

    def __init__(self, inner, delay):
        self._inner = inner
        self._delay = delay
        self._first = True

    def next_detection(self):
        if not self._first:
            time.sleep(self._delay)
        self._first = False
        return self._inner.next_detection()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _start_dashboard(cfg, port):
    import threading
    from dashboard.server import serve
    status_dir = cfg.repo_root / cfg.telemetry.dir
    t = threading.Thread(target=serve, args=(status_dir, port), daemon=True)
    t.start()
    time.sleep(0.3)


if __name__ == "__main__":
    sys.exit(main())
