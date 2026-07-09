#!/usr/bin/env python3
"""
M0 dry-run: prove the "brain" with no robot.

    # decode a saved label image and print identity -> cell
    python scripts/perception_dryrun.py --image tests/sample-label.jpg

    # live from the top-down RealSense (segment topmost packet, read, decide)
    python scripts/perception_dryrun.py --live

Exercises the REAL reader + decision table. This is the cheap, high-uncertainty
part of the pipeline; if this reads reliably, the concept is de-risked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorting.config import load_config          # noqa: E402
from sorting.decision import DecisionTable       # noqa: E402
from sorting.reader import BarcodeReader         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="tests/sample-label.jpg")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    reader = BarcodeReader(part_pattern=cfg.reading.part_pattern,
                           upscale_retries=cfg.reading.upscale_retries)
    decision = DecisionTable(cfg.parts_to_cell)

    import cv2
    if args.live:
        from sorting.perception import RealSensePerception
        per = RealSensePerception()
        print("acquiring from RealSense… (Ctrl-C to stop)")
        try:
            while True:
                color, depth = per.acquire()
                det = per.detect_topmost(color, depth)
                if not det.present:
                    print("  no packet in view")
                    continue
                res = reader.read_part_number(det.label_frame)
                dec = decision.decide(res.identity)
                print(f"  read={res.identity!r} [{res.method}]  ->  "
                      f"cell {dec.cell}  ({dec.reason})")
        except KeyboardInterrupt:
            per.close()
        return

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    res = reader.read_part_number(img)
    print(f"\n  image      : {args.image}")
    print(f"  codes      : {[c['text'] for c in res.codes]}")
    print(f"  identity   : {res.identity!r}  [{res.method}]")
    dec = decision.decide(res.identity)
    print(f"  decision   : cell {dec.cell}   ({dec.reason})\n")
    sys.exit(0 if dec.ok else 1)


if __name__ == "__main__":
    main()
