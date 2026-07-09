#!/usr/bin/env python3
"""Dry-run the scene scan on an image (or a live scan-cam grab).

  python scripts/scan_dryrun.py --image frame.jpg
  python scripts/scan_dryrun.py --live            # grabs /dev/video14 (exclusive)

Prints the inventory: accepted parts (+ qty, confidence, position) and anything
abstained/rejected. Known SKU set comes from configs/sorting.yaml parts_to_cell
if present, else identity is accepted on confidence alone.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from sorting.scan import ScanReader, grab_scan_frame

def load_known():
    try:
        import yaml
        with open("configs/sorting.yaml") as f:
            cfg = yaml.safe_load(f)
        return set((cfg.get("parts_to_cell") or {}).keys())
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.80)
    ap.add_argument("--no-known", action="store_true", help="ignore known-SKU gate")
    a = ap.parse_args()

    img = grab_scan_frame() if a.live else cv2.imread(a.image)
    if img is None:
        print("no image (grab failed or bad path)"); sys.exit(1)

    known = None if a.no_known else load_known()
    r = ScanReader(known_parts=known, min_conf=a.min_conf).read_inventory(img)

    print(f"scene: {img.shape[1]}x{img.shape[0]}  known_set={sorted(known) if known else 'off'}")
    print(f"accepted {len(r.accepted)} / {len(r.items)} detections\n")
    for i in r.items:
        tag = "OK " if i.accepted else "ABSTAIN"
        q = f"x{i.qty}" if i.qty else "x?"
        print(f"  [{tag}] {str(i.part):<16} {q:<4} conf={i.conf:<5} @{tuple(i.center)} {i.reason}")
    print("\ncounts:", json.dumps(r.counts()))

if __name__ == "__main__":
    main()
