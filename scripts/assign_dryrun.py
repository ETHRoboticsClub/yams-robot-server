#!/usr/bin/env python3
"""Kitting-plan dry run: scan the mat, assign SKUs to compartments, draw the plan.

  python scripts/assign_dryrun.py --scan tests/scan-live.jpg

Prints the plan table (SKU -> compartment, grouped so identical parts share a
bin) and writes an overlay tagging each detected packet with its target
compartment. No robot needed.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from sorting.scan import ScanReader
from sorting.assign import CompartmentAssigner


def load_known():
    try:
        import yaml
        cfg = yaml.safe_load(open("configs/sorting.yaml"))
        return set((cfg.get("parts_to_cell") or {}).keys())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--box", default="configs/box_compartments.json")
    ap.add_argument("--strategy", default="count-size")
    ap.add_argument("--out", default="/tmp/kitting_plan.jpg")
    a = ap.parse_args()

    img = cv2.imread(a.scan)
    if img is None:
        print("bad scan image"); sys.exit(1)
    known = load_known()
    scan = ScanReader(known_parts=known, min_conf=0.80).read_inventory(img)
    counts = scan.counts()
    if not counts:
        print("no confident SKUs read"); sys.exit(1)

    assigner = CompartmentAssigner.from_file(a.box)
    assignments = assigner.assign(counts, strategy=a.strategy)
    part2comp = {x.part: x.compartment for x in assignments}

    # ---- plan table ----
    print(f"\nKitting plan  ({len(counts)} SKUs, {sum(counts.values())} bags "
          f"→ {len(assigner.comps)} compartments, strategy={a.strategy})\n")
    print(f"  {'SKU':<16} {'bags':>4}  {'→ compartment':>14}")
    print("  " + "-" * 40)
    for x in assignments:
        print(f"  {x.part:<16} {x.count:>4}  {'c'+str(x.compartment):>14}")
    free = sorted({c['id'] for c in assigner.comps} - set(part2comp.values()))
    print(f"\n  free compartments: {['c'+str(i) for i in free] or 'none'}")

    # ---- plan overlay: tag each packet with its target compartment ----
    for it in scan.items:
        cx, cy = it.center
        ok = it.accepted and it.part in part2comp
        col = (60, 220, 60) if ok else (0, 140, 255)
        cv2.circle(img, (cx, cy), 6, col, -1)
        label = f"{it.part}->c{part2comp[it.part]}" if ok else f"{it.part or '??'} (abstain)"
        cv2.putText(img, label, (cx - 90, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, label, (cx - 90, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    cv2.imwrite(a.out, img)
    print(f"\n  overlay -> {a.out}")
    print("\n" + json.dumps({"plan": part2comp, "counts": counts}, indent=1))


if __name__ == "__main__":
    main()
