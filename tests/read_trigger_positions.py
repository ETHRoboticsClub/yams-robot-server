#!/usr/bin/env python3
"""Read raw gripper encoder values from both leader arm triggers.

Run this script, then squeeze and release each trigger to find the
open/closed encoder values to set in BiYamsLeaderConfig.
"""

import time
import yaml
from pathlib import Path
from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

ARMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"


def read_trigger(side: str, port: str, duration_s: float = 15.0) -> None:
    cfg = YamsLeaderConfig(port=port, side=side)
    leader = YamsLeader(cfg)
    leader.connect()

    print(f"\n[{side}] Squeeze and release the trigger over the next {duration_s:.0f}s...")
    print(f"{'Time':>6}  {'Raw gripper':>12}")
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < duration_s:
            try:
                val = leader.bus.read("Present_Position", "gripper", normalize=False)
                elapsed = time.perf_counter() - start
                print(f"{elapsed:6.1f}s  {val:12d}")
            except Exception:
                pass
            time.sleep(0.2)
    finally:
        leader.disconnect()


def main() -> None:
    with open(ARMS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    ports = {
        "left": cfg["leader"]["left_arm"]["port"],
        "right": cfg["leader"]["right_arm"]["port"],
    }

    for side, port in ports.items():
        read_trigger(side, port)
        print(f"\n[{side}] Done. Note the min and max values above for open/closed positions.\n")


if __name__ == "__main__":
    main()
