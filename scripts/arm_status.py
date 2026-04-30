"""Live terminal status for both YAM arms."""
import math
import sys
import time

import portal

LEFT_PORT = 11333
RIGHT_PORT = 11334
POLL_HZ = 5
JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6", "GRP"]


def _connect(port: int, label: str) -> portal.Client:
    while True:
        try:
            c = portal.Client(port)
            c.get_joint_pos()
            return c
        except Exception:
            sys.stdout.write(f"\rWaiting for {label} arm server on :{port}...  ")
            sys.stdout.flush()
            time.sleep(1.0)


def _fmt_row(joints) -> str:
    if joints is None:
        return "  ".join("    ---" for _ in JOINTS)
    return "  ".join(f"{math.degrees(j):+7.1f}°" for j in joints)


def main() -> None:
    print("Connecting to arm servers...")
    left = _connect(LEFT_PORT, "left")
    right = _connect(RIGHT_PORT, "right")

    header = "  ".join(f"{n:>8}" for n in JOINTS)
    sep = "─" * (len(JOINTS) * 11)

    try:
        while True:
            t0 = time.monotonic()
            try:
                lj = left.get_joint_pos()
                rj = right.get_joint_pos()
            except Exception as exc:
                sys.stdout.write(f"\rRead error: {exc}  ")
                sys.stdout.flush()
                time.sleep(1.0 / POLL_HZ)
                continue

            out = (
                f"\033[H\033[2J"
                f"  YAM Arm Status — {time.strftime('%H:%M:%S')}\n\n"
                f"  {'ARM':<8}  {header}\n"
                f"  {sep}\n"
                f"  {'LEFT':<8}  {_fmt_row(lj)}\n"
                f"  {'RIGHT':<8}  {_fmt_row(rj)}\n\n"
                f"  Ctrl+C to exit\n"
            )
            sys.stdout.write(out)
            sys.stdout.flush()

            elapsed = time.monotonic() - t0
            rem = 1.0 / POLL_HZ - elapsed
            if rem > 0:
                time.sleep(rem)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
