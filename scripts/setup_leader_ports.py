import subprocess
import time
from pathlib import Path


RULES_PATH = Path("/etc/udev/rules.d/99-dynamixel-leaders.rules")


def tty_devices() -> set[str]:
    return {path.name for path in Path("/dev").glob("ttyUSB*")}


def wait_for_new_device(side: str) -> str:
    input(f"Unplug the {side} leader cable, then press Enter.")
    before = tty_devices()
    input(f"Plug in the {side} leader cable, then press Enter.")
    while True:
        new_devices = tty_devices() - before
        if new_devices:
            device = sorted(new_devices)[0]
            print(f"Detected {device} for {side}.")
            return device
        time.sleep(0.2)


def serial_for(device: str) -> str:
    result = subprocess.run(
        ["udevadm", "info", "-q", "property", "-n", f"/dev/{device}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("ID_SERIAL_SHORT="):
            return line.removeprefix("ID_SERIAL_SHORT=")
    raise RuntimeError(f"Could not find ID_SERIAL_SHORT for {device}")


def main():
    if subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip() != "0":
        raise SystemExit(f"Run this with sudo so it can write {RULES_PATH}")

    serial_by_side = {
        side: serial_for(wait_for_new_device(side)) for side in ("left", "right")
    }
    rules = "\n".join(
        f'SUBSYSTEM=="tty", ATTRS{{serial}}=="{serial}", SYMLINK+="leader-{side}"'
        for side, serial in serial_by_side.items()
    )
    RULES_PATH.write_text(rules + "\n")
    subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["udevadm", "trigger"], check=True)
    print(f"Wrote {RULES_PATH}")
    print("Replug both leader cables to get /dev/leader-left and /dev/leader-right.")


if __name__ == "__main__":
    main()
