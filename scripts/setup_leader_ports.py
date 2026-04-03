import os
import subprocess
import time
from pathlib import Path


RULES_PATH = Path("/etc/udev/rules.d/99-dynamixel-leaders.rules")
DETECTION_TIMEOUT_S = 5 * 60
SIGNATURE_KEYS = (
    "ID_VENDOR_ID",
    "ID_MODEL_ID",
    "ID_VENDOR",
    "ID_MODEL",
    "ID_USB_DRIVER",
)


def tty_devices() -> set[str]:
    return {path.name for path in Path("/dev").glob("ttyUSB*")}


def wait_for_new_device(side: str, timeout_s: float = DETECTION_TIMEOUT_S) -> str:
    input(f"Unplug the {side} leader cable, then press Enter.")
    before = tty_devices()
    input(f"Plug in the {side} leader cable, then press Enter.")
    deadline = time.monotonic() + timeout_s
    while True:
        new_devices = tty_devices() - before
        if new_devices:
            device = sorted(new_devices)[0]
            print(f"Detected {device} for {side}.")
            return device
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {side} leader after {int(timeout_s)} seconds.")
        time.sleep(0.2)


def udev_properties_for(device: str) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "-q", "property", "-n", f"/dev/{device}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )


def signature_for(properties: dict[str, str]) -> tuple[str, ...]:
    return tuple(properties.get(key, "") for key in SIGNATURE_KEYS)


def infer_second_device(first_device: str, first_properties: dict[str, str]) -> str | None:
    candidates = []
    first_serial = first_properties.get("ID_SERIAL_SHORT")
    first_signature = signature_for(first_properties)
    for device in sorted(tty_devices()):
        if device == first_device:
            continue
        properties = udev_properties_for(device)
        if properties.get("ID_SERIAL_SHORT") == first_serial:
            continue
        if signature_for(properties) == first_signature:
            candidates.append(device)
    return candidates[0] if len(candidates) == 1 else None


def main():
    if os.geteuid() != 0:
        raise SystemExit(f"Run this with sudo so it can write {RULES_PATH}")

    left_device = wait_for_new_device("left")
    left_properties = udev_properties_for(left_device)
    left_serial = left_properties.get("ID_SERIAL_SHORT")
    if not left_serial:
        raise RuntimeError(f"Could not find ID_SERIAL_SHORT for {left_device}")

    right_device = infer_second_device(left_device, left_properties)
    if right_device:
        print(f"Inferred {right_device} for right leader from matching USB signature.")
    else:
        right_device = wait_for_new_device("right")
    right_properties = udev_properties_for(right_device)
    right_serial = right_properties.get("ID_SERIAL_SHORT")
    if not right_serial:
        raise RuntimeError(f"Could not find ID_SERIAL_SHORT for {right_device}")

    serial_by_side = {"left": left_serial, "right": right_serial}
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
