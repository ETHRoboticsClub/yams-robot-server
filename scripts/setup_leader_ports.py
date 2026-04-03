import argparse
import json
import os
import subprocess
import time
from pathlib import Path


RULES_PATH = Path("/etc/udev/rules.d/99-dynamixel-leaders.rules")
MEMO_PATH = Path("/etc/yams-leader-signatures.json")
DETECTION_TIMEOUT_SECONDS = 5 * 60
SIGNATURE_KEYS = (
    "ID_VENDOR_ID",
    "ID_MODEL_ID",
    "ID_VENDOR",
    "ID_MODEL",
    "ID_USB_DRIVER",
)
MEMO_KEYS = ("ID_SERIAL_SHORT", *SIGNATURE_KEYS)


def tty_devices() -> set[str]:
    return {path.name for path in Path("/dev").glob("ttyUSB*")}


def wait_for_new_device(side: str, timeout_s: float = DETECTION_TIMEOUT_SECONDS) -> str:
    input(f"Unplug the {side} leader cable, then press Enter.")
    before = tty_devices()
    input(f"Plug in the {side} leader cable, then press Enter.")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        new_devices = tty_devices() - before
        if new_devices:
            device = sorted(new_devices)[0]
            print(f"Detected {device} for {side}.")
            return device
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {side} leader after {timeout_s:g} seconds.")


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


def memo_signature_for(properties: dict[str, str]) -> dict[str, str]:
    return {key: properties.get(key, "") for key in MEMO_KEYS}


def infer_second_device(
    first_device: str,
    first_properties: dict[str, str],
    devices_with_properties: dict[str, dict[str, str]],
) -> str | None:
    candidates = []
    first_serial = first_properties.get("ID_SERIAL_SHORT")
    first_signature = signature_for(first_properties)
    for device, properties in sorted(devices_with_properties.items()):
        if device == first_device:
            continue
        if properties.get("ID_SERIAL_SHORT") == first_serial:
            continue
        if signature_for(properties) == first_signature:
            candidates.append(device)
            if len(candidates) > 1:
                return None
    return candidates[0] if len(candidates) == 1 else None


def scan_tty_properties() -> dict[str, dict[str, str]]:
    return {device: udev_properties_for(device) for device in sorted(tty_devices())}


def load_memo() -> dict[str, dict[str, str]] | None:
    if not MEMO_PATH.exists():
        return None
    data = json.loads(MEMO_PATH.read_text())
    if isinstance(data, dict):
        return data
    return None


def save_memo(serial_by_side: dict[str, dict[str, str]]) -> None:
    MEMO_PATH.write_text(json.dumps(serial_by_side, indent=2, sort_keys=True) + "\n")


def find_devices_from_memo(
    memo_by_side: dict[str, dict[str, str]],
    devices_with_properties: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    if not all(side in memo_by_side for side in ("left", "right")):
        return None
    devices_by_side: dict[str, str] = {}
    used_devices: set[str] = set()
    for side in ("left", "right"):
        expected = memo_by_side[side]
        matches = [
            device
            for device, properties in devices_with_properties.items()
            if all(properties.get(key, "") == expected.get(key, "") for key in MEMO_KEYS)
        ]
        if len(matches) != 1 or matches[0] in used_devices:
            return None
        used_devices.add(matches[0])
        devices_by_side[side] = matches[0]
    return devices_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map leader USB devices to stable udev symlinks.")
    parser.add_argument(
        "--force-manual",
        action="store_true",
        help="Force manual unplug/replug detection even if memoized signatures exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit(f"Run this with sudo so it can write {RULES_PATH}")

    devices_with_properties = scan_tty_properties()
    devices_by_side: dict[str, str] | None = None
    if not args.force_manual:
        memo_by_side = load_memo()
        if memo_by_side:
            devices_by_side = find_devices_from_memo(memo_by_side, devices_with_properties)
            if devices_by_side:
                print("Detected both leaders from memoized signatures.")

    if not devices_by_side:
        left_device = wait_for_new_device("left")
        left_properties = udev_properties_for(left_device)
        devices_with_properties = scan_tty_properties()
        devices_with_properties[left_device] = left_properties
        right_device = None if args.force_manual else infer_second_device(
            left_device, left_properties, devices_with_properties
        )
        if right_device:
            print(f"Inferred {right_device} for right leader from matching USB signature.")
        else:
            right_device = wait_for_new_device("right")
        devices_by_side = {"left": left_device, "right": right_device}

    left_properties = udev_properties_for(devices_by_side["left"])
    left_serial = left_properties.get("ID_SERIAL_SHORT")
    if not left_serial:
        raise RuntimeError(f"Could not find ID_SERIAL_SHORT for {devices_by_side['left']}")

    right_properties = udev_properties_for(devices_by_side["right"])
    right_serial = right_properties.get("ID_SERIAL_SHORT")
    if not right_serial:
        raise RuntimeError(f"Could not find ID_SERIAL_SHORT for {devices_by_side['right']}")

    serial_by_side = {"left": left_serial, "right": right_serial}
    save_memo(
        {
            "left": memo_signature_for(left_properties),
            "right": memo_signature_for(right_properties),
        }
    )
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
