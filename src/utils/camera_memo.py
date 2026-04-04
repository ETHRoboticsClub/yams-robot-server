import json
import subprocess
from pathlib import Path

CAMERA_MEMO_PATH = Path(__file__).resolve().parents[2] / ".camera-signatures.json"
CAMERA_SIGNATURE_KEYS = ("ID_SERIAL_SHORT", "ID_SERIAL", "ID_PATH")


def _device_path(index_or_path):
    if isinstance(index_or_path, int) or (isinstance(index_or_path, str) and index_or_path.isdigit()):
        return f"/dev/video{int(index_or_path)}"
    return str(index_or_path)


def _udev_properties(device_path: str) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "-q", "property", "-n", str(Path(device_path).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _camera_signature(properties: dict[str, str]) -> dict[str, str]:
    return {key: properties[key] for key in CAMERA_SIGNATURE_KEYS if properties.get(key)}


def _load_camera_memo() -> dict[str, dict[str, str]] | None:
    if not CAMERA_MEMO_PATH.exists():
        return None
    try:
        data = json.loads(CAMERA_MEMO_PATH.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _save_camera_memo(memo: dict[str, dict[str, str]]) -> None:
    CAMERA_MEMO_PATH.write_text(json.dumps(memo, indent=2, sort_keys=True) + "\n")


def _scan_video_devices() -> dict[str, dict[str, str]]:
    devices: dict[str, dict[str, str]] = {}
    for path in sorted(Path("/dev").glob("video*")):
        try:
            devices[str(path)] = _udev_properties(str(path))
        except Exception:
            continue
    return devices


def _find_devices_from_memo(
    memo_by_name: dict[str, dict[str, str]],
    devices_with_properties: dict[str, dict[str, str]],
) -> dict[str, str]:
    device_by_name: dict[str, str] = {}
    used_devices: set[str] = set()
    for name, expected_signature in memo_by_name.items():
        if not expected_signature:
            continue
        matches = [
            device
            for device, properties in devices_with_properties.items()
            if all(properties.get(k) == v for k, v in expected_signature.items())
        ]
        if len(matches) == 1 and matches[0] not in used_devices:
            used_devices.add(matches[0])
            device_by_name[name] = matches[0]
    return device_by_name


def resolve_camera_configs(camera_configs: dict, logger=None) -> dict:
    configs = {name: dict(cfg) for name, cfg in camera_configs.items()}
    opencv_camera_names = [
        name for name, cfg in configs.items() if cfg.get("type", "zed") in ("opencv", "opencv-cached")
    ]
    if not opencv_camera_names:
        return configs

    devices_with_properties = _scan_video_devices()
    memo_by_name = _load_camera_memo() or {}
    for name, device in _find_devices_from_memo(memo_by_name, devices_with_properties).items():
        if name in configs:
            configs[name]["index_or_path"] = device

    memo_to_save: dict[str, dict[str, str]] = {}
    for name in opencv_camera_names:
        camera_path = _device_path(configs[name].get("index_or_path", ""))
        if not camera_path:
            continue
        try:
            signature = _camera_signature(_udev_properties(camera_path))
        except Exception:
            continue
        if signature:
            memo_to_save[name] = signature

    if memo_to_save:
        _save_camera_memo(memo_to_save)
        if logger:
            logger.info("Memoized camera signatures for: %s", ", ".join(sorted(memo_to_save)))

    return configs
