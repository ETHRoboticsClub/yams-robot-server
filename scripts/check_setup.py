from pathlib import Path
import re
import subprocess

import cv2
import pyrealsense2 as rs
import yaml

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

ROOT = Path(__file__).resolve().parents[1]
ARMS_CONFIG = ROOT / "configs" / "arms.yaml"
CAPTURED_IMAGES = ROOT / "outputs" / "captured_images"
CAMERA_PROFILE_SCRIPT = ROOT / "scripts" / "set_camera_profile.sh"


def load_config() -> dict:
    with open(ARMS_CONFIG, "r") as f:
        return yaml.safe_load(f)


def check_cans(config: dict) -> None:
    for side in ("left", "right"):
        can = config["follower"][f"{side}_arm"]["can_port"]
        state_path = Path("/sys/class/net") / can / "operstate"
        if not state_path.exists():
            raise RuntimeError(
                f"{side} follower CAN interface not found: {can}\n"
                f"{can_fix_instructions(can)}"
            )
        if state_path.read_text().strip() == "down":
            raise RuntimeError(
                f"{side} follower CAN interface is down: {can}\n"
                f"{can_fix_instructions(can)}"
            )
    print("Okay, cans connected")


def check_leaders(config: dict) -> None:
    for side in ("left", "right"):
        port = config["leader"][f"{side}_arm"]["port"]
        if not Path(port).exists():
            raise RuntimeError(f"{side} leader port not found: {port}")

        leader = YamsLeader(YamsLeaderConfig(port=port, side=side))
        try:
            leader.bus.connect()
            positions = leader.bus.sync_read(normalize=False, data_name="Present_Position")
        finally:
            if leader.bus.is_connected:
                leader.bus.disconnect()

        expected = set(config["leader"][f"{side}_arm"]["motors"])
        missing = expected - set(positions)
        if missing:
            raise RuntimeError(f"{side} leader did not return positions for: {sorted(missing)}")

        calibration = ROOT / "src/lerobot_teleoperator_gello/calibration" / f"leader_calibration_{side}.yaml"
        if not calibration.exists():
            raise RuntimeError(f"{side} leader calibration offsets not found: {calibration}")
    print("Okay, leader USBs receiving offsets")


def opencv_device_path(index_or_path) -> Path:
    if isinstance(index_or_path, int) or str(index_or_path).isdigit():
        return Path(f"/dev/video{int(index_or_path)}")
    return Path(str(index_or_path))


def run_command_text(command: list[str]) -> tuple[int | None, str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None, f"{command[0]} is not installed or not on PATH."
    return result.returncode, (result.stdout + result.stderr).strip()


def can_fix_instructions(can: str) -> str:
    _, ip_link = run_command_text(["ip", "link", "show", can])
    return "\n".join(
        [
            "Fix from README:",
            "1. Turn on both power strips. Follower fans should start making noise.",
            "2. Check CAN state: ip link show",
            "3. Reset CAN buses: sudo sh third_party/i2rt/scripts/reset_all_can.sh",
            "4. Check again: ip link show",
            "5. Rerun: uv run python scripts/check_setup.py",
            "",
            f"Current {can} state:",
            ip_link or f"No output for {can}.",
        ]
    )


def capture_path(path: Path) -> Path:
    resolved = path.resolve()
    if str(resolved).startswith("/dev/video"):
        return resolved
    return path


def opencv_capture_target(path: Path) -> tuple[int | str, str]:
    match = re.fullmatch(r"/dev/video(\d+)", str(path))
    if match:
        index = int(match.group(1))
        return index, f"camera index {index} ({path})"
    return str(path), str(path)


def available_video_devices() -> str:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        return "No /dev/video* devices were visible."
    return "Visible camera devices: " + ", ".join(str(device) for device in devices)


def symlink_report(directory: Path) -> list[str]:
    if not directory.exists():
        return [f"{directory} does not exist."]

    lines = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_symlink():
            continue
        target = entry.resolve()
        if str(target).startswith("/dev/video"):
            lines.append(f"{entry.name} -> {target}")
    return lines or [f"No /dev/video* symlinks found in {directory}."]


def video_permission_hint(path: Path) -> str | None:
    _, output = run_command_text(["id"])
    if "video" not in output:
        return (
            f"{path} may need video-group access. Run: sudo usermod -aG video $USER; "
            "then log out and back in or reboot. Confirm with: id"
        )
    return None


def camera_controls_summary() -> list[str]:
    lines = []
    for device in sorted(Path("/dev").glob("video*")):
        returncode, output = run_command_text(["v4l2-ctl", "-d", str(device), "--list-ctrls"])
        if returncode is None:
            return [output]
        if returncode != 0:
            lines.append(f"{device}: cannot read controls ({output})")
            permission_hint = video_permission_hint(device)
            if permission_hint:
                lines.append(f"  {permission_hint}")
            continue

        has_brightness = "brightness" in output
        has_exposure = "exposure_time_absolute" in output
        if has_brightness and has_exposure:
            lines.append(f"{device}: wrist-like controls found")
        else:
            lines.append(f"{device}: not a wrist profile device")

    return lines or ["No /dev/video* devices found."]


def camera_mapping_diagnostics() -> str:
    sections = [
        "Camera diagnostics:",
        available_video_devices(),
        "",
        "Stable paths by physical USB path:",
        *symlink_report(Path("/dev/v4l/by-path")),
        "",
        "Stable paths by camera id:",
        *symlink_report(Path("/dev/v4l/by-id")),
        "",
        "Control check:",
        *camera_controls_summary(),
        "",
        "Rule of thumb:",
        "- Wrist cameras should expose brightness and exposure_time_absolute controls.",
        "- RealSense OpenCV views may show up as /dev/video2 or /dev/video4 but are not wrist cameras.",
        "- If /dev/video6 or /dev/video8 exists but says Permission denied, add the user to the video group and relogin.",
    ]
    return "\n".join(sections)


def camera_fix_instructions(name: str, configured_path: Path, attempted_path: Path) -> str:
    _, attempted_label = opencv_capture_target(attempted_path)
    hint = [
        f"{name} camera is configured as: {configured_path}",
        f"OpenCV tried to read from: {attempted_label}",
        "",
        "Fix:",
        "1. Run: uv run lerobot-find-cameras",
        f"2. Open the captured images in: {CAPTURED_IMAGES}",
        f"3. Find the image for {name}.",
        (
            f"4. In {ARMS_CONFIG}, set cameras.configs.{name}.index_or_path "
            "to the matching short /dev/videoN path."
        ),
        "   Example: index_or_path: /dev/video0",
        "5. For wrist cameras only, apply the profile to that same short path:",
        "   ./scripts/set_camera_profile.sh /dev/videoN",
        "6. Rerun: uv run python scripts/check_setup.py",
        "",
        "If OpenCV says it cannot open the camera by index, that /dev/videoN is probably busy, "
        "not a capture stream, or not the wrist camera.",
        camera_mapping_diagnostics(),
    ]
    if "/dev/v4l/by-path/" in str(configured_path):
        hint.insert(
            3,
            "The configured value is a long /dev/v4l/by-path/... symlink. "
            "Use the short /dev/videoN path from lerobot-find-cameras instead.",
        )
    return "\n".join(hint)


def should_apply_camera_profile(camera: dict) -> bool:
    return any(key.startswith("auto_exposure_") for key in camera)


def apply_camera_profile(name: str, path: Path) -> None:
    try:
        ctrl_check = subprocess.run(
            ["v4l2-ctl", "-d", str(path), "--list-ctrls"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("v4l2-ctl is not installed; install v4l-utils first.") from exc

    controls = ctrl_check.stdout
    if ctrl_check.returncode != 0:
        detail = (ctrl_check.stderr or ctrl_check.stdout).strip()
        raise RuntimeError(
            f"{name} camera controls could not be read from {path}.\n"
            f"{detail}\n"
            f"{camera_fix_instructions(name, path, path)}"
        )

    required_controls = ("brightness", "exposure_time_absolute")
    if not all(control in controls for control in required_controls):
        raise RuntimeError(
            f"{name} camera at {path} does not expose the wrist camera controls.\n"
            "This is usually the wrong /dev/videoN node.\n"
            f"{camera_fix_instructions(name, path, path)}"
        )

    result = subprocess.run(
        [str(CAMERA_PROFILE_SCRIPT), str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"{name} camera profile failed for {path}.\n"
            f"{detail}\n"
            f"{camera_fix_instructions(name, path, path)}"
        )
    print(f"Okay, applied camera profile to {name}: {path}")


def check_opencv_camera(name: str, camera: dict) -> None:
    path = opencv_device_path(camera["index_or_path"])
    if not path.exists():
        raise RuntimeError(
            f"{name} camera path not found: {path}\n"
            f"{camera_fix_instructions(name, path, path)}"
        )

    attempted_path = capture_path(path)
    if should_apply_camera_profile(camera):
        apply_camera_profile(name, attempted_path)

    capture_target, _ = opencv_capture_target(attempted_path)
    cap = cv2.VideoCapture(capture_target, cv2.CAP_V4L2)
    try:
        opened = cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["height"])
        cap.set(cv2.CAP_PROP_FPS, camera["fps"])
        if "fourcc" in camera:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera["fourcc"]))
        ok, frame = cap.read()
    finally:
        cap.release()

    if not opened:
        raise RuntimeError(
            f"{name} camera could not be opened.\n"
            f"{camera_fix_instructions(name, path, attempted_path)}"
        )

    if not ok or frame is None:
        raise RuntimeError(
            f"{name} camera did not return a frame.\n"
            f"{camera_fix_instructions(name, path, attempted_path)}"
        )


def check_realsense_camera(name: str, camera: dict) -> None:
    serial = str(camera["serial_number_or_name"])
    devices = list(rs.context().query_devices())
    serials = {device.get_info(rs.camera_info.serial_number) for device in devices}
    if serial not in serials:
        found = ", ".join(serials) or "none"
        _, usb = run_command_text(["lsusb"])
        raise RuntimeError(
            f"{name} RealSense not found: {serial}\n"
            f"RealSense serials visible to pyrealsense2: {found}\n"
            "Expected to see an Intel RealSense device in lsusb. If it is missing, "
            "replug the RealSense into a USB3 port/cable and avoid passive hubs. "
            "If it still does not show up, try a different cable; sometimes unplugging "
            "and plugging it back in a few times randomly makes it enumerate.\n"
            f"lsusb output:\n{usb}"
        )


def check_cameras(config: dict) -> None:
    errors = []
    for name, camera in config.get("cameras", {}).get("configs", {}).items():
        camera_type = camera.get("type")
        try:
            if camera_type in ("opencv", "opencv-cached"):
                check_opencv_camera(name, camera)
            elif camera_type == "intelrealsense-cached":
                check_realsense_camera(name, camera)
            else:
                raise RuntimeError(f"{name} has unsupported camera type: {camera_type}")
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("\n\n".join(errors))
    print("Okay, USB cameras receiving frames")


def main() -> None:
    config = load_config()
    check_cans(config)
    check_leaders(config)
    check_cameras(config)
    print("Done")


if __name__ == "__main__":
    main()
