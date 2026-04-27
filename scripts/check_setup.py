from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

ROOT = Path(__file__).resolve().parents[1]
ARMS_CONFIG = ROOT / "configs" / "arms.yaml"
CAPTURED_IMAGES = ROOT / "outputs" / "captured_images"
REFERENCE_IMAGES = ROOT / "outputs" / "camera_reference_images"
CAMERA_PROFILE_SCRIPT = ROOT / "scripts" / "set_camera_profile.sh"
ALIGN_TOPDOWN_SCRIPT = ROOT / "scripts" / "align_topdown_visual.py"
CAMERA_MEMO_PATH = ROOT / ".camera-signatures.json"
CAMERA_SIGNATURE_KEYS = ("ID_SERIAL_SHORT", "ID_SERIAL", "ID_PATH")
WRIST_CAMERA_NAMES = ("right_wrist", "left_wrist")
IMAGE_MATCH_MIN_SCORE = 0.72
IMAGE_MATCH_MIN_MARGIN = 0.04


class TopdownPoseDriftError(RuntimeError):
    """Raised when the topdown RealSense is mounted at the wrong angle.

    Distinct from generic RuntimeError so check_cameras can offer the
    guided alignment tool instead of just aborting. Carries the Pose
    object itself so the caller can classify severity and format a
    per-axis breakdown without re-measuring.
    """

    def __init__(self, message: str, pose=None):
        super().__init__(message)
        self.pose = pose


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


def normalize_video_path(value) -> str:
    path = opencv_device_path(value)
    if str(value).startswith("/dev/v4l/"):
        return str(path.resolve())
    return str(path)


def udev_properties(device_path: str | Path) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "-q", "property", "-n", str(Path(device_path).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def camera_signature(properties: dict[str, str]) -> dict[str, str]:
    return {key: properties[key] for key in CAMERA_SIGNATURE_KEYS if properties.get(key)}


def load_camera_memo() -> dict[str, dict[str, str]]:
    if not CAMERA_MEMO_PATH.exists():
        return {}
    try:
        data = json.loads(CAMERA_MEMO_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_camera_memo(config: dict) -> None:
    memo: dict[str, dict[str, str]] = load_camera_memo()
    for name in WRIST_CAMERA_NAMES:
        camera = config.get("cameras", {}).get("configs", {}).get(name)
        if not camera:
            continue
        path = normalize_video_path(camera.get("index_or_path"))
        try:
            signature = camera_signature(udev_properties(path))
        except Exception:
            continue
        if signature:
            memo[name] = signature

    if memo:
        CAMERA_MEMO_PATH.write_text(json.dumps(memo, indent=2, sort_keys=True) + "\n")


def scan_video_device_properties() -> dict[str, dict[str, str]]:
    devices: dict[str, dict[str, str]] = {}
    for device in sorted(Path("/dev").glob("video*")):
        try:
            devices[str(device)] = udev_properties(device)
        except Exception:
            continue
    return devices


def find_devices_from_memo() -> dict[str, str]:
    memo = load_camera_memo()
    devices = scan_video_device_properties()
    found: dict[str, str] = {}
    used_devices: set[str] = set()
    for name, expected_signature in memo.items():
        if name not in WRIST_CAMERA_NAMES or not expected_signature:
            continue
        matches = [
            device
            for device, properties in devices.items()
            if all(properties.get(key) == value for key, value in expected_signature.items())
        ]
        if len(matches) == 1 and matches[0] not in used_devices:
            found[name] = matches[0]
            used_devices.add(matches[0])
    return found


def replace_camera_paths_in_yaml(paths_by_name: dict[str, str]) -> None:
    lines = ARMS_CONFIG.read_text().splitlines()
    current_camera: str | None = None
    updated_names: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        camera_header = re.match(r"^    ([A-Za-z0-9_]+):\s*$", line)
        if camera_header:
            current_camera = camera_header.group(1)

        index_line = re.match(r"^(\s*)index_or_path:\s*.*$", line)
        if index_line and current_camera in paths_by_name:
            new_lines.append(f"{index_line.group(1)}index_or_path: {paths_by_name[current_camera]}")
            updated_names.add(current_camera)
        else:
            new_lines.append(line)

    missing = sorted(set(paths_by_name) - updated_names)
    if missing:
        raise RuntimeError(f"Could not update index_or_path in {ARMS_CONFIG} for: {missing}")

    ARMS_CONFIG.write_text("\n".join(new_lines) + "\n")


def apply_memoized_camera_paths(config: dict) -> bool:
    found = find_devices_from_memo()
    if not found:
        return False

    changed_paths: dict[str, str] = {}
    camera_configs = config.get("cameras", {}).get("configs", {})
    for name, path in found.items():
        camera = camera_configs.get(name)
        if camera and normalize_video_path(camera.get("index_or_path")) != path:
            camera["index_or_path"] = path
            changed_paths[name] = path

    if changed_paths:
        replace_camera_paths_in_yaml(changed_paths)
        print(
            "Updated camera paths from saved hardware signatures: "
            + ", ".join(f"{name}={path}" for name, path in sorted(changed_paths.items()))
        )
    return bool(found)


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


def is_wrist_profile_device(path: Path) -> bool:
    returncode, output = run_command_text(["v4l2-ctl", "-d", str(path), "--list-ctrls"])
    return returncode == 0 and "brightness" in output and "exposure_time_absolute" in output


def wrist_profile_devices() -> list[Path]:
    return [device for device in sorted(Path("/dev").glob("video*")) if is_wrist_profile_device(device)]


def captured_image_for_device(path: Path) -> Path:
    return CAPTURED_IMAGES / f"opencv__dev_video{path.name.removeprefix('video')}.png"


def reference_image_for_camera(name: str) -> Path:
    return REFERENCE_IMAGES / f"{name}.png"


def seed_reference_images_from_captures(config: dict) -> None:
    REFERENCE_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in WRIST_CAMERA_NAMES:
        reference = reference_image_for_camera(name)
        if reference.exists():
            continue
        camera = config.get("cameras", {}).get("configs", {}).get(name)
        if not camera:
            continue
        capture = captured_image_for_device(Path(normalize_video_path(camera.get("index_or_path"))))
        if capture.exists():
            shutil.copyfile(capture, reference)


def prepare_image_for_similarity(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def image_similarity(reference: Path, candidate: Path) -> float | None:
    reference_image = prepare_image_for_similarity(reference)
    candidate_image = prepare_image_for_similarity(candidate)
    if reference_image is None or candidate_image is None:
        return None

    ref = reference_image.astype(np.float32)
    cand = candidate_image.astype(np.float32)
    ref_std = float(ref.std())
    cand_std = float(cand.std())
    if ref_std < 1.0 or cand_std < 1.0:
        return None

    corr = float(np.corrcoef(ref.reshape(-1), cand.reshape(-1))[0, 1])
    corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))

    ref_hist = cv2.calcHist([reference_image], [0], None, [32], [0, 256])
    cand_hist = cv2.calcHist([candidate_image], [0], None, [32], [0, 256])
    cv2.normalize(ref_hist, ref_hist)
    cv2.normalize(cand_hist, cand_hist)
    hist_score = 1.0 - float(cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA))
    hist_score = max(0.0, min(1.0, hist_score))

    return (0.75 * corr_score) + (0.25 * hist_score)


def image_similarity_report(candidates: list[Path]) -> tuple[dict[str, dict[str, float]], list[str]]:
    scores: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    for name in WRIST_CAMERA_NAMES:
        reference = reference_image_for_camera(name)
        if not reference.exists():
            problems.append(f"missing reference image for {name}: {reference}")
            continue
        scores[name] = {}
        for candidate in candidates:
            candidate_image = captured_image_for_device(candidate)
            if not candidate_image.exists():
                problems.append(f"missing new capture for {candidate}: {candidate_image}")
                continue
            score = image_similarity(reference, candidate_image)
            if score is None:
                problems.append(f"could not compare {reference} to {candidate_image}")
                continue
            scores[name][str(candidate)] = score
    return scores, problems


def best_image_mapping(candidates: list[Path]) -> tuple[dict[str, str] | None, str]:
    if len(candidates) < len(WRIST_CAMERA_NAMES):
        return None, "Not enough wrist-like camera candidates to assign left and right."

    scores, problems = image_similarity_report(candidates)
    if any(name not in scores or not scores[name] for name in WRIST_CAMERA_NAMES):
        details = "\n".join(problems) if problems else "No usable similarity scores were available."
        return None, f"Image matching cannot run yet.\n{details}"

    candidate_paths = [str(candidate) for candidate in candidates]
    best_assignment: dict[str, str] | None = None
    best_total = -1.0
    second_total = -1.0

    for right_candidate in candidate_paths:
        for left_candidate in candidate_paths:
            if left_candidate == right_candidate:
                continue
            assignment = {"right_wrist": right_candidate, "left_wrist": left_candidate}
            try:
                total = sum(scores[name][path] for name, path in assignment.items())
            except KeyError:
                continue
            if total > best_total:
                second_total = best_total
                best_total = total
                best_assignment = assignment
            elif total > second_total:
                second_total = total

    if best_assignment is None:
        return None, "No valid one-to-one image assignment was found."

    selected_scores = {name: scores[name][path] for name, path in best_assignment.items()}
    lowest_score = min(selected_scores.values())
    margin = best_total - second_total if second_total >= 0 else best_total

    score_lines = [
        f"{name}: {path} score={score:.3f}"
        for name, path in best_assignment.items()
        for score in [selected_scores[name]]
    ]
    if lowest_score < IMAGE_MATCH_MIN_SCORE:
        return (
            None,
            "Image match was too weak to trust automatically.\n"
            + "\n".join(score_lines)
            + f"\nMinimum required score is {IMAGE_MATCH_MIN_SCORE:.2f}.",
        )
    if margin < IMAGE_MATCH_MIN_MARGIN:
        return (
            None,
            "Image match was ambiguous, so I will not guess.\n"
            + "\n".join(score_lines)
            + f"\nBest-vs-second margin was {margin:.3f}; required margin is {IMAGE_MATCH_MIN_MARGIN:.2f}.",
        )

    return best_assignment, "Auto-matched by comparing fresh captures to saved reference images."


def remove_old_opencv_captures() -> None:
    CAPTURED_IMAGES.mkdir(parents=True, exist_ok=True)
    for image in CAPTURED_IMAGES.glob("opencv__dev_video*.png"):
        image.unlink()


def run_lerobot_find_cameras() -> None:
    print("Running: uv run lerobot-find-cameras")
    remove_old_opencv_captures()
    result = subprocess.run(["uv", "run", "lerobot-find-cameras"], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "lerobot-find-cameras failed. Fix that first, then rerun: "
            "uv run python scripts/check_setup.py"
        )


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def parse_video_choice(answer: str) -> Path:
    answer = answer.strip()
    match = re.fullmatch(r"(?:/dev/video)?(\d+)", answer)
    if not match:
        raise ValueError("Enter a device number like 0 or a path like /dev/video0.")
    return Path(f"/dev/video{match.group(1)}")


def prompt_for_wrist_camera(name: str, candidates: list[Path]) -> Path:
    while True:
        answer = input(f"Which device number is {name}? ").strip()
        try:
            path = parse_video_choice(answer)
        except ValueError as exc:
            print(exc)
            continue
        if path not in candidates:
            print(
                f"{path} is not in the wrist-like candidates. "
                "Use one of: " + ", ".join(str(candidate) for candidate in candidates)
            )
            continue
        return path


def apply_selected_camera_mapping(selected: dict[str, str]) -> dict:
    replace_camera_paths_in_yaml(selected)
    config = load_config()
    save_camera_memo(config)
    print(
        "Updated camera config: "
        + ", ".join(f"{name}={path}" for name, path in selected.items())
    )

    for name, path in selected.items():
        apply_camera_profile(name, Path(path))

    return config


def repair_camera_mapping_interactively(config: dict) -> dict:
    seed_reference_images_from_captures(config)
    run_lerobot_find_cameras()

    candidates = wrist_profile_devices()
    if not candidates:
        raise RuntimeError(
            "No wrist-like cameras were found. Check USB cables, camera permissions, and power.\n"
            f"{camera_mapping_diagnostics()}"
        )

    selected, match_message = best_image_mapping(candidates)
    print(match_message)
    if selected:
        print(
            "Image match accepted: "
            + ", ".join(f"{name}={path}" for name, path in selected.items())
        )
        return apply_selected_camera_mapping(selected)

    print("")
    print("Captured images are here:")
    print(f"  {CAPTURED_IMAGES}")
    print("Reference images are here:")
    print(f"  {REFERENCE_IMAGES}")
    print("")
    print("Wrist-like camera candidates:")
    for candidate in candidates:
        image = captured_image_for_device(candidate)
        image_status = str(image) if image.exists() else "no captured image found"
        print(f"  {candidate.name.removeprefix('video')}: {candidate} -> {image_status}")
    print("")
    print("Open the fresh captured images and compare them to the reference images.")
    print("Then enter the device number for each wrist camera, for example 0 or 8.")

    selected: dict[str, str] = {}
    used: set[Path] = set()
    for name in WRIST_CAMERA_NAMES:
        path = prompt_for_wrist_camera(name, candidates)
        while path in used:
            print(f"{path} was already selected. Pick the other wrist camera.")
            path = prompt_for_wrist_camera(name, candidates)
        used.add(path)
        selected[name] = str(path)

    return apply_selected_camera_mapping(selected)


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


def check_opencv_camera(name: str, camera: dict) -> np.ndarray:
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

    return frame


def save_reference_frames(frames_by_name: dict[str, np.ndarray]) -> None:
    if not frames_by_name:
        return
    REFERENCE_IMAGES.mkdir(parents=True, exist_ok=True)
    for name, frame in frames_by_name.items():
        if frame is None:
            continue
        cv2.imwrite(str(reference_image_for_camera(name)), frame)


def check_realsense_camera(name: str, camera: dict) -> list[np.ndarray]:
    """Verify the RealSense is present and grab a burst of color frames.

    Returns a list of 30 BGR frames captured AFTER the configured warmup
    period. Raises RuntimeError if the device is missing or if streaming
    doesn't come up. Used both for the wrist-style reference-image save
    and, for the topdown, for the pose gate.

    Always issues hardware_reset() before opening the pipeline. Without
    this, a prior aborted lerobot-record leaves the device in a state
    where pipeline.start() succeeds but frames never arrive (VIDIOC_S_FMT
    EBUSY). The reset costs ~2s and makes cold-start deterministic.
    """
    import time

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
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            device.hardware_reset()
            break
    time.sleep(2)

    width = int(camera.get("width", 640))
    height = int(camera.get("height", 480))
    fps = int(camera.get("fps", 30))
    warmup_s = float(camera.get("warmup_s", 3))
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(rs_config)
    try:
        first_timeout_ms = max(8000, int(warmup_s * 1000) + 2000)
        pipeline.wait_for_frames(timeout_ms=first_timeout_ms)
        for _ in range(int(warmup_s * fps)):
            pipeline.wait_for_frames(timeout_ms=2000)
        burst: list[np.ndarray] = []
        for _ in range(30):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            burst.append(np.asanyarray(color.get_data()).copy())
        return burst
    finally:
        pipeline.stop()


def check_topdown_pose(frames: list[np.ndarray]) -> None:
    """Average the topdown burst and compare pose vs the committed reference.

    Three-tier verdict from evaluate_pose:
      - OK: print success, continue.
      - MARGINAL (drift within DRIFT_MULTIPLIER× tolerance): print warning
        with the current offsets and continue — the setup is slightly off
        but the training data tolerates it.
      - DRIFT (beyond DRIFT_MULTIPLIER× tolerance): raise so the caller
        can offer the alignment tool.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from utils.camera_pose import evaluate_pose, load_reference  # noqa: E402

    avg = np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
    reference = load_reference()
    pose, ok, msg = evaluate_pose(avg, reference)
    if not ok:
        raise TopdownPoseDriftError(msg, pose=pose)
    if "MARGINAL" in msg:
        print(f"Heads up, topdown {msg}")
        print("  (slightly off, within acceptable wiggle — continuing)")
        return
    print(f"Okay, topdown {msg}")


def present_topdown_drift(exc: TopdownPoseDriftError) -> bool:
    """Print a formatted drift report and return whether the operator should
    be urged to run the alignment tool. Returns True when the worst-axis
    ratio meets or exceeds RECOMMEND_ALIGNMENT_MULTIPLIER.
    """
    from utils.camera_pose import (  # noqa: E402
        RECOMMEND_ALIGNMENT_MULTIPLIER,
        format_drift_breakdown,
        worst_tolerance_ratio,
    )

    pose = exc.pose
    print("")
    print("Topdown camera pose is off from the committed reference:")
    if pose is None:
        # Obstruction / low-confidence path — no per-axis numbers available.
        print(f"  {exc}")
        print("")
        return True

    print(format_drift_breakdown(pose))
    worst = worst_tolerance_ratio(pose)
    print("")
    print(
        f"Worst-axis drift is {worst:.1f}× tolerance "
        f"(recommend threshold is {RECOMMEND_ALIGNMENT_MULTIPLIER:g}×)."
    )
    recommend = worst >= RECOMMEND_ALIGNMENT_MULTIPLIER
    if recommend:
        print("Recommendation: RUN the alignment tool — the setup is visibly off.")
    else:
        print(
            "Recommendation: alignment is OPTIONAL — the setup is workable, "
            "you can skip and proceed."
        )
    return recommend


def repair_topdown_pose_interactively() -> None:
    """Launch the real-time alignment tool so the user can physically re-aim
    the mount. Returns once the user exits the tool (Ctrl+C). The caller
    is expected to re-capture a burst and re-evaluate pose afterward.
    """
    print("")
    print("Opening the real-time visual alignment tool. It writes live")
    print("overlays (blend, checkerboard, diff, side-by-side) to")
    print("outputs/alignment_diff/ — open any of them in VS Code and its image")
    print("preview will auto-reload as you re-aim the mount.")
    print("")
    print("Only the top band (mat + table background) is used for alignment.")
    print("The gripper/scene region is masked out, so ignore differences there.")
    print("Press Ctrl+C to exit the tool.")
    print("")
    subprocess.run(
        ["uv", "run", "python", str(ALIGN_TOPDOWN_SCRIPT)],
        cwd=ROOT,
        check=False,
    )
    # Give the RealSense a moment to fully release before we re-open it.
    time.sleep(1.5)
    print("")
    print("Re-checking topdown pose...")


def check_cameras(config: dict) -> None:
    apply_memoized_camera_paths(config)

    repaired = False
    topdown_pose_repaired = False
    reference_frames: dict[str, np.ndarray] = {}
    while True:
        restart_checks = False
        reference_frames = {}
        for name, camera in config.get("cameras", {}).get("configs", {}).items():
            camera_type = camera.get("type")
            if camera_type in ("opencv", "opencv-cached"):
                try:
                    frame = check_opencv_camera(name, camera)
                    if name in WRIST_CAMERA_NAMES:
                        reference_frames[name] = frame
                except RuntimeError as exc:
                    print(exc)
                    if repaired or name not in WRIST_CAMERA_NAMES:
                        raise
                    if not prompt_yes_no("Camera check failed. Run guided camera remapping now?"):
                        raise
                    config = repair_camera_mapping_interactively(config)
                    repaired = True
                    restart_checks = True
                    break
            elif camera_type == "intelrealsense-cached":
                frames = check_realsense_camera(name, camera)
                if name == "topdown":
                    try:
                        check_topdown_pose(frames)
                    except TopdownPoseDriftError as exc:
                        if topdown_pose_repaired:
                            # Alignment was already attempted this session
                            # and pose is still off — alert loudly but do NOT
                            # fail: pose drift never blocks the record script.
                            print("")
                            print(
                                f"WARNING: topdown camera pose still off "
                                f"after alignment: {exc}"
                            )
                            print(
                                "Proceeding anyway — pose drift is alert-only. "
                                "Fix the mount later with "
                                "`uv run python scripts/align_topdown_visual.py` "
                                "if the numbers matter for this session."
                            )
                        else:
                            recommend = present_topdown_drift(exc)
                            print("")
                            if prompt_yes_no(
                                "Launch the real-time alignment tool now?",
                                default=recommend,
                            ):
                                repair_topdown_pose_interactively()
                                topdown_pose_repaired = True
                                restart_checks = True
                                break
                            # Declined → warn and proceed. Pose drift never
                            # aborts recording, regardless of severity.
                            if recommend:
                                print(
                                    f"WARNING: topdown camera is significantly "
                                    f"off (alignment recommended, declined): {exc}"
                                )
                                print(
                                    "Run `uv run python scripts/align_topdown_visual.py` "
                                    "later to fix it. Proceeding with recording now."
                                )
                            else:
                                print(
                                    "Proceeding without alignment. The setup "
                                    "is off but within the acceptable range "
                                    "for this session."
                                )
            else:
                raise RuntimeError(f"{name} has unsupported camera type: {camera_type}")

        if not restart_checks:
            break

    save_camera_memo(config)
    save_reference_frames(reference_frames)
    print("Okay, USB cameras receiving frames")


def main() -> None:
    config = load_config()
    check_cans(config)
    check_leaders(config)
    check_cameras(config)
    print("Done")


if __name__ == "__main__":
    main()
