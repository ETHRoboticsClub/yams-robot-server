from pathlib import Path
import platform
from typing import Any

import cv2

MAX_OPENCV_INDEX = 15


def find_opencv_cameras() -> list[dict[str, Any]]:
    found_cameras_info = []

    targets_to_scan: list[str | int]
    if platform.system() == "Linux":
        possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
        targets_to_scan = [str(p) for p in possible_paths]
    else:
        targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

    for target in targets_to_scan:
        camera = cv2.VideoCapture(target)
        if camera.isOpened():
            default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps = camera.get(cv2.CAP_PROP_FPS)
            default_format = camera.get(cv2.CAP_PROP_FORMAT)

            default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
            default_fourcc_code_int = int(default_fourcc_code)
            default_fourcc = "".join(
                [chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)]
            )

            camera_info = {
                "name": f"OpenCV Camera @ {target}",
                "type": "OpenCV",
                "id": target,
                "backend_api": camera.getBackendName(),
                "default_stream_profile": {
                    "format": default_format,
                    "fourcc": default_fourcc,
                    "width": default_width,
                    "height": default_height,
                    "fps": default_fps,
                },
            }

            found_cameras_info.append(camera_info)
            camera.release()

    return found_cameras_info
