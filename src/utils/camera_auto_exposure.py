from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

import numpy as np


def frame_brightness(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float(np.percentile(crop.mean(axis=2), 75))


def next_exposure(
    exposure: int,
    brightness: float,
    target: float,
    deadband: float,
    speed: float,
    min_exposure: int,
    max_exposure: int,
) -> int:
    error = target - brightness
    if abs(error) <= deadband:
        return exposure
    scale = 1.0 + speed * error / max(target, 1.0)
    return max(min_exposure, min(max_exposure, round(exposure * scale)))


@dataclass
class CameraAutoExposure:
    device: str | Path
    exposure: int
    target: float
    deadband: float
    speed: float
    min_exposure: int
    max_exposure: int
    period_s: float
    last_update: float = 0.0

    def tick(self, frame: np.ndarray) -> int | None:
        now = time.monotonic()
        if now - self.last_update < self.period_s:
            return None
        self.last_update = now
        exposure = next_exposure(
            exposure=self.exposure,
            brightness=frame_brightness(frame),
            target=self.target,
            deadband=self.deadband,
            speed=self.speed,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
        )
        if exposure == self.exposure:
            return None
        self.exposure = exposure
        _set_exposure(self.device, exposure)
        return exposure


def _set_exposure(device: str | Path, exposure: int) -> None:
    subprocess.run(
        ["v4l2-ctl", "-d", str(device), f"--set-ctrl=exposure_time_absolute={exposure}"],
        check=True,
        capture_output=True,
        text=True,
    )


def get_exposure(device: str | Path) -> int:
    result = subprocess.run(
        ["v4l2-ctl", "-d", str(device), "--get-ctrl=exposure_time_absolute"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().split(":")[-1])
