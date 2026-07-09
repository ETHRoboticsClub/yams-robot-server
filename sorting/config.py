"""Config loading for the sorting pipeline. One YAML, dataclass-validated."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sorting.yaml"


@dataclass
class ReadingConfig:
    method: str = "barcode"  # "barcode" | "barcode+ocr"
    # Accept anything that looks like an industrial part code by default.
    part_pattern: str = r"^[A-Z0-9][A-Z0-9\-]{4,}$"
    upscale_retries: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    max_read_attempts: int = 4


@dataclass
class ScanConfig:
    """Real-mode identity via the fixed vertical scan camera (OCR, not barcode —
    barcodes are ~4-6x under the decode floor at mat framing). Safe because reads
    are validated against the closed parts_to_cell SKU set + an abstain threshold."""
    part_pattern: str = r"[A-Z]{2,4}-\d{4,5}-\d{3}"
    min_conf: float = 0.80
    dedup_px: int = 90
    camera: str = "realsense_color"  # RealSense color node, auto-detected
    width: int = 1280
    height: int = 720


@dataclass
class CameraConfig:
    reading_camera: str = "topdown"
    reading_width: int = 1920
    reading_height: int = 1080
    # Empirically measured decode floor: a Code128 will not decode below ~800px
    # across the label. Warn if the reading camera is configured under this.
    min_reading_width: int = 1280


@dataclass
class GridConfig:
    rows: int = 2
    cols: int = 3
    file: str = "configs/grid.yaml"


@dataclass
class TelemetryConfig:
    enabled: bool = True
    dir: str = "outputs/sorting"


@dataclass
class SortingConfig:
    backend: str = "mock"  # "mock" | "real"
    pick_mode: str = "mock"  # "mock" | "policy"
    policy_path: str | None = None
    camera: CameraConfig = field(default_factory=CameraConfig)
    reading: ReadingConfig = field(default_factory=ReadingConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    # part number (barcode payload / OCR'd part-no) -> [row, col]
    parts_to_cell: dict[str, list[int]] = field(default_factory=dict)
    # human-readable name per part, for the dashboard (optional)
    part_names: dict[str, str] = field(default_factory=dict)
    # pieces per bag ("N ST" on the label) — a fixed property of the SKU
    part_qty: dict[str, int] = field(default_factory=dict)

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def cell_of(self, identity: str) -> tuple[int, int] | None:
        rc = self.parts_to_cell.get(identity)
        return (int(rc[0]), int(rc[1])) if rc else None

    def qty_of(self, identity: str) -> int | None:
        q = self.part_qty.get(identity)
        return int(q) if q is not None else None


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    """Shallow-merge a dict onto a dataclass instance, recursing one level."""
    for key, val in (data or {}).items():
        if not hasattr(dc, key):
            continue
        cur = getattr(dc, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _merge(cur, val)
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | Path | None = None) -> SortingConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = SortingConfig()
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        _merge(cfg, raw)
    return cfg
