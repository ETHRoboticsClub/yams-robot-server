"""
sorting — Bühler messy-box → sorted-grid kitting pipeline (MVP).

Approach B (Hybrid, CEO-reviewed 2026-07-08): the learned policy only PICKS a
packet out of clutter. Everything else is classical and deterministic:

    perception  →  pick (learned/mock)  →  read barcode  →  decide cell  →  place

The "which cell" decision is a plain dict lookup, never a neural net. Identity
comes from a checksummed barcode (zxing-cpp), not from OCR of wrinkled text.

Runs fully in mock/demo mode with no robot and no trained policy, so the whole
loop and the dashboard are exercisable today. Real backends plug in unchanged
once the grid is calibrated and an ACT checkpoint exists.
"""

from .config import SortingConfig, load_config
from .decision import DecisionTable
from .reader import BarcodeReader, ReadResult
from .telemetry import Telemetry

__all__ = [
    "SortingConfig",
    "load_config",
    "DecisionTable",
    "BarcodeReader",
    "ReadResult",
    "Telemetry",
]
