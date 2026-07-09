"""
Label reading. Primary path: decode the barcode (checksummed, effectively never
wrong). Fallback: OCR (optional, only if enabled and installed).

Design rules from the research pass:
  - Read the CODE, not the text. A wrong-but-confident read is the worst failure
    a sorter can have; barcodes are self-validating.
  - Localize-then-decode + active re-imaging matters more than model choice.
  - Multi-scale upscaling recovers marginal codes near the pixels-per-module floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import zxingcpp
    _HAVE_ZXING = True
except Exception:  # pragma: no cover
    _HAVE_ZXING = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


@dataclass
class ReadResult:
    identity: str | None = None            # chosen part number
    method: str | None = None              # "barcode" | "ocr" | None
    codes: list[dict[str, Any]] = field(default_factory=list)  # all decoded codes
    ok: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _to_rgb(image: Any) -> np.ndarray:
    """Accept a BGR np array (cv2), grayscale np array, or PIL image → RGB np."""
    if image is None:
        raise ValueError("no image")
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.ndim == 2:
        if _HAVE_CV2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[2] == 3 and _HAVE_CV2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class BarcodeReader:
    def __init__(self, part_pattern: str = r"^[A-Z0-9][A-Z0-9\-]{4,}$",
                 upscale_retries: list[float] | None = None,
                 use_ocr: bool = False):
        self.pattern = re.compile(part_pattern)
        self.upscale_retries = upscale_retries or [1.0, 1.5, 2.0]
        self.use_ocr = use_ocr
        self._ocr = None  # lazy

    # -- barcode -------------------------------------------------------

    def decode_barcodes(self, image: Any) -> list[dict[str, Any]]:
        """Return every valid barcode found, trying progressive upscales."""
        if not _HAVE_ZXING:
            return []
        rgb = _to_rgb(image)
        seen: dict[str, dict[str, Any]] = {}
        for scale in self.upscale_retries:
            img = rgb
            if scale != 1.0 and _HAVE_CV2:
                img = cv2.resize(rgb, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            try:
                results = zxingcpp.read_barcodes(img)
            except Exception:
                results = []
            for r in results:
                if not getattr(r, "valid", False):
                    continue
                text = r.text
                if text and text not in seen:
                    seen[text] = {"text": text, "format": str(r.format),
                                  "valid": True, "scale": scale}
            if seen:  # got at least one clean decode; good enough
                break
        return list(seen.values())

    # -- part-number selection -----------------------------------------

    def read_part_number(self, image: Any) -> ReadResult:
        codes = self.decode_barcodes(image)
        # Prefer a code that matches the part-number pattern (e.g. MDDY14614001),
        # skipping pure-numeric ids and placeholders like "n/a".
        candidates = [c["text"] for c in codes]
        chosen = self._pick_part(candidates)
        if chosen:
            return ReadResult(identity=chosen, method="barcode", codes=codes, ok=True)

        if self.use_ocr:
            text = self._try_ocr(image)
            if text:
                return ReadResult(identity=text, method="ocr", codes=codes, ok=True)

        return ReadResult(identity=None, method=None, codes=codes, ok=False)

    def _pick_part(self, candidates: list[str]) -> str | None:
        # 1) pattern match, non-numeric preferred (part codes have letters)
        lettered = [c for c in candidates
                    if self.pattern.match(c) and any(ch.isalpha() for ch in c)]
        if lettered:
            return sorted(lettered, key=len, reverse=True)[0]
        # 2) any pattern match
        matched = [c for c in candidates if self.pattern.match(c)]
        if matched:
            return sorted(matched, key=len, reverse=True)[0]
        return None

    # -- optional OCR fallback -----------------------------------------

    def _try_ocr(self, image: Any) -> str | None:
        try:
            if self._ocr is None:
                from paddleocr import PaddleOCR  # type: ignore
                self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            rgb = _to_rgb(image)
            out = self._ocr.ocr(rgb, cls=True)
            for line in (out or []):
                for _box, (txt, _conf) in (line or []):
                    txt = re.sub(r"\s+", "", txt.upper())
                    if self.pattern.match(txt):
                        return txt
        except Exception:
            return None
        return None
