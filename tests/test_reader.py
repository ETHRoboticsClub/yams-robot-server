"""Reader tests. Proves the real Bühler label decodes, and round-trips a
generated Code128 through the real reader."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorting.reader import BarcodeReader          # noqa: E402
from sorting.perception import render_label       # noqa: E402

LABEL = Path(__file__).resolve().parent / "sample-label.jpg"


def test_real_label_decodes():
    import cv2
    assert LABEL.exists(), "sample-label.jpg missing"
    img = cv2.imread(str(LABEL))
    reader = BarcodeReader()
    codes = [c["text"] for c in reader.decode_barcodes(img)]
    assert "MDDY14614001" in codes, codes


def test_real_label_part_number():
    import cv2
    img = cv2.imread(str(LABEL))
    reader = BarcodeReader()
    res = reader.read_part_number(img)
    assert res.ok and res.identity == "MDDY14614001", res


def test_generated_label_roundtrip():
    reader = BarcodeReader()
    for part in ["MDKR20871003", "MDGH11902005", "MDZK88031004"]:
        label = render_label(part, "demo")
        res = reader.read_part_number(label)
        assert res.ok and res.identity == part, (part, res)


if __name__ == "__main__":
    test_real_label_decodes()
    test_real_label_part_number()
    test_generated_label_roundtrip()
    print("reader tests: OK")
