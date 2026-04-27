"""Extract the topdown pose reference image from the analytics NPZ.

Run once. The output PNG is committed to the repo and shipped with the
collection server so future operators don't need the analytics directory.
"""
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("/home/ethrc/Desktop/analytics/output/frames_full/wed-tom-elias/topdown/ep000.npz")
DEST = Path(__file__).resolve().parents[1] / "outputs" / "camera_reference_images" / "topdown.png"


def main() -> None:
    data = np.load(SOURCE)
    img = data["averaged_ref"]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEST), img)
    print(f"wrote {DEST}  shape={img.shape} dtype={img.dtype}")


if __name__ == "__main__":
    main()
