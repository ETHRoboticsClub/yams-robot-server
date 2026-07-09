"""
Best-effort telemetry — the OPTIONAL visual-feedback layer.

Hard contract: nothing here may ever raise into the pipeline. Every public
method swallows all exceptions. If the disk is full, the dir is missing, or a
frame is malformed, the sorter keeps sorting. The dashboard is a separate
process that reads these files; if telemetry is off or broken, the robot is
unaffected. This is what makes the dashboard "never load-bearing".

State is written atomically (tmp file + os.replace) so a reader never sees a
half-written JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


class Telemetry:
    def __init__(self, out_dir: str | Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.dir = Path(out_dir)
        self.status_path = self.dir / "status.json"
        self.frame_path = self.dir / "frame.jpg"
        self._state: dict[str, Any] = {
            "started_at": time.time(),
            "state": "init",
            "current": None,
            "placed": [],       # list of {cell, identity, ts}
            "events": [],       # rolling log tail
            "stats": {"picked": 0, "placed": 0, "read_ok": 0, "read_fail": 0, "errors": 0},
        }
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.enabled = False
        self._flush()

    # -- public API (all swallow) --------------------------------------

    def update(self, **fields: Any) -> None:
        try:
            self._state.update(fields)
            self._state["updated_at"] = time.time()
            self._flush()
        except Exception:
            pass

    def event(self, msg: str) -> None:
        try:
            self._state.setdefault("events", []).append({"ts": time.time(), "msg": str(msg)})
            self._state["events"] = self._state["events"][-40:]
            self._flush()
        except Exception:
            pass

    def bump(self, key: str, n: int = 1) -> None:
        try:
            self._state["stats"][key] = self._state["stats"].get(key, 0) + n
            self._flush()
        except Exception:
            pass

    def mark_placed(self, cell: Any, identity: str | None) -> None:
        try:
            self._state.setdefault("placed", []).append(
                {"cell": list(cell) if cell is not None else None,
                 "identity": identity, "ts": time.time()}
            )
            self.bump("placed")
        except Exception:
            pass

    def set_frame(self, image: Any) -> None:
        """Write the latest camera/label frame as JPEG for the dashboard."""
        if not self.enabled or not _HAVE_CV2 or image is None:
            return
        try:
            import cv2
            tmp = self.frame_path.with_suffix(".tmp.jpg")
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                tmp.write_bytes(buf.tobytes())
                os.replace(tmp, self.frame_path)
        except Exception:
            pass

    # -- internals -----------------------------------------------------

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._state, fh)
            os.replace(tmp, self.status_path)
        except Exception:
            pass
