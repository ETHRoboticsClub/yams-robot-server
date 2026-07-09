"""
Dependency-free dashboard server (Python stdlib only).

Serves:
  GET /              the single-page dashboard (index.html)
  GET /status.json   the pipeline's latest status (or {} if none yet)
  GET /frame.jpg     the latest camera/label frame (or 204 if none)

It only READS the telemetry files. It never writes anything the pipeline reads,
holds no locks, and its failures are isolated to the browser. Kill it anytime.

    python -m dashboard.server outputs/sorting 8770
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"


class Handler(BaseHTTPRequestHandler):
    status_dir: Path = Path(".")

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                body = INDEX.read_bytes() if INDEX.exists() else b"<h1>dashboard</h1>"
                return self._send(200, body, "text/html; charset=utf-8")
            if self.path.startswith("/status.json"):
                f = self.status_dir / "status.json"
                body = f.read_bytes() if f.exists() else b"{}"
                return self._send(200, body, "application/json")
            if self.path.startswith("/frame.jpg"):
                f = self.status_dir / "frame.jpg"
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/jpeg")
                return self._send(204, b"", "image/jpeg")
            return self._send(404, b"not found", "text/plain")
        except Exception as e:
            try:
                self._send(500, str(e).encode(), "text/plain")
            except Exception:
                pass


def serve(status_dir, port: int = 8770):
    handler = partial(Handler)
    Handler.status_dir = Path(status_dir)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"[dashboard] http://localhost:{port} · reading {status_dir}")
    httpd.serve_forever()


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/sorting"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 8770
    serve(d, p)
