import json
import os
import queue
import threading
import time
from base64 import b64encode
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np


def _joint_keys(data: dict[str, Any]) -> list[str]:
    return sorted(
        k for k, v in data.items() if k.endswith('.pos') and k.startswith(('left_', 'right_'))
    )


class LiveJointPlotter:
    def __init__(
        self,
        joint_keys: list[str],
        *,
        hz: float = 20.0,
        history_s: float = 20.0,
        title: str = 'Live Joint Positions',
        backend: str = 'auto',
        web_port: int = 8988,
        camera_hz: float = 5.0,
    ):
        if not joint_keys:
            raise ValueError('joint_keys must not be empty')
        if backend not in {'auto', 'web', 'gui'}:
            raise ValueError('backend must be one of: auto, web, gui')

        self.joint_keys = joint_keys
        self.hz = hz
        self.history_s = history_s
        self.title = title
        self.backend = (
            'web'
            if backend == 'auto' and os.getenv('SSH_CONNECTION')
            else ('gui' if backend == 'auto' else backend)
        )
        self.web_port = web_port
        self.camera_hz = camera_hz

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[queue.Queue[bytes]] = set()
        self._clients_lock = threading.Lock()
        self._last_camera_t = 0.0

    def _html(self) -> bytes:
        keys_json = json.dumps(self.joint_keys)
        max_points = max(8, int(self.hz * self.history_s))
        max_buffer_points = max(max_points, int(self.hz * 300.0))
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{self.title}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --sub: #6b7280;
      --grid: #e5e7eb;
      --follower: #0ea5e9;
      --leader: #ef4444;
    }}
    body {{ margin: 0; padding: 16px; background: var(--bg); color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    h1 {{ margin: 0 0 4px; font-size: 18px; }}
    .meta {{ color: var(--sub); margin-bottom: 10px; font-size: 12px; }}
    .sections {{ display: grid; gap: 12px; }}
    .section {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 10px; }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    .section-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
    .section-status {{ font-size: 12px; color: var(--sub); }}
    .halves {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .half-title {{ font-size: 12px; color: var(--sub); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }}
    .card {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 8px; }}
    .name {{ font-size: 12px; margin-bottom: 4px; }}
    canvas {{ width: 100%; height: 120px; display: block; border-radius: 4px; background: #fff; }}
    .val {{ margin-top: 4px; color: var(--sub); font-size: 11px; }}
    .badge {{ display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
    .ctrl {{ display: inline-flex; align-items: center; gap: 6px; margin-left: 10px; }}
    input[type="number"] {{ width: 72px; font: inherit; padding: 2px 4px; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}
    .camera-card img {{ width: 100%; height: auto; display: block; border-radius: 4px; background: #111827; }}
    @media (max-width: 1000px) {{ .halves {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <div class=\"meta\" id=\"status\">Connecting...</div>
  <div class=\"meta\">
    <label class=\"ctrl\">history (s)
      <input id=\"history-s\" type=\"number\" min=\"1\" step=\"1\" value=\"{self.history_s:.3g}\" />
    </label>
  </div>
  <div class=\"meta\"><span class=\"badge\" style=\"background:var(--follower)\"></span>follower <span class=\"badge\" style=\"background:var(--leader);margin-left:10px\"></span>leader</div>
  <div class=\"sections\" id=\"sections\"></div>
  <section class=\"section\">
    <div class=\"section-head\">
      <div class=\"section-title\">Cameras</div>
      <div class=\"section-status\" id=\"camera-status\">camera not connected</div>
    </div>
    <div class=\"camera-grid\" id=\"camera-grid\"></div>
  </section>

  <script>
    const keys = {keys_json};
    const hz = {self.hz};
    const maxBufferPoints = {max_buffer_points};
    const leftKeys = keys.filter((k) => k.startsWith('left_'));
    const rightKeys = keys.filter((k) => k.startsWith('right_'));
    const statusEl = document.getElementById('status');
    const sectionsEl = document.getElementById('sections');
    const historyInput = document.getElementById('history-s');
    const cameraGridEl = document.getElementById('camera-grid');
    const cameraStatusEl = document.getElementById('camera-status');
    let historyS = {self.history_s};
    let maxPoints = Math.max(8, Math.round(hz * historyS));

    class JointCard {{
      constructor(parent, key, color) {{
        this.points = [];
        this.key = key;
        this.el = document.createElement('div');
        this.el.className = 'card';
        this.el.innerHTML = `<div class=\"name\">${{key}}</div><canvas width=\"600\" height=\"220\"></canvas><div class=\"val\">value: <span>-</span></div>`;
        parent.appendChild(this.el);
        this.canvas = this.el.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.valueEl = this.el.querySelector('span');
        this.color = color;
      }}

      trim(nowT) {{
        const tMin = nowT - historyS;
        while (this.points.length > maxBufferPoints || (this.points.length && this.points[0].t < tMin)) this.points.shift();
        while (this.points.length > maxPoints) this.points.shift();
      }}

      add(t, value) {{
        if (value === undefined || value === null) return;
        this.points.push({{ t, value }});
        this.trim(t);
        this.valueEl.textContent = Number(value).toFixed(3);
      }}

      render() {{
        const ctx = this.ctx;
        const w = this.canvas.width, h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = '#e5e7eb';
        for (let i = 0; i <= 4; i++) {{
          const y = 10 + i * ((h - 20) / 4);
          ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(w - 10, y); ctx.stroke();
        }}
        if (this.points.length < 2) return;
        let minY = Infinity, maxY = -Infinity;
        for (const p of this.points) {{ minY = Math.min(minY, p.value); maxY = Math.max(maxY, p.value); }}
        const pad = Math.max(0.02, 0.1 * (maxY - minY || 1));
        minY -= pad; maxY += pad;
        const t0 = this.points[0].t;
        const t1 = this.points[this.points.length - 1].t;
        ctx.beginPath();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2;
        let moved = false;
        for (const p of this.points) {{
          const x = 10 + ((p.t - t0) / Math.max(1e-6, (t1 - t0))) * (w - 20);
          const y = 10 + (1 - (p.value - minY) / Math.max(1e-6, (maxY - minY))) * (h - 20);
          if (!moved) {{ ctx.moveTo(x, y); moved = true; }} else {{ ctx.lineTo(x, y); }}
        }}
        if (moved) ctx.stroke();
      }}
    }}

    class ArmSection {{
      constructor(parent, title, color, sourceKey) {{
        this.sourceKey = sourceKey;
        this.cards = [];
        this.lastSeenMs = 0;
        this.name = sourceKey === 'obs' ? 'follower' : 'leader';
        this.root = document.createElement('section');
        this.root.className = 'section';
        this.root.innerHTML = `
          <div class=\"section-head\">
            <div class=\"section-title\">${{title}}</div>
            <div class=\"section-status\">${{title.toLowerCase()}} not connected</div>
          </div>
          <div class=\"halves\">
            <div><div class=\"half-title\">Left</div><div class=\"grid left-grid\"></div></div>
            <div><div class=\"half-title\">Right</div><div class=\"grid right-grid\"></div></div>
          </div>`;
        parent.appendChild(this.root);
        this.statusEl = this.root.querySelector('.section-status');
        const leftGrid = this.root.querySelector('.left-grid');
        const rightGrid = this.root.querySelector('.right-grid');
        for (const key of leftKeys) this.cards.push(new JointCard(leftGrid, key, color));
        for (const key of rightKeys) this.cards.push(new JointCard(rightGrid, key, color));
      }}

      addPoint(msg) {{
        const src = msg[this.sourceKey] || {{}};
        let hasValue = false;
        for (const card of this.cards) {{
          const v = src[card.key];
          if (v !== undefined && v !== null) hasValue = true;
          card.add(msg.t, v);
        }}
        if (hasValue) this.lastSeenMs = performance.now();
      }}

      updateStatus(nowMs) {{
        this.statusEl.textContent = this.lastSeenMs > 0 && nowMs - this.lastSeenMs <= 2000
          ? 'connected'
          : `${{this.name}} not connected`;
      }}

      render() {{
        for (const card of this.cards) card.render();
      }}

      trimAll() {{
        for (const card of this.cards) {{
          if (card.points.length) card.trim(card.points[card.points.length - 1].t);
        }}
      }}
    }}

    class CameraCard {{
      constructor(parent, key) {{
        this.key = key;
        this.el = document.createElement('div');
        this.el.className = 'card camera-card';
        this.el.innerHTML = `<div class=\"name\">${{key}}</div><img alt=\"${{key}}\" />`;
        parent.appendChild(this.el);
        this.imgEl = this.el.querySelector('img');
      }}

      setFrame(dataUrl) {{
        this.imgEl.src = dataUrl;
      }}
    }}

    class CameraSection {{
      constructor(parent, statusEl) {{
        this.statusEl = statusEl;
        this.parent = parent;
        this.cards = new Map();
        this.lastSeenMs = 0;
      }}

      addPoint(msg) {{
        const cams = msg.cams || {{}};
        let hasFrame = false;
        for (const [key, frame] of Object.entries(cams)) {{
          if (!frame) continue;
          if (!this.cards.has(key)) this.cards.set(key, new CameraCard(this.parent, key));
          this.cards.get(key).setFrame(frame);
          hasFrame = true;
        }}
        if (hasFrame) this.lastSeenMs = performance.now();
      }}

      updateStatus(nowMs) {{
        this.statusEl.textContent = this.lastSeenMs > 0 && nowMs - this.lastSeenMs <= 2000
          ? 'connected'
          : 'camera not connected';
      }}
    }}

    class JointDashboard {{
      constructor() {{
        this.follower = new ArmSection(sectionsEl, 'Follower', '#0ea5e9', 'obs');
        this.leader = new ArmSection(sectionsEl, 'Leader', '#ef4444', 'act');
        this.cameras = new CameraSection(cameraGridEl, cameraStatusEl);
      }}

      addPoint(msg) {{
        this.follower.addPoint(msg);
        this.leader.addPoint(msg);
        this.cameras.addPoint(msg);
      }}

      trimAll() {{
        this.follower.trimAll();
        this.leader.trimAll();
      }}

      render() {{
        const nowMs = performance.now();
        this.follower.updateStatus(nowMs);
        this.leader.updateStatus(nowMs);
        this.cameras.updateStatus(nowMs);
        this.follower.render();
        this.leader.render();
      }}
    }}

    const dashboard = new JointDashboard();
    const es = new EventSource('/events');
    es.onopen = () => statusEl.textContent = 'Connected';
    es.onerror = () => statusEl.textContent = 'Disconnected; retrying...';
    es.onmessage = (evt) => dashboard.addPoint(JSON.parse(evt.data));
    historyInput.onchange = () => {{
      const next = Number(historyInput.value);
      if (!Number.isFinite(next) || next <= 0) {{
        historyInput.value = historyS.toFixed(2);
        return;
      }}
      historyS = next;
      maxPoints = Math.max(8, Math.round(hz * historyS));
      dashboard.trimAll();
    }};
    function tick() {{
      dashboard.render();
      requestAnimationFrame(tick);
    }}
    tick();
  </script>
</body>
</html>
""".encode('utf-8')

    def start(self) -> None:
        if self.backend == 'web':
            from utils.connection import _free_port

            _free_port(self.web_port)

        plotter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    body = plotter._html()
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == '/events':
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    client_q: queue.Queue[bytes] = queue.Queue(maxsize=128)
                    with plotter._clients_lock:
                        plotter._clients.add(client_q)

                    try:
                        self.wfile.write(b'retry: 1000\n\n')
                        self.wfile.flush()
                        while not plotter._stop.is_set():
                            try:
                                msg = client_q.get(timeout=1.0)
                            except queue.Empty:
                                self.wfile.write(b': ping\n\n')
                                self.wfile.flush()
                                continue
                            self.wfile.write(msg)
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        with plotter._clients_lock:
                            plotter._clients.discard(client_q)
                    return

                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(('127.0.0.1', self.web_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if self.backend == 'web':
            self.debug_webagg()

    def debug_webagg(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with urlopen(f'http://127.0.0.1:{self.web_port}/', timeout=1) as r:
                    print(f'[stream] GET / -> {r.status}')
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
        print(f'[stream] open http://127.0.0.1:{self.web_port}/')

    def push(self, observation: dict[str, Any] | None = None, action: dict[str, Any] | None = None) -> None:
        obs = {k: float(observation[k]) for k in self.joint_keys if observation and k in observation}
        act = {k: float(action[k]) for k in self.joint_keys if action and k in action}
        now = time.monotonic()
        cams: dict[str, str] = {}
        if observation and self.camera_hz > 0 and now - self._last_camera_t >= 1.0 / self.camera_hz:
            for key, value in observation.items():
                if key in self.joint_keys or not isinstance(value, np.ndarray) or value.ndim != 3:
                    continue
                frame = value
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    cams[key] = f"data:image/jpeg;base64,{b64encode(encoded).decode('ascii')}"
            self._last_camera_t = now
        payload = json.dumps({'t': now, 'obs': obs, 'act': act, 'cams': cams}, separators=(',', ':'))
        frame = f'data: {payload}\n\n'.encode('utf-8')

        with self._clients_lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def start_joint_plotter(
    bi_follower: Any,
    *,
    hz: float = 20.0,
    history_s: float = 20.0,
    title: str = 'Live Joint Positions',
    backend: str = 'auto',
    web_port: int = 8988,
    camera_hz: float = 5.0,
) -> LiveJointPlotter:
    keys = _joint_keys(bi_follower.get_observation(with_cameras=False))
    if not keys:
        raise ValueError('No joint position keys found in follower observation.')
    plotter = LiveJointPlotter(
        keys,
        hz=hz,
        history_s=history_s,
        title=title,
        backend=backend,
        web_port=web_port,
        camera_hz=camera_hz,
    )
    plotter.start()
    return plotter
