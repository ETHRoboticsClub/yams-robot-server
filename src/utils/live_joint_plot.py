import json
import os
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import urlopen


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

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[queue.Queue[bytes]] = set()
        self._clients_lock = threading.Lock()

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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }}
    .card {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 8px; }}
    .name {{ font-size: 12px; margin-bottom: 4px; }}
    canvas {{ width: 100%; height: 120px; display: block; border-radius: 4px; background: #fff; }}
    .vals {{ margin-top: 4px; color: var(--sub); font-size: 11px; }}
    .badge {{ display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
    .ctrl {{ display: inline-flex; align-items: center; gap: 6px; margin-left: 10px; }}
    input[type="number"] {{ width: 72px; font: inherit; padding: 2px 4px; }}
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
  <div class=\"grid\" id=\"grid\"></div>

  <script>
    const keys = {keys_json};
    const hz = {self.hz};
    const maxBufferPoints = {max_buffer_points};
    let historyS = {self.history_s};
    let maxPoints = Math.max(8, Math.round(hz * historyS));
    const state = new Map();
    const canvases = new Map();
    const statusEl = document.getElementById('status');
    const grid = document.getElementById('grid');
    const historyInput = document.getElementById('history-s');

    for (const key of keys) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<div class=\"name\">${{key}}</div><canvas width=\"600\" height=\"220\"></canvas><div class=\"vals\">obs: <span class=\"obs\">-</span> act: <span class=\"act\">-</span></div>`;
      grid.appendChild(card);
      state.set(key, []);
      canvases.set(key, {{
        canvas: card.querySelector('canvas'),
        obs: card.querySelector('.obs'),
        act: card.querySelector('.act')
      }});
    }}

    function trim(arr, nowT) {{
      const tMin = nowT - historyS;
      while (arr.length > maxBufferPoints || (arr.length && arr[0].t < tMin)) arr.shift();
      while (arr.length > maxPoints) arr.shift();
    }}

    function addPoint(msg) {{
      for (const key of keys) {{
        const arr = state.get(key);
        arr.push({{ t: msg.t, obs: msg.obs[key], act: msg.act[key] }});
        trim(arr, msg.t);
        const ui = canvases.get(key);
        if (msg.obs[key] !== undefined) ui.obs.textContent = Number(msg.obs[key]).toFixed(3);
        if (msg.act[key] !== undefined) ui.act.textContent = Number(msg.act[key]).toFixed(3);
      }}
    }}

    function drawLine(ctx, points, x0, y0, w, h, t0, t1, minY, maxY, color, key) {{
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      let moved = false;
      for (const p of points) {{
        const v = p[key];
        if (v === undefined || v === null) continue;
        const x = x0 + ((p.t - t0) / Math.max(1e-6, (t1 - t0))) * w;
        const y = y0 + h - ((v - minY) / Math.max(1e-6, (maxY - minY))) * h;
        if (!moved) {{ ctx.moveTo(x, y); moved = true; }} else {{ ctx.lineTo(x, y); }}
      }}
      if (moved) ctx.stroke();
    }}

    function render() {{
      for (const key of keys) {{
        const pts = state.get(key);
        const ui = canvases.get(key);
        const ctx = ui.canvas.getContext('2d');
        const w = ui.canvas.width, h = ui.canvas.height;
        ctx.clearRect(0, 0, w, h);

        ctx.strokeStyle = '#e5e7eb';
        for (let i = 0; i <= 4; i++) {{
          const y = 10 + i * ((h - 20) / 4);
          ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(w - 10, y); ctx.stroke();
        }}

        if (pts.length < 2) continue;

        let minY = Infinity, maxY = -Infinity;
        for (const p of pts) {{
          if (p.obs !== undefined) {{ minY = Math.min(minY, p.obs); maxY = Math.max(maxY, p.obs); }}
          if (p.act !== undefined) {{ minY = Math.min(minY, p.act); maxY = Math.max(maxY, p.act); }}
        }}
        if (!isFinite(minY)) {{ minY = -1; maxY = 1; }}
        const pad = Math.max(0.02, 0.1 * (maxY - minY || 1));
        minY -= pad;
        maxY += pad;

        const t0 = pts[0].t;
        const t1 = pts[pts.length - 1].t;
        drawLine(ctx, pts, 10, 10, w - 20, h - 20, t0, t1, minY, maxY, '#0ea5e9', 'obs');
        drawLine(ctx, pts, 10, 10, w - 20, h - 20, t0, t1, minY, maxY, '#ef4444', 'act');
      }}
      requestAnimationFrame(render);
    }}

    const es = new EventSource('/events');
    es.onopen = () => statusEl.textContent = 'Connected';
    es.onerror = () => statusEl.textContent = 'Disconnected; retrying...';
    es.onmessage = (evt) => addPoint(JSON.parse(evt.data));
    historyInput.onchange = () => {{
      const next = Number(historyInput.value);
      if (!Number.isFinite(next) || next <= 0) {{
        historyInput.value = historyS.toFixed(2);
        return;
      }}
      historyS = next;
      maxPoints = Math.max(8, Math.round(hz * historyS));
      for (const arr of state.values()) {{
        if (arr.length) trim(arr, arr[arr.length - 1].t);
      }}
    }};
    render();
  </script>
</body>
</html>
""".encode('utf-8')

    def start(self) -> None:
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

    def push(self, observation: dict[str, Any], action: dict[str, Any] | None = None) -> None:
        obs = {k: float(observation[k]) for k in self.joint_keys if k in observation}
        act = {k: float(action[k]) for k in self.joint_keys if action and k in action}
        payload = json.dumps({'t': time.monotonic(), 'obs': obs, 'act': act}, separators=(',', ':'))
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
    )
    plotter.start()
    return plotter
