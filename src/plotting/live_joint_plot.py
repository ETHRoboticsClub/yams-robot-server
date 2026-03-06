import json
import os
import queue
import threading
import time
from base64 import b64encode
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np


def _joint_keys(data: dict[str, Any]) -> list[str]:
    return sorted(
        k for k, v in data.items() if k.endswith('.pos') and k.startswith(('left_', 'right_'))
    )


_WEB_DIR = Path(__file__).with_name('web')
_INDEX_TEMPLATE = (_WEB_DIR / 'index.html').read_text(encoding='utf-8')
_STYLES_CSS = (_WEB_DIR / 'styles.css').read_bytes()
_APP_JS = (_WEB_DIR / 'app.js').read_bytes()


class LiveJointPlotter:
    def __init__(
        self,
        joint_keys: list[str],
        *,
        hz: float = 20.0,
        history_s: float = 20.0,
        title: str = 'Live Joint Positions',
        backend: str = 'web',
        web_port: int = 8988,
        camera_hz: float = 5.0,
        follower_joint_label_map: dict[str, str] | None = None,
        leader_joint_label_map: dict[str, str] | None = None,
        camera_label_map: dict[str, str] | None = None,
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
        self.follower_joint_label_map = follower_joint_label_map or {}
        self.leader_joint_label_map = leader_joint_label_map or {}
        self.camera_label_map = camera_label_map or {}

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[queue.Queue[bytes]] = set()
        self._clients_lock = threading.Lock()

        self._history_lock = threading.Lock()
        self._history_buffer_s = max(300.0, history_s)
        self._history_max_points = max(8, int(hz * self._history_buffer_s))
        self._history: deque[tuple[float, bytes]] = deque()

        self._control_lock = threading.Lock()
        self._control_messages: deque[dict[str, Any]] = deque(maxlen=512)

        self._last_camera_t = 0.0

        self.trajectory_dir: Path | None = None
        self.task_names: list[str] = []
        self.task_goals: dict[str, int | None] = {}
        self.session_episodes: set[tuple[str, str]] = set()  # (task, ep_num)
        self._current_task: str = 'NA'

    def process_trajectory_controls(self, trajectory: list, collecting) -> None:
        """Process any queued UI control messages for trajectory start/stop."""
        from utils.teleop_data import save_trajectory
        for msg in self.pop_control_messages():
            if msg.get('type') == 'trajectory':
                if msg.get('action') == 'start':
                    self._current_task = msg.get('task', 'NA')
                    collecting.set()
                elif msg.get('action') == 'stop' and collecting.is_set():
                    collecting.clear()
                    import logging
                    ep_dir = save_trajectory(list(trajectory), self._current_task, logging.getLogger(__name__))
                    self.session_episodes.add((self._current_task, ep_dir.name))
                    trajectory.clear()

    def _render_index(self) -> bytes:
        config = {
            'keys': self.joint_keys,
            'labels': {
                'obs': self.follower_joint_label_map,
                'act': self.leader_joint_label_map,
                'cams': self.camera_label_map,
            },
            'hz': self.hz,
            'historyS': self.history_s,
            'maxBufferPoints': max(max(8, int(self.hz * self.history_s)), int(self.hz * 300.0)),
            'tasks': self.task_names,
            'taskGoals': self.task_goals,
        }
        html = _INDEX_TEMPLATE.replace('__TITLE__', self.title).replace(
            '__CONFIG_JSON__', json.dumps(config, separators=(',', ':'))
        )
        return html.encode('utf-8')

    def start(self) -> None:
        if self.backend == 'web':
            from utils.connection import _free_port

            _free_port(self.web_port)
        plotter = self

        class Handler(BaseHTTPRequestHandler):
            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                return json.loads(raw)

            def do_GET(self):
                if self.path == '/':
                    body = plotter._render_index()
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == '/static/styles.css':
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/css; charset=utf-8')
                    self.send_header('Content-Length', str(len(_STYLES_CSS)))
                    self.end_headers()
                    self.wfile.write(_STYLES_CSS)
                    return

                if self.path == '/static/app.js':
                    body = (_WEB_DIR / 'app.js').read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'application/javascript; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == '/trajectories':
                    from utils.teleop_data import get_trajectory_metadata
                    tree: dict[str, list[dict]] = {}
                    tdir = plotter.trajectory_dir
                    if tdir and tdir.is_dir():
                        for task_dir in sorted(tdir.iterdir()):
                            if task_dir.is_dir():
                                eps = sorted(
                                    (p for p in task_dir.iterdir() if p.is_dir()),
                                    key=lambda p: int(p.name) if p.name.isdigit() else p.name,
                                )
                                tree[task_dir.name] = [
                                    {
                                        'name': p.name,
                                        'marked_bad': get_trajectory_metadata(p).get('marked_bad', False),
                                        'session': (task_dir.name, p.name) in plotter.session_episodes,
                                    }
                                    for p in eps
                                ]
                    body = json.dumps(tree, separators=(',', ':')).encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'application/json')
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
                    with plotter._history_lock:
                        history = [frame for _, frame in plotter._history]

                    try:
                        self.wfile.write(b'retry: 1000\n\n')
                        for frame in history:
                            self.wfile.write(frame)
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

            def do_POST(self):
                if self.path == '/control':
                    try:
                        message = self._read_json()
                    except Exception:
                        self.send_response(HTTPStatus.BAD_REQUEST)
                        self.end_headers()
                        return
                    wrapped = {'t': time.time(), **message}
                    with plotter._control_lock:
                        plotter._control_messages.append(wrapped)
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return

                if self.path == '/mark_bad':
                    from utils.teleop_data import set_trajectory_marked_bad
                    try:
                        msg = self._read_json()
                        set_trajectory_marked_bad(msg['task'], msg['episode'], msg.get('bad', True))
                    except Exception:
                        self.send_response(HTTPStatus.BAD_REQUEST)
                        self.end_headers()
                        return
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
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

        with self._history_lock:
            self._history.append((now, frame))
            t_min = now - self._history_buffer_s
            while self._history and (
                len(self._history) > self._history_max_points or self._history[0][0] < t_min
            ):
                self._history.popleft()

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

    def pop_control_messages(self) -> list[dict[str, Any]]:
        with self._control_lock:
            out = list(self._control_messages)
            self._control_messages.clear()
        return out

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
    follower_joint_label_map: dict[str, str] | None = None,
    leader_joint_label_map: dict[str, str] | None = None,
    camera_label_map: dict[str, str] | None = None,
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
        follower_joint_label_map=follower_joint_label_map,
        leader_joint_label_map=leader_joint_label_map,
        camera_label_map=camera_label_map,
    )
    plotter.start()
    return plotter
