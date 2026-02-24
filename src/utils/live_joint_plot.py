from collections import deque
from math import ceil
from multiprocessing import Event, Process, Queue
import os
from queue import Empty, Full
import time
from typing import Any
from urllib.request import urlopen


def _joint_keys(data: dict[str, Any]) -> list[str]:
    return sorted(
        k for k, v in data.items() if k.endswith(".pos") and k.startswith(("left_", "right_"))
    )


def _plot_worker(
    queue: Queue,
    stop: Event,
    keys: list[str],
    hz: float,
    history_s: float,
    title: str,
    backend: str,
    web_port: int,
) -> None:
    import matplotlib
    if backend == "web":
        matplotlib.use("WebAgg", force=True)
        matplotlib.rcParams["webagg.address"] = "127.0.0.1"
        matplotlib.rcParams["webagg.port"] = web_port
        matplotlib.rcParams["webagg.open_in_browser"] = False
        print(f"Live joint plot: http://127.0.0.1:{web_port}")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n = max(1, int(hz * history_s))
    ts = deque([0.0] * n, maxlen=n)
    follower = {k: deque([0.0] * n, maxlen=n) for k in keys}
    leader = {k: deque([0.0] * n, maxlen=n) for k in keys}

    cols = 4
    rows = ceil(len(keys) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.5), sharex=True)
    axes = list(axs.flat) if hasattr(axs, "flat") else [axs]
    lines = {}

    for i, k in enumerate(keys):
        ax = axes[i]
        lf, = ax.plot([], [], lw=1.3, label="follower")
        ll, = ax.plot([], [], lw=1.1, alpha=0.85, label="leader")
        ax.set_title(k, fontsize=9)
        ax.grid(alpha=0.25)
        lines[k] = (lf, ll, ax)

    for ax in axes[len(keys):]:
        ax.set_visible(False)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()

    dt = 1.0 / hz

    def update(_):
        if stop.is_set():
            plt.close(fig)
            return []

        latest = None
        while True:
            try:
                latest = queue.get_nowait()
            except Empty:
                break

        if latest is None:
            return [line for pair in lines.values() for line in pair[:2]]

        obs, act = latest
        ts.append(ts[-1] + dt)

        for k in keys:
            follower[k].append(obs.get(k, follower[k][-1]))
            leader[k].append(act.get(k, leader[k][-1]))
            x = list(ts)
            y_f = list(follower[k])
            y_l = list(leader[k])
            lf, ll, ax = lines[k]
            lf.set_data(x, y_f)
            ll.set_data(x, y_l)
            y_min = min(min(y_f), min(y_l))
            y_max = max(max(y_f), max(y_l))
            pad = max(0.02, 0.1 * (y_max - y_min))
            ax.set_xlim(x[0], x[-1] if x[-1] > x[0] else x[0] + dt)
            ax.set_ylim(y_min - pad, y_max + pad)

        return [line for pair in lines.values() for line in pair[:2]]

    ani = FuncAnimation(fig, update, interval=int(1000 / hz), cache_frame_data=False)
    fig._ani = ani
    plt.show()


class LiveJointPlotter:
    def __init__(
        self,
        joint_keys: list[str],
        *,
        hz: float = 20.0,
        history_s: float = 20.0,
        title: str = "Live Joint Positions",
        backend: str = "auto",
        web_port: int = 8988,
    ):
        if not joint_keys:
            raise ValueError("joint_keys must not be empty")
        if backend not in {"auto", "web", "gui"}:
            raise ValueError("backend must be one of: auto, web, gui")
        self.joint_keys = joint_keys
        self.hz = hz
        self.history_s = history_s
        self.title = title
        self.backend = (
            "web"
            if backend == "auto" and os.getenv("SSH_CONNECTION")
            else ("gui" if backend == "auto" else backend)
        )
        self.web_port = web_port
        self._queue = Queue(maxsize=8)
        self._stop = Event()
        self._proc = Process(
            target=_plot_worker,
            args=(
                self._queue,
                self._stop,
                self.joint_keys,
                self.hz,
                self.history_s,
                self.title,
                self.backend,
                self.web_port,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def debug_webagg(self, timeout_s: float = 5.0) -> None:
        if self.backend != "web":
            return
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{self.web_port}/", timeout=1) as r:
                    print(f"[webagg] GET / -> {r.status}")
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
        for path in ["/js/mpl.js", "/_static/js/mpl_tornado.js"]:
            with urlopen(f"http://127.0.0.1:{self.web_port}{path}", timeout=2) as r:
                print(f"[webagg] GET {path} -> {r.status}")
        print("[webagg] waiting for browser websocket on /1/ws")

    def push(self, observation: dict[str, Any], action: dict[str, Any] | None = None) -> None:
        obs = {k: float(observation[k]) for k in self.joint_keys if k in observation}
        act = {k: float(action[k]) for k in self.joint_keys if action and k in action}
        try:
            self._queue.put_nowait((obs, act))
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait((obs, act))

    def close(self) -> None:
        self._stop.set()
        if self._proc.is_alive():
            self._proc.join(timeout=1.0)
        if self._proc.is_alive():
            self._proc.terminate()


def start_joint_plotter(
    bi_follower: Any,
    *,
    hz: float = 20.0,
    history_s: float = 20.0,
    title: str = "Live Joint Positions",
    backend: str = "auto",
    web_port: int = 8988,
) -> LiveJointPlotter:
    keys = _joint_keys(bi_follower.get_observation())
    if not keys:
        raise ValueError("No joint position keys found in follower observation.")
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
