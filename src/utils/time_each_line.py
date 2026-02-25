import inspect
import sys
import time
from collections import defaultdict
from functools import wraps


def time_each_line(fn):
    src_lines, start = inspect.getsourcelines(fn)
    labels = {
        start + i: (line.strip() or "<blank>")[:40]
        for i, line in enumerate(src_lines)
        if line.strip() and not line.strip().startswith("#")
    }

    @wraps(fn)
    def wrapped(*args, **kwargs):
        line_dt = defaultdict(float)
        prev_line = None
        prev_t = time.perf_counter()

        def tracer(frame, event, arg):
            nonlocal prev_line, prev_t
            if frame.f_code is fn.__code__ and event == "line":
                now = time.perf_counter()
                if prev_line is not None:
                    line_dt[prev_line] += now - prev_t
                prev_line = frame.f_lineno
                prev_t = now
            return tracer

        prev_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            out = fn(*args, **kwargs)
        finally:
            now = time.perf_counter()
            if prev_line is not None:
                line_dt[prev_line] += now - prev_t
            sys.settrace(prev_trace)

        return out, {labels.get(n, f"L{n}"): dt for n, dt in line_dt.items()}

    return wrapped


def new_timing_stats():
    return defaultdict(lambda: {"n": 0, "sum": 0.0, "min": float("inf"), "max": 0.0})


def record_timing(stats, name: str, dt_s: float) -> None:
    s = stats[name]
    s["n"] += 1
    s["sum"] += dt_s
    s["min"] = min(s["min"], dt_s)
    s["max"] = max(s["max"], dt_s)


def format_timing(stats) -> str:
    parts = []
    for name, s in stats.items():
        if not s["n"]:
            continue
        parts.append(
            f"{name}: avg={s['sum']/s['n']*1e3:.1f}ms min={s['min']*1e3:.1f}ms max={s['max']*1e3:.1f}ms"
        )
    return " | ".join(parts)
