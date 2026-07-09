"""Optional operations dashboard for the sorting pipeline.

Reads the best-effort telemetry files the pipeline writes and serves a live view.
Runs as a separate process/thread; the pipeline is completely unaware of it and
runs identically whether the dashboard is up, down, or crashing.
"""
