#!/usr/bin/env bash
set -euo pipefail

# Launch control panel popup in background
PYTHONPATH=src uv run python scripts/teleop_control.py &
CONTROL_PID=$!
trap 'kill $CONTROL_PID 2>/dev/null || true' EXIT INT TERM

# Run recording (blocking)
bash scripts/record.sh "$@"
