#!/usr/bin/env bash

PYTHONPATH=src uv run python scripts/teleop_control.py 2>/dev/null &
CONTROL_PID=$!
trap 'kill $CONTROL_PID 2>/dev/null || true' EXIT INT TERM

bash scripts/record.sh "$@"
