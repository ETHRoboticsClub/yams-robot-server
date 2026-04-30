#!/usr/bin/env bash
set -euo pipefail

SESSION=teleop-ui

if ! command -v tmux &>/dev/null; then
    echo "tmux is required: sudo apt install tmux"
    exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "==> Running setup check..."
PYTHONPATH=src uv run python scripts/check_setup.py || exit 1
echo

tmux new-session -d -s "$SESSION"

PANE_MAIN=$(tmux display-message -p -t "$SESSION:0" '#{pane_id}')
PANE_VIS_R=$(tmux split-window -h -p 30 -d -t "$PANE_MAIN" -P -F '#{pane_id}')
PANE_VIS_L=$(tmux split-window -v -d -t "$PANE_VIS_R" -P -F '#{pane_id}')

# Teleop — main pane
tmux send-keys -t "$PANE_MAIN" "bash scripts/teleop.sh" Enter

# MuJoCo arm visualizers — wait for portal servers to be ready
tmux send-keys -t "$PANE_VIS_R" "sleep 12 && uv run python examples/vis.py right" Enter
tmux send-keys -t "$PANE_VIS_L" "sleep 12 && uv run python examples/vis.py left" Enter

# Control panel GUI — pops up as a separate window
(sleep 8 && PYTHONPATH=src uv run python scripts/teleop_control.py) &

tmux attach-session -t "$SESSION"
