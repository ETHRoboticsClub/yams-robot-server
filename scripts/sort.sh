#!/usr/bin/env bash
# Run the sorting pipeline. Defaults to the no-hardware demo with the dashboard.
#
#   ./scripts/sort.sh                       # demo + dashboard on :8770
#   MODE=mock ./scripts/sort.sh             # decode the real label image
#   MODE=real ./scripts/sort.sh             # RealSense + robot + ACT (M2+)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${MODE:-demo}
PORT=${PORT:-8770}
DELAY=${DELAY:-0.8}
DASH=${DASH:-true}

ARGS=(--mode "$MODE" --step-delay "$DELAY" -v --dashboard-port "$PORT")
[ "$DASH" = "true" ] && ARGS+=(--dashboard)

PYTHONPATH=. uv run python -m sorting.run "${ARGS[@]}"
