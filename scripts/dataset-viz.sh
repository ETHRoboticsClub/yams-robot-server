#!/usr/bin/env bash
set -euo pipefail

YAML=${YAML:-configs/arms.yaml}
REPO=${REPO:-ETHRC/fake4}
EPISODE=${EPISODE:-0}

uv run lerobot-dataset-viz --repo-id "$REPO" --episode-index "$EPISODE" --mode local
