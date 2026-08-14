#!/usr/bin/env bash
# One-shot GPU smoke test for nvidia/Cosmos3-Edge-Policy-DROID served by
# cosmos-framework's official openpi-protocol WebSocket server.
#
# Run on a GPU node (from the behavior-cosmos repo root):
#   srun -p h100 --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:40:00 \
#     bash scripts/smoke_droid_edge.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSMOS_DIR="$REPO_ROOT/cosmos-framework"
PY="$COSMOS_DIR/.venv/bin/python"
CKPT="$REPO_ROOT/checkpoints/Cosmos3-Edge-Policy-DROID"
PORT="${PORT:-8901}"

[ -x "$PY" ] || { echo "cosmos-framework venv missing — run: cd cosmos-framework && uv sync --all-extras --group=cu128-train --group=policy-server"; exit 1; }
[ -d "$CKPT" ] || { echo "checkpoint missing at $CKPT"; exit 1; }

echo "Starting Cosmos3-Edge-Policy-DROID server on port $PORT..."
(
  cd "$COSMOS_DIR"
  PYTHONPATH="$COSMOS_DIR" exec "$PY" -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path "$CKPT" \
    --format-prompt-as-json True \
    --host 127.0.0.1 \
    --port "$PORT"
) &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

PYTHONPATH="$COSMOS_DIR" "$PY" "$REPO_ROOT/scripts/smoke_droid_client.py" --port "$PORT" --timeout 900
