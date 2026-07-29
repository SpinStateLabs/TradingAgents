#!/usr/bin/env bash
# Sync the working tree from the Windows dev box to the GB10 runtime host.
#
# The GB10 is the runtime: it holds the database, the models and the market
# data, and it is where the trading loop actually runs. The Windows box is only
# an editor. This script is the bridge, and it is deliberately one-way --
# never edit on the GB10, or the next sync silently discards your changes.
#
# Usage:
#   ops/sync.sh              # sync code, then run the test suite remotely
#   ops/sync.sh --no-test    # sync only
set -euo pipefail

REMOTE_HOST="${SPINTRADER_HOST:-spinner@10.0.0.62}"
REMOTE_DIR="${SPINTRADER_REMOTE_DIR:-/home/spinner/spintrader}"
SSH_KEY="${SPINTRADER_SSH_KEY:-$HOME/.ssh/gx10_ed25519}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE_HOST")

run_tests=1
[[ "${1:-}" == "--no-test" ]] && run_tests=0

echo "==> syncing $LOCAL_DIR -> $REMOTE_HOST:$REMOTE_DIR"

# rsync is not installed on the Windows side, so a tar stream over ssh does the
# job: one round trip, no per-file latency, and the exclude list keeps the
# 99 GB of model weights and the local venv out of the pipe.
tar -C "$LOCAL_DIR" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='node_modules' \
    --exclude='assets' \
    --exclude='.env' \
    --exclude='data' \
    -czf - . | "${SSH[@]}" "mkdir -p '$REMOTE_DIR' && tar -C '$REMOTE_DIR' -xzf -"

echo "==> installing dependencies (uv sync)"
"${SSH[@]}" "cd '$REMOTE_DIR' && export PATH=\$HOME/.local/bin:\$PATH && uv venv --python 3.12 .venv 2>/dev/null; uv pip install --python .venv/bin/python -q -e '.[dev]' 2>&1 | tail -5"

if [[ $run_tests -eq 1 ]]; then
  echo "==> running tests on the GB10"
  "${SSH[@]}" "cd '$REMOTE_DIR' && .venv/bin/python -m unittest discover -s tests_spintrader -t . 2>&1 | tail -20"
fi

echo "==> done"
