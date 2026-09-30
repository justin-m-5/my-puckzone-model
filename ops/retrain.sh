#!/usr/bin/env bash
# ops/retrain.sh — weekly candidate retrain. Never changes the live models.
#
#   1. Refresh the materialized feature store (model_game_features)
#   2. Train candidates/win_model.pkl and candidates/goals_model.pkl
#      (scored on the latest complete season, then refit on all seasons)
#   3. Print the live-vs-candidate report
#
# Review the report, then promote by hand:
#   PYTHONPATH=. python3 -m scripts.artifacts.promote

set -uo pipefail
export TZ=America/New_York

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODEL_DIR"
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi
export PYTHONPATH=.

LOCK="${TMPDIR:-/tmp}/puckzone-retrain.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T %Z') another retrain is in progress, skipping"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

step() {
  local name="$1"; shift
  echo "=== $(date '+%F %T %Z') $name"
  if ! "$@"; then echo "!!! $name failed; stopping"; exit 1; fi
}

echo "PuckZone weekly retrain — $(date '+%F %T %Z')"
step "materialize features" "$PY" -m scripts.materialize.run
step "train win model"      "$PY" -m scripts.train.win
step "train goals model"    "$PY" -m scripts.train.goals
step "report"               "$PY" -m scripts.artifacts.report
echo "=== $(date '+%F %T %Z') done — candidates ready for review (nothing promoted)"
