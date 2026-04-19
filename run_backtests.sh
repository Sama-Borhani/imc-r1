#!/usr/bin/env bash
#
# Naive-sim replay of submission.py on all ROUND_2 days. Directional sanity
# only — NOT the official prosperity matcher (see r1/notes.md for why the
# numbers under-count fills).
#
# For final sign-off, use ./backtest.sh (invokes prosperity3bt).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/myvenv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$HERE/.venv/bin/python"
fi
DATA="$HERE/ROUND_2"
TRADER="$HERE/submission.py"

echo "=== submission.py on ROUND_2 ==="
for d in -1 0 1; do
  echo "--- day $d ---"
  "$PY" "$HERE/backtest_sim.py" "$TRADER" "$DATA/prices_round_2_day_${d}.csv"
done
