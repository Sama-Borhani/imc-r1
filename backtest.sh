#!/usr/bin/env bash
# Run submission.py through prosperity4btest against ROUND_2 data.
#
# Usage:
#   ./backtest.sh                    # label=baseline, all days
#   ./backtest.sh my-experiment      # label=my-experiment, all days

set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:-baseline}"
ALGO="submission.py"
DATA="./data"
SRC_DATA="./ROUND_2"

mkdir -p "$DATA/round2" backtests

# Symlink ROUND_2/*.csv into data/round2/ where prosperity4btest looks.
for d in -1 0 1; do
  for kind in prices trades; do
    src="$SRC_DATA/${kind}_round_2_day_${d}.csv"
    dst="$DATA/round2/${kind}_round_2_day_${d}.csv"
    if [[ -f "$src" && ! -e "$dst" ]]; then
      ln -sf "../../${src#./}" "$dst"
    fi
  done
done

echo "Running Prosperity 4 backtester on round 2 data..."
prosperity4btest "$ALGO" 2 --data "$DATA" --merge-pnl --out "backtests/${LABEL}.log"

echo "Conservative pass with no market-trade matching..."
prosperity4btest "$ALGO" 2 --data "$DATA" --merge-pnl --match-trades none --out "backtests/${LABEL}_notrades.log"

echo "Moderate pass with only worse-price market-trade matching..."
prosperity4btest "$ALGO" 2 --data "$DATA" --merge-pnl --match-trades worse --out "backtests/${LABEL}_worse.log"
