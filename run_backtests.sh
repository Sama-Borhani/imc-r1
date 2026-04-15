#!/usr/bin/env bash
set -e

echo "Running local research summary..."
python evaluate.py

echo "Running Prosperity 4 backtester on round 1 custom data..."
prosperity4btest trader.py 1 --data data --merge-pnl --out results/logs/backtest_round1.log

echo "Conservative pass with no market-trade matching..."
prosperity4btest trader.py 1 --data data --merge-pnl --match-trades none --out results/logs/backtest_round1_notrades.log

echo "Moderate pass with only worse-price market-trade matching..."
prosperity4btest trader.py 1 --data data --merge-pnl --match-trades worse --out results/logs/backtest_round1_worse.log
