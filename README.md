# IMC Prosperity

Trading strategies for IMC Prosperity rounds.

## Products

- `ASH_COATED_OSMIUM` — EMA-blended fair value, imbalance/residual/spread alpha
- `INTARIAN_PEPPER_ROOT` — OLS trend fair value, residual mean-reversion alpha

## Structure

```
submission.py       active submission (Round 2)
params.py           per-product config / tunable knobs
evaluate.py         data analysis
datamodel.py        exchange datamodel
backtest_sim.py     naive local replay (directional sanity check)
backtest.sh         official backtester
run_backtests.sh    runs backtest_sim.py across all ROUND_2 days
ROUND_2/            price + trade CSVs for Round 2
r1/                 archived Round 1 files
```

## Backtesting

Naive sim (fast, approximate):
```bash
bash run_backtests.sh
```

Official backtester:
```bash
bash backtest.sh
```
