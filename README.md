# IMC Prosperity Round 1

This repo is a lean Round 1 setup for:
- `ASH_COATED_OSMIUM`
- `INTARIAN_PEPPER_ROOT`

## Strategy summary

### Osmium
- Treat as fixed fair value around `10000`
- Take stale quotes first
- Then quote inside the spread
- Skew reservation price by inventory

### Pepper
- Treat fair value as:
  `intercept + timestamp / 1000`
- Estimate intercept online from robust midpoints
- Take stale quotes first
- Then quote inside the spread
- Use slightly stronger inventory control than Osmium

## Setup

1. Upload the provided CSV files into `data/`
2. Open in GitHub Codespaces or locally in VS Code
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the data analysis summary:

```bash
python evaluate.py
```

5. Install the public Prosperity 4 backtester:

```bash
pip install -U prosperity4btest
```

6. Run:

```bash
bash run_backtests.sh
```

## Current priorities
- Confirm official position limits
- Confirm exact product names in the simulator
- Backtest v1 trader
- Tune inventory skew and taker thresholds
- Compare local behavior with official site results

## Files
- `trader.py`: submission trader
- `params.py`: tunable thresholds
- `evaluate.py`: quick data research and plots
- `research_r1.ipynb`: notebook for ad hoc validation
- `run_backtests.sh`: wrapper for local runs
