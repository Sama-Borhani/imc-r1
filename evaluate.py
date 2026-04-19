from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path("ROUND_2")
FIG_DIR = Path("results/figures")
SUMMARY_DIR = Path("results/summary")

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_COL = "product"
TIMESTAMP_COL = "timestamp"

PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
DAYS = [-1, 0, 1]


def load_price_data() -> pd.DataFrame:
    dfs = []
    for day in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_2_day_{day}.csv", sep=";")
        df["day"] = day
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def valid_book_mid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["bid_price_1"] > 0) & (out["ask_price_1"] > 0)].copy()
    out["mid"] = 0.5 * (out["bid_price_1"] + out["ask_price_1"])
    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    return out


def compute_osmium_metrics(df: pd.DataFrame) -> dict:
    osm = df[df[PRODUCT_COL] == "ASH_COATED_OSMIUM"].copy()
    osm["residual_vs_10000"] = osm["mid"] - 10000
    return {
        "mean_mid_by_day": osm.groupby("day")["mid"].mean().round(4).to_dict(),
        "overall_mean_mid": float(osm["mid"].mean()),
        "std_residual_vs_10000": float(osm["residual_vs_10000"].std()),
        "mean_spread": float(osm["spread"].mean()),
        "median_spread": float(osm["spread"].median()),
    }


def compute_pepper_metrics(df: pd.DataFrame) -> dict:
    pep = df[df[PRODUCT_COL] == "INTARIAN_PEPPER_ROOT"].copy()

    X = pep[["day", "timestamp"]].copy()
    X["timestamp"] = X["timestamp"] / 1000.0
    y = pep["mid"]

    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    residuals = y - preds

    return {
        "coef_day": float(model.coef_[0]),
        "coef_timestamp_over_1000": float(model.coef_[1]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "residual_std": float(np.std(residuals)),
        "mean_mid_by_day": pep.groupby("day")["mid"].mean().round(4).to_dict(),
        "mean_spread": float(pep["spread"].mean()),
        "median_spread": float(pep["spread"].median()),
    }


def plot_mid_and_spread(df: pd.DataFrame, product: str) -> None:
    sub = df[df[PRODUCT_COL] == product].copy()

    plt.figure(figsize=(12, 5))
    for day, grp in sub.groupby("day"):
        plt.plot(grp[TIMESTAMP_COL], grp["mid"], label=f"day {day}", linewidth=0.6)
    plt.legend()
    plt.title(f"{product} mid price")
    plt.xlabel("timestamp")
    plt.ylabel("mid")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{product}_mid.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for day, grp in sub.groupby("day"):
        plt.plot(grp[TIMESTAMP_COL], grp["spread"], label=f"day {day}", linewidth=0.6)
    plt.legend()
    plt.title(f"{product} spread")
    plt.xlabel("timestamp")
    plt.ylabel("spread")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{product}_spread.png")
    plt.close()


def main() -> None:
    raw = load_price_data()
    df = valid_book_mid(raw)

    summary = {
        "osmium": compute_osmium_metrics(df),
        "pepper": compute_pepper_metrics(df),
    }

    for product in PRODUCTS:
        plot_mid_and_spread(df, product)

    out_path = SUMMARY_DIR / "round2_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved figures to {FIG_DIR}")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
