"""
Per-product configs for r2.
"""
from __future__ import annotations

from dataclasses import dataclass


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"


@dataclass(frozen=True)
class OsmiumConfig:
    # sizing / limits
    position_limit: int = 80
    inventory_skew_ticks: float = 6.0       # was 4.0 in v2. pushes book flat faster, avoids pinning at limit
    take_threshold: float = 1.0
    min_edge_to_quote: float = 0.5          # B2: was 1.0. let best_bid+1/best_ask-1 be used when FV is inside the spread
    default_order_size: int = 12

    # fair-value model (EMA of mid blended with anchor)
    anchor_fv: float = 10000.0              # long-run fair value
    fv_ema_alpha: float = 0.02              # EMA decay for online FV tracker
    fv_anchor_weight: float = 0.10          # final FV = 0.10 * anchor + 0.90 * ema

    # alpha components
    spread_base: float = 16.0
    imbalance_mult: float = 4.7
    residual_mult: float = -0.03
    spread_mult: float = -0.02
    alpha_clip: float = 3.5                 # B2: was 2.5. regression shows dmid=4.76*imb, old clip truncated most signal


@dataclass(frozen=True)
class PepperConfig:
    # sizing / limits
    position_limit: int = 80
    inventory_skew_ticks: float = 6.0
    take_threshold: float = 2.0
    min_edge_to_quote: float = 1.0
    default_order_size: int = 8

    # trend model: mid ≈ intercept + slope * (t/1000)
    intercept_seed: float = 12000.0 
    slope_seed: float = 1.0
    ols_window: int = 200
    ols_warmup: int = 30
    outlier_sigma: float = 3.0
    residual_std: float = 1.4

    # alpha
    residual_mult: float = -0.75
    alpha_clip: float = 4.0
    strong_z: float = 1.5

    # Inventory reservation is skewed from `target_inventory`
    # instead of from zero, so the maker book leans short above target and long
    # below, cancelling net drag on trend-capture.
    target_inventory: int = 15

# Day is 1,000,000 timestamps (0..999,900 step 100). Flatten in the last 3%.
LATE_TS = 970_000
ADVERSE_SHIFT_TICKS = 0.75
ADVERSE_MAX_SHIFT = 2.0

OSMIUM_CONFIG = OsmiumConfig()
PEPPER_CONFIG = PepperConfig()

PRODUCT_CONFIG = {
    OSMIUM: OSMIUM_CONFIG,
    PEPPER: PEPPER_CONFIG,
}
