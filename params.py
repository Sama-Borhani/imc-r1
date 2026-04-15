from dataclasses import dataclass

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"


@dataclass(frozen=True)
class ProductConfig:
    position_limit: int
    inventory_skew_ticks: float
    take_threshold: float
    reversion_threshold: float
    min_edge_to_quote: float
    default_order_size: int


# Official Prosperity 4 Round 1 limits in the public backtester are 80 for both products.
PRODUCT_CONFIG = {
    OSMIUM: ProductConfig(
        position_limit=80,
        inventory_skew_ticks=4.0,
        take_threshold=1.0,
        reversion_threshold=0.0,
        min_edge_to_quote=1.0,
        default_order_size=10,
    ),
    PEPPER: ProductConfig(
        position_limit=80,
        inventory_skew_ticks=5.0,
        take_threshold=2.0,
        reversion_threshold=0.0,
        min_edge_to_quote=1.0,
        default_order_size=10,
    ),
}


OSMIUM_FAIR_VALUE = 10000

# Pepper fair value model:
# fair_value ~= intercept + timestamp / 1000
# Initialize with historical fit, then update online.
PEPPER_INITIAL_INTERCEPT = 12000.0
PEPPER_BLEND_TREND = 0.8
PEPPER_BLEND_POPULAR_MID = 0.2
PEPPER_INTERCEPT_WINDOW = 80


def inventory_adjustment(position: int, position_limit: int, skew_ticks: float) -> float:
    if position_limit <= 0:
        return 0.0
    return skew_ticks * (position / position_limit)
