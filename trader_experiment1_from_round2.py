
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"


@dataclass(frozen=True)
class ProductConfig:
    position_limit: int
    inventory_skew_ticks: float
    take_threshold: float
    min_edge_to_quote: float
    default_order_size: int


PRODUCT_CONFIG = {
    OSMIUM: ProductConfig(
        position_limit=80,
        inventory_skew_ticks=4.0,
        take_threshold=1.0,
        min_edge_to_quote=1.0,
        default_order_size=12,
    ),
    PEPPER: ProductConfig(
        position_limit=80,
        inventory_skew_ticks=6.0,
        take_threshold=2.0,
        min_edge_to_quote=1.0,
        default_order_size=8,
    ),
}

# -----------------------------
# Baseline model constants
# -----------------------------
OSMIUM_FAIR_VALUE = 10000

PEPPER_INITIAL_INTERCEPT = 12000.0
PEPPER_INTERCEPT_WINDOW = 50
PEPPER_RESIDUAL_STD = 2.2

OSMIUM_SPREAD_BASE = 16.0
OSMIUM_IMBALANCE_MULT = 4.7
OSMIUM_RESIDUAL_MULT = -0.03
OSMIUM_SPREAD_MULT = -0.02
OSMIUM_ALPHA_CLIP = 2.5

PEPPER_RESIDUAL_MULT = -0.75
PEPPER_ALPHA_CLIP = 4.0
PEPPER_STRONG_Z = 1.0

# -----------------------------
# Experiment 1: PEPPER short asymmetry
# -----------------------------
MAF_BID = 3000

PEPPER_LONG_THRESHOLD_MULT = 1.00
PEPPER_SHORT_THRESHOLD_MULT = 1.25
PEPPER_SHORT_SIZE_MULT = 0.65

# If PEPPER is locally rising, make shorts more selective and stop passive
# sell quoting during mild/medium short signals.
PEPPER_SHORT_TREND_BLOCK = 1.5
PEPPER_SHORT_TREND_EXTRA_THRESHOLD = 0.75

# Light inventory safety so the experiment does not behave stupidly near limits.
PEPPER_EMERGENCY_POSITION = 60
PEPPER_EMERGENCY_UNWIND_THRESHOLD = 0.5

PEPPER_MID_HISTORY_WINDOW = 8
PEPPER_SLOPE_LOOKBACK = 3


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def inventory_adjustment(position: int, position_limit: int, skew_ticks: float) -> float:
    if position_limit <= 0:
        return 0.0
    return skew_ticks * (position / position_limit)


@dataclass
class ProductMemory:
    pepper_intercept_samples: List[float] = field(default_factory=list)
    pepper_intercept_estimate: float = PEPPER_INITIAL_INTERCEPT
    pepper_mid_history: List[float] = field(default_factory=list)


class Trader:
    def __init__(self) -> None:
        self.memory = ProductMemory()

    def bid(self) -> int:
        return MAF_BID

    def load_memory(self, trader_data: str) -> None:
        if not trader_data:
            self.memory = ProductMemory()
            return

        try:
            raw = json.loads(trader_data)
            self.memory = ProductMemory(
                pepper_intercept_samples=raw.get("pepper_intercept_samples", []),
                pepper_intercept_estimate=float(
                    raw.get("pepper_intercept_estimate", PEPPER_INITIAL_INTERCEPT)
                ),
                pepper_mid_history=[float(x) for x in raw.get("pepper_mid_history", [])][-PEPPER_MID_HISTORY_WINDOW:],
            )
        except Exception:
            self.memory = ProductMemory()

    def dump_memory(self) -> str:
        payload = {
            "pepper_intercept_samples": self.memory.pepper_intercept_samples[-PEPPER_INTERCEPT_WINDOW:],
            "pepper_intercept_estimate": self.memory.pepper_intercept_estimate,
            "pepper_mid_history": self.memory.pepper_mid_history[-PEPPER_MID_HISTORY_WINDOW:],
        }
        return json.dumps(payload, separators=(",", ":"))

    def run(self, state: TradingState):
        self.load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product == OSMIUM:
                result[product] = self.trade_osmium(state, order_depth)
            elif product == PEPPER:
                result[product] = self.trade_pepper(state, order_depth)

        conversions = 0
        trader_data = self.dump_memory()
        return result, conversions, trader_data

    def trade_osmium(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        config = PRODUCT_CONFIG[OSMIUM]
        position = state.position.get(OSMIUM, 0)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        mid = self.get_mid(best_bid, best_ask)
        spread = self.get_spread(best_bid, best_ask)
        imbalance = self.get_top_imbalance(best_bid_volume, best_ask_volume)

        alpha = 0.0
        if imbalance is not None:
            alpha += OSMIUM_IMBALANCE_MULT * imbalance
        if mid is not None:
            alpha += OSMIUM_RESIDUAL_MULT * (mid - OSMIUM_FAIR_VALUE)
        if spread is not None:
            alpha += OSMIUM_SPREAD_MULT * (spread - OSMIUM_SPREAD_BASE)

        alpha = clip(alpha, -OSMIUM_ALPHA_CLIP, OSMIUM_ALPHA_CLIP)

        reservation = (
            OSMIUM_FAIR_VALUE
            + alpha
            - inventory_adjustment(position, config.position_limit, config.inventory_skew_ticks)
        )

        buy_size = config.default_order_size
        sell_size = config.default_order_size

        if imbalance is not None and abs(imbalance) >= 0.4:
            if imbalance > 0:
                buy_size = min(20, config.default_order_size + 4)
                sell_size = max(6, config.default_order_size - 2)
            else:
                sell_size = min(20, config.default_order_size + 4)
                buy_size = max(6, config.default_order_size - 2)

        return self.make_orders(
            product=OSMIUM,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            buy_take_threshold=config.take_threshold,
            sell_take_threshold=config.take_threshold,
            allow_passive_buy=True,
            allow_passive_sell=True,
        )

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        config = PRODUCT_CONFIG[PEPPER]
        position = state.position.get(PEPPER, 0)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        popular_mid = self.get_popular_mid(order_depth)
        raw_mid = self.get_mid(best_bid, best_ask)
        signal_mid = popular_mid if popular_mid is not None else raw_mid

        trend_fv = self.estimate_pepper_trend(state.timestamp, signal_mid)

        if raw_mid is not None:
            self.record_pepper_mid(raw_mid)

        residual = 0.0
        if raw_mid is not None:
            residual = raw_mid - trend_fv

        z_score = residual / PEPPER_RESIDUAL_STD if PEPPER_RESIDUAL_STD > 0 else 0.0
        residual_alpha = clip(PEPPER_RESIDUAL_MULT * residual, -PEPPER_ALPHA_CLIP, PEPPER_ALPHA_CLIP)

        reservation = (
            trend_fv
            + residual_alpha
            - inventory_adjustment(position, config.position_limit, config.inventory_skew_ticks)
        )

        local_slope = self.get_pepper_local_slope()

        buy_size = config.default_order_size
        sell_size = config.default_order_size
        allow_passive_buy = True
        allow_passive_sell = True

        buy_take_threshold = config.take_threshold * PEPPER_LONG_THRESHOLD_MULT
        sell_take_threshold = config.take_threshold * PEPPER_SHORT_THRESHOLD_MULT

        # Long side: leave close to original behavior.
        if z_score <= -PEPPER_STRONG_Z:
            buy_size = min(16, config.default_order_size + 4)
            sell_size = max(4, config.default_order_size - 2)
            allow_passive_sell = False

        # Short side: stricter entry, smaller size, and block passive sells if
        # price is still locally rising.
        if z_score >= PEPPER_STRONG_Z:
            sell_size = max(1, int(round(config.default_order_size * PEPPER_SHORT_SIZE_MULT)))
            buy_size = max(4, config.default_order_size - 2)
            allow_passive_buy = False

            if local_slope is not None and local_slope > PEPPER_SHORT_TREND_BLOCK:
                sell_take_threshold += PEPPER_SHORT_TREND_EXTRA_THRESHOLD
                allow_passive_sell = False

        # Emergency unwind when PEPPER inventory gets too large.
        if position >= PEPPER_EMERGENCY_POSITION:
            allow_passive_buy = False
            sell_take_threshold = min(sell_take_threshold, PEPPER_EMERGENCY_UNWIND_THRESHOLD)
            sell_size = min(config.position_limit + position, max(sell_size, 16))
        elif position <= -PEPPER_EMERGENCY_POSITION:
            allow_passive_sell = False
            buy_take_threshold = min(buy_take_threshold, PEPPER_EMERGENCY_UNWIND_THRESHOLD)
            buy_size = min(config.position_limit - position, max(buy_size, 16))

        return self.make_orders(
            product=PEPPER,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            buy_take_threshold=buy_take_threshold,
            sell_take_threshold=sell_take_threshold,
            allow_passive_buy=allow_passive_buy,
            allow_passive_sell=allow_passive_sell,
        )

    def estimate_pepper_trend(
        self,
        timestamp: int,
        signal_mid: Optional[float],
    ) -> float:
        if signal_mid is not None:
            intercept_candidate = signal_mid - (timestamp / 1000.0)
            self.memory.pepper_intercept_samples.append(intercept_candidate)
            self.memory.pepper_intercept_samples = self.memory.pepper_intercept_samples[-PEPPER_INTERCEPT_WINDOW:]

            sorted_samples = sorted(self.memory.pepper_intercept_samples)
            n = len(sorted_samples)
            if n > 0:
                if n % 2 == 1:
                    self.memory.pepper_intercept_estimate = sorted_samples[n // 2]
                else:
                    self.memory.pepper_intercept_estimate = 0.5 * (
                        sorted_samples[n // 2 - 1] + sorted_samples[n // 2]
                    )

        return self.memory.pepper_intercept_estimate + (timestamp / 1000.0)

    def record_pepper_mid(self, raw_mid: float) -> None:
        self.memory.pepper_mid_history.append(float(raw_mid))
        self.memory.pepper_mid_history = self.memory.pepper_mid_history[-PEPPER_MID_HISTORY_WINDOW:]

    def get_pepper_local_slope(self) -> Optional[float]:
        history = self.memory.pepper_mid_history
        if len(history) <= PEPPER_SLOPE_LOOKBACK:
            return None
        return history[-1] - history[-1 - PEPPER_SLOPE_LOOKBACK]

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        reservation_price: float,
        position: int,
        config: ProductConfig,
        buy_size: int,
        sell_size: int,
        buy_take_threshold: float,
        sell_take_threshold: float,
        allow_passive_buy: bool,
        allow_passive_sell: bool,
    ) -> List[Order]:
        orders: List[Order] = []

        buy_capacity = max(0, config.position_limit - position)
        sell_capacity = max(0, config.position_limit + position)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        # 1) Take stale quotes first.
        if best_ask is not None and buy_capacity > 0:
            if best_ask <= reservation_price - buy_take_threshold:
                qty = min(buy_capacity, abs(best_ask_volume), buy_size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_capacity -= qty

        if best_bid is not None and sell_capacity > 0:
            if best_bid >= reservation_price + sell_take_threshold:
                qty = min(sell_capacity, abs(best_bid_volume), sell_size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_capacity -= qty

        # 2) Place passive quotes if still worthwhile.
        passive_bid: Optional[int] = None
        passive_ask: Optional[int] = None

        if best_bid is None and best_ask is None:
            passive_bid = int(reservation_price - 1)
            passive_ask = int(reservation_price + 1)

        elif best_bid is None and best_ask is not None:
            passive_bid = min(int(reservation_price - 1), best_ask - 2)
            passive_ask = max(int(reservation_price + 1), best_ask)

        elif best_ask is None and best_bid is not None:
            passive_bid = min(best_bid + 1, int(reservation_price - 1))
            passive_ask = max(int(reservation_price + 1), best_bid + 2)

        else:
            candidate_bid = best_bid + 1
            candidate_ask = best_ask - 1

            if candidate_bid <= reservation_price - config.min_edge_to_quote:
                passive_bid = candidate_bid
            else:
                passive_bid = int(reservation_price - 1)

            if candidate_ask >= reservation_price + config.min_edge_to_quote:
                passive_ask = candidate_ask
            else:
                passive_ask = int(reservation_price + 1)

            if passive_bid >= passive_ask:
                passive_bid = int(reservation_price - 1)
                passive_ask = int(reservation_price + 1)

        if allow_passive_buy and passive_bid is not None and buy_capacity > 0:
            qty = min(buy_capacity, buy_size)
            if qty > 0:
                orders.append(Order(product, passive_bid, qty))

        if allow_passive_sell and passive_ask is not None and sell_capacity > 0:
            qty = min(sell_capacity, sell_size)
            if qty > 0:
                orders.append(Order(product, passive_ask, -qty))

        return orders

    @staticmethod
    def get_best_bid(order_depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not order_depth.buy_orders:
            return None, 0
        best_bid = max(order_depth.buy_orders.keys())
        return best_bid, order_depth.buy_orders[best_bid]

    @staticmethod
    def get_best_ask(order_depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not order_depth.sell_orders:
            return None, 0
        best_ask = min(order_depth.sell_orders.keys())
        return best_ask, order_depth.sell_orders[best_ask]

    @staticmethod
    def get_mid(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is None or best_ask is None:
            return None
        return 0.5 * (best_bid + best_ask)

    @staticmethod
    def get_spread(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is None or best_ask is None:
            return None
        return float(best_ask - best_bid)

    @staticmethod
    def get_top_imbalance(best_bid_volume: int, best_ask_volume: int) -> Optional[float]:
        bid_vol = abs(best_bid_volume)
        ask_vol = abs(best_ask_volume)
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    @staticmethod
    def get_popular_mid(order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        popular_bid = max(order_depth.buy_orders.items(), key=lambda kv: abs(kv[1]))[0]
        popular_ask = max(order_depth.sell_orders.items(), key=lambda kv: abs(kv[1]))[0]
        return 0.5 * (popular_bid + popular_ask)
