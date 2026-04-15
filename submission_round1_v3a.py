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

EARLY_TS = 1200
LATE_TS = 9000

OSMIUM_NO_TRADE_ALPHA = 0.45
OSMIUM_STRONG_ALPHA = 1.25
OSMIUM_MIN_SPREAD_TO_MAKE = 12.0

PEPPER_NO_TRADE_Z = 0.35
PEPPER_MODERATE_Z = 0.75
PEPPER_STRONG_Z = 1.15
PEPPER_MIN_SPREAD_TO_MAKE = 11.0


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


class Trader:
    def __init__(self) -> None:
        self.memory = ProductMemory()

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
            )
        except Exception:
            self.memory = ProductMemory()

    def dump_memory(self) -> str:
        payload = {
            "pepper_intercept_samples": self.memory.pepper_intercept_samples[-PEPPER_INTERCEPT_WINDOW:],
            "pepper_intercept_estimate": self.memory.pepper_intercept_estimate,
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

        buy_size = self.scaled_size(config.default_order_size, state.timestamp)
        sell_size = self.scaled_size(config.default_order_size, state.timestamp)
        allow_passive_buy = True
        allow_passive_sell = True
        take_threshold = config.take_threshold

        # No-trade zone: weak signal + mediocre spread + low inventory.
        if (
            spread is not None
            and spread <= OSMIUM_MIN_SPREAD_TO_MAKE
            and abs(alpha) < OSMIUM_NO_TRADE_ALPHA
            and abs(position) <= 8
        ):
            return []

        # Strong directional mode: quote mostly on the favored side.
        if alpha >= OSMIUM_STRONG_ALPHA:
            allow_passive_sell = False
            buy_size = min(20, buy_size + 4)
            sell_size = max(4, sell_size - 4)
            take_threshold = 0.5
        elif alpha <= -OSMIUM_STRONG_ALPHA:
            allow_passive_buy = False
            sell_size = min(20, sell_size + 4)
            buy_size = max(4, buy_size - 4)
            take_threshold = 0.5

        # Late-day inventory flattening.
        if state.timestamp >= LATE_TS:
            if position > 0:
                allow_passive_buy = False
                sell_size = min(24, sell_size + 4)
                take_threshold = min(take_threshold, 0.5)
            elif position < 0:
                allow_passive_sell = False
                buy_size = min(24, buy_size + 4)
                take_threshold = min(take_threshold, 0.5)

        # Inventory emergency handling.
        if abs(position) >= int(0.65 * config.position_limit):
            if position > 0:
                allow_passive_buy = False
                sell_size = min(28, sell_size + 6)
                take_threshold = 0.5
            else:
                allow_passive_sell = False
                buy_size = min(28, buy_size + 6)
                take_threshold = 0.5

        return self.make_orders(
            product=OSMIUM,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            allow_passive_buy=allow_passive_buy,
            allow_passive_sell=allow_passive_sell,
            take_threshold=take_threshold,
        )

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        config = PRODUCT_CONFIG[PEPPER]
        position = state.position.get(PEPPER, 0)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        popular_mid = self.get_popular_mid(order_depth)
        raw_mid = self.get_mid(best_bid, best_ask)
        spread = self.get_spread(best_bid, best_ask)
        signal_mid = popular_mid if popular_mid is not None else raw_mid

        trend_fv = self.estimate_pepper_trend(state.timestamp, signal_mid)

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

        buy_size = self.scaled_size(config.default_order_size, state.timestamp)
        sell_size = self.scaled_size(config.default_order_size, state.timestamp)
        allow_passive_buy = True
        allow_passive_sell = True
        take_threshold = config.take_threshold

        # No-trade zone: weak residual signal + uninspiring spread + flat inventory.
        if (
            spread is not None
            and spread <= PEPPER_MIN_SPREAD_TO_MAKE
            and abs(z_score) < PEPPER_NO_TRADE_Z
            and abs(position) <= 6
        ):
            return []

        # Moderate signal: reduce wrong-side passive risk.
        if z_score >= PEPPER_MODERATE_Z:
            buy_size = max(4, buy_size - 2)
            sell_size = min(16, sell_size + 2)
        elif z_score <= -PEPPER_MODERATE_Z:
            sell_size = max(4, sell_size - 2)
            buy_size = min(16, buy_size + 2)

        # Strong signal: one-sided quoting + slightly more taker bias.
        if z_score >= PEPPER_STRONG_Z:
            allow_passive_buy = False
            sell_size = min(18, sell_size + 3)
            buy_size = max(4, buy_size - 3)
            take_threshold = 1.0
        elif z_score <= -PEPPER_STRONG_Z:
            allow_passive_sell = False
            buy_size = min(18, buy_size + 3)
            sell_size = max(4, sell_size - 3)
            take_threshold = 1.0

        # Early day: smaller risk until trend estimate stabilizes.
        if state.timestamp <= EARLY_TS:
            buy_size = max(4, int(round(buy_size * 0.75)))
            sell_size = max(4, int(round(sell_size * 0.75)))

        # Late day: stop adding to the wrong inventory.
        if state.timestamp >= LATE_TS:
            if position > 0:
                allow_passive_buy = False
                sell_size = min(18, sell_size + 3)
                take_threshold = min(take_threshold, 1.0)
            elif position < 0:
                allow_passive_sell = False
                buy_size = min(18, buy_size + 3)
                take_threshold = min(take_threshold, 1.0)

        # Inventory emergency handling.
        if abs(position) >= int(0.6 * config.position_limit):
            if position > 0:
                allow_passive_buy = False
                sell_size = min(22, sell_size + 4)
                take_threshold = min(take_threshold, 1.0)
            else:
                allow_passive_sell = False
                buy_size = min(22, buy_size + 4)
                take_threshold = min(take_threshold, 1.0)

        return self.make_orders(
            product=PEPPER,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            allow_passive_buy=allow_passive_buy,
            allow_passive_sell=allow_passive_sell,
            take_threshold=take_threshold,
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

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        reservation_price: float,
        position: int,
        config: ProductConfig,
        buy_size: int,
        sell_size: int,
        allow_passive_buy: bool,
        allow_passive_sell: bool,
        take_threshold: float,
    ) -> List[Order]:
        orders: List[Order] = []

        buy_capacity = max(0, config.position_limit - position)
        sell_capacity = max(0, config.position_limit + position)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        # 1) Take stale quotes first.
        if best_ask is not None and buy_capacity > 0:
            if best_ask <= reservation_price - take_threshold:
                qty = min(buy_capacity, abs(best_ask_volume), buy_size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_capacity -= qty

        if best_bid is not None and sell_capacity > 0:
            if best_bid >= reservation_price + take_threshold:
                qty = min(sell_capacity, abs(best_bid_volume), sell_size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_capacity -= qty

        # 2) Passive quoting, if enabled.
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
    def scaled_size(base_size: int, timestamp: int) -> int:
        if timestamp <= EARLY_TS:
            return max(4, int(round(base_size * 0.7)))
        if timestamp >= LATE_TS:
            return max(4, int(round(base_size * 0.8)))
        return base_size

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
