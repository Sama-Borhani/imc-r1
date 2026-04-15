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
        inventory_skew_ticks=5.5,
        take_threshold=1.5,
        min_edge_to_quote=1.0,
        default_order_size=8,
    ),
}

OSMIUM_FAIR_VALUE = 10000

PEPPER_INITIAL_INTERCEPT = 12000.0
PEPPER_INTERCEPT_WINDOW = 50
PEPPER_RESIDUAL_STD = 2.2

OSMIUM_SPREAD_BASE = 16.0
OSMIUM_IMBALANCE_MULT = 3.8
OSMIUM_RESIDUAL_MULT = -0.03
OSMIUM_SPREAD_MULT = -0.015
OSMIUM_ALPHA_CLIP = 2.0

PEPPER_RESIDUAL_MULT = -0.55
PEPPER_ALPHA_CLIP = 3.0

LATE_TS = 9700
ADVERSE_SHIFT = 0.75


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

        adv_shift, buy_adj, sell_adj = self.get_fill_reaction(state, OSMIUM, mid)

        reservation = (
            OSMIUM_FAIR_VALUE
            + alpha
            + adv_shift
            - inventory_adjustment(position, config.position_limit, config.inventory_skew_ticks)
        )

        buy_size = config.default_order_size
        sell_size = config.default_order_size
        buy_improve = 1
        sell_improve = 1
        take_threshold = config.take_threshold

        if imbalance is not None and imbalance >= 0.35:
            buy_size += 2
            sell_size -= 2
            sell_improve = 0
        elif imbalance is not None and imbalance <= -0.35:
            sell_size += 2
            buy_size -= 2
            buy_improve = 0

        buy_size = max(4, min(20, buy_size + buy_adj))
        sell_size = max(4, min(20, sell_size + sell_adj))

        if abs(position) >= 55:
            if position > 0:
                reservation -= 1.0
                buy_improve = 0
                sell_improve = 1
                sell_size = min(24, sell_size + 4)
                take_threshold = 0.5
            else:
                reservation += 1.0
                sell_improve = 0
                buy_improve = 1
                buy_size = min(24, buy_size + 4)
                take_threshold = 0.5

        if state.timestamp >= LATE_TS:
            if position > 0:
                reservation -= 1.5
                buy_improve = 0
                sell_improve = 1
                sell_size = min(24, sell_size + 4)
                take_threshold = 0.5
            elif position < 0:
                reservation += 1.5
                sell_improve = 0
                buy_improve = 1
                buy_size = min(24, buy_size + 4)
                take_threshold = 0.5

        return self.make_orders(
            product=OSMIUM,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            buy_improve=buy_improve,
            sell_improve=sell_improve,
            take_threshold=take_threshold,
        )

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        config = PRODUCT_CONFIG[PEPPER]
        position = state.position.get(PEPPER, 0)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)
        raw_mid = self.get_mid(best_bid, best_ask)
        popular_mid = self.get_popular_mid(order_depth)
        signal_mid = popular_mid if popular_mid is not None else raw_mid

        trend_fv = self.estimate_pepper_trend(state.timestamp, signal_mid)

        residual = 0.0
        if raw_mid is not None:
            residual = raw_mid - trend_fv

        z_score = residual / PEPPER_RESIDUAL_STD if PEPPER_RESIDUAL_STD > 0 else 0.0
        residual_alpha = clip(PEPPER_RESIDUAL_MULT * residual, -PEPPER_ALPHA_CLIP, PEPPER_ALPHA_CLIP)

        adv_shift, buy_adj, sell_adj = self.get_fill_reaction(state, PEPPER, raw_mid)

        reservation = (
            trend_fv
            + residual_alpha
            + adv_shift
            - inventory_adjustment(position, config.position_limit, config.inventory_skew_ticks)
        )

        buy_size = config.default_order_size
        sell_size = config.default_order_size
        buy_improve = 1
        sell_improve = 1
        take_threshold = config.take_threshold

        if z_score >= 0.75:
            sell_size += 2
            buy_size -= 1
            buy_improve = 0
        elif z_score <= -0.75:
            buy_size += 2
            sell_size -= 1
            sell_improve = 0

        if z_score >= 1.4:
            sell_size += 2
            take_threshold = 1.0
        elif z_score <= -1.4:
            buy_size += 2
            take_threshold = 1.0

        buy_size = max(4, min(18, buy_size + buy_adj))
        sell_size = max(4, min(18, sell_size + sell_adj))

        if abs(position) >= 50:
            if position > 0:
                reservation -= 1.0
                buy_improve = 0
                sell_improve = 1
                sell_size = min(20, sell_size + 4)
                take_threshold = 1.0
            else:
                reservation += 1.0
                sell_improve = 0
                buy_improve = 1
                buy_size = min(20, buy_size + 4)
                take_threshold = 1.0

        if state.timestamp >= LATE_TS:
            if position > 0:
                reservation -= 1.5
                buy_improve = 0
                sell_improve = 1
                sell_size = min(20, sell_size + 4)
                take_threshold = 1.0
            elif position < 0:
                reservation += 1.5
                sell_improve = 0
                buy_improve = 1
                buy_size = min(20, buy_size + 4)
                take_threshold = 1.0

        return self.make_orders(
            product=PEPPER,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
            buy_size=buy_size,
            sell_size=sell_size,
            buy_improve=buy_improve,
            sell_improve=sell_improve,
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

    def get_fill_reaction(
        self,
        state: TradingState,
        product: str,
        current_mid: Optional[float],
    ) -> Tuple[float, int, int]:
        if current_mid is None:
            return 0.0, 0, 0

        own = state.own_trades.get(product, [])
        if not own:
            return 0.0, 0, 0

        shift = 0.0
        buy_adj = 0
        sell_adj = 0

        for trade in own:
            if getattr(trade, "buyer", None) == "SUBMISSION":
                if current_mid < trade.price:
                    shift -= ADVERSE_SHIFT
                    buy_adj -= 1
                    sell_adj += 1
            if getattr(trade, "seller", None) == "SUBMISSION":
                if current_mid > trade.price:
                    shift += ADVERSE_SHIFT
                    sell_adj -= 1
                    buy_adj += 1

        return clip(shift, -2.0, 2.0), clip(buy_adj, -2, 2), clip(sell_adj, -2, 2)

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        reservation_price: float,
        position: int,
        config: ProductConfig,
        buy_size: int,
        sell_size: int,
        buy_improve: int,
        sell_improve: int,
        take_threshold: float,
    ) -> List[Order]:
        orders: List[Order] = []

        buy_capacity = max(0, config.position_limit - position)
        sell_capacity = max(0, config.position_limit + position)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

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

        passive_bid: Optional[int] = None
        passive_ask: Optional[int] = None

        if best_bid is None and best_ask is None:
            passive_bid = int(reservation_price - 1)
            passive_ask = int(reservation_price + 1)

        elif best_bid is None and best_ask is not None:
            passive_bid = min(int(reservation_price - 1), best_ask - 2)
            passive_ask = max(int(reservation_price + 1), best_ask)

        elif best_ask is None and best_bid is not None:
            passive_bid = min(best_bid + buy_improve, int(reservation_price - 1))
            passive_ask = max(int(reservation_price + 1), best_bid + 2)

        else:
            candidate_bid = best_bid + buy_improve
            candidate_ask = best_ask - sell_improve

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

        if passive_bid is not None and buy_capacity > 0:
            qty = min(buy_capacity, buy_size)
            if qty > 0:
                orders.append(Order(product, passive_bid, qty))

        if passive_ask is not None and sell_capacity > 0:
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
