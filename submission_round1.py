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
    reversion_threshold: float
    min_edge_to_quote: float
    default_order_size: int


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
PEPPER_INITIAL_INTERCEPT = 12000.0
PEPPER_BLEND_TREND = 0.8
PEPPER_BLEND_POPULAR_MID = 0.2
PEPPER_INTERCEPT_WINDOW = 80


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
                pepper_intercept_estimate=raw.get(
                    "pepper_intercept_estimate",
                    PEPPER_INITIAL_INTERCEPT,
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

        reservation = OSMIUM_FAIR_VALUE - inventory_adjustment(
            position,
            config.position_limit,
            config.inventory_skew_ticks,
        )

        return self.make_orders(
            product=OSMIUM,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
        )

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        config = PRODUCT_CONFIG[PEPPER]
        position = state.position.get(PEPPER, 0)

        popular_mid = self.get_popular_mid(order_depth)
        raw_trend_fv = self.estimate_pepper_fair_value(state.timestamp, popular_mid)

        if popular_mid is None:
            fair_value = raw_trend_fv
        else:
            fair_value = (
                PEPPER_BLEND_TREND * raw_trend_fv
                + PEPPER_BLEND_POPULAR_MID * popular_mid
            )

        reservation = fair_value - inventory_adjustment(
            position,
            config.position_limit,
            config.inventory_skew_ticks,
        )

        return self.make_orders(
            product=PEPPER,
            order_depth=order_depth,
            reservation_price=reservation,
            position=position,
            config=config,
        )

    def estimate_pepper_fair_value(
        self,
        timestamp: int,
        popular_mid: Optional[float],
    ) -> float:
        if popular_mid is not None:
            intercept_candidate = popular_mid - (timestamp / 1000.0)
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
    ) -> List[Order]:
        orders: List[Order] = []

        buy_capacity = max(0, config.position_limit - position)
        sell_capacity = max(0, config.position_limit + position)

        best_bid, best_bid_volume = self.get_best_bid(order_depth)
        best_ask, best_ask_volume = self.get_best_ask(order_depth)

        # 1) Take stale quotes first.
        if best_ask is not None and buy_capacity > 0:
            if best_ask <= reservation_price - config.take_threshold:
                qty = min(buy_capacity, abs(best_ask_volume), config.default_order_size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_capacity -= qty

        if best_bid is not None and sell_capacity > 0:
            if best_bid >= reservation_price + config.take_threshold:
                qty = min(sell_capacity, abs(best_bid_volume), config.default_order_size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_capacity -= qty

        # 2) Place passive quotes.
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

        if passive_bid is not None and buy_capacity > 0:
            qty = min(buy_capacity, config.default_order_size)
            if qty > 0:
                orders.append(Order(product, passive_bid, qty))

        if passive_ask is not None and sell_capacity > 0:
            qty = min(sell_capacity, config.default_order_size)
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
    def get_popular_mid(order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        popular_bid = max(order_depth.buy_orders.items(), key=lambda kv: abs(kv[1]))[0]
        popular_ask = max(order_depth.sell_orders.items(), key=lambda kv: abs(kv[1]))[0]
        return 0.5 * (popular_bid + popular_ask)