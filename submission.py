from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

from params import (
    ADVERSE_MAX_SHIFT,
    ADVERSE_SHIFT_TICKS,
    LATE_TS,
    OSMIUM,
    OSMIUM_CONFIG,
    PEPPER,
    PEPPER_CONFIG,
    OsmiumConfig,
    PepperConfig,
)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def inventory_adjustment(position: int, position_limit: int, skew_ticks: float) -> float:
    if position_limit <= 0:
        return 0.0
    return skew_ticks * (position / position_limit)



def _get_best_bid(order_depth: OrderDepth) -> Tuple[Optional[int], int]:
    if not order_depth.buy_orders:
        return None, 0
    best_bid = max(order_depth.buy_orders.keys())
    return best_bid, order_depth.buy_orders[best_bid]


def _get_best_ask(order_depth: OrderDepth) -> Tuple[Optional[int], int]:
    if not order_depth.sell_orders:
        return None, 0
    best_ask = min(order_depth.sell_orders.keys())
    return best_ask, order_depth.sell_orders[best_ask]


def _get_mid(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    return 0.5 * (best_bid + best_ask)


def _get_spread(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    return float(best_ask - best_bid)


def _get_top_imbalance(best_bid_volume: int, best_ask_volume: int) -> Optional[float]:
    bid_vol = abs(best_bid_volume)
    ask_vol = abs(best_ask_volume)
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def _get_popular_mid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    popular_bid = max(order_depth.buy_orders.items(), key=lambda kv: abs(kv[1]))[0]
    popular_ask = max(order_depth.sell_orders.items(), key=lambda kv: abs(kv[1]))[0]
    return 0.5 * (popular_bid + popular_ask)


@dataclass
class DepthSnapshot:
    best_bid: Optional[int]
    best_bid_volume: int
    best_ask: Optional[int]
    best_ask_volume: int
    mid: Optional[float]
    spread: Optional[float]
    imbalance: Optional[float]
    popular_mid: Optional[float]

    @classmethod
    def from_depth(cls, depth: OrderDepth) -> "DepthSnapshot":
        bb, bbv = _get_best_bid(depth)
        ba, bav = _get_best_ask(depth)
        return cls(
            best_bid=bb,
            best_bid_volume=bbv,
            best_ask=ba,
            best_ask_volume=bav,
            mid=_get_mid(bb, ba),
            spread=_get_spread(bb, ba),
            imbalance=_get_top_imbalance(bbv, bav),
            popular_mid=_get_popular_mid(depth),
        )


# shared state

@dataclass
class ProductMemory:
    # Pepper OLS buffer.
    pepper_ols_xs: List[float] = field(default_factory=list)
    pepper_ols_ys: List[float] = field(default_factory=list)
    pepper_intercept: float = 0.0   # seeded from config on first use (see PepperTrader.__init__)
    pepper_slope: float = 0.0

    # Osmium EMA of mid.
    osmium_fv_ema: Optional[float] = None

    # Adverse-fill cross-product bookkeeping.
    last_own_trades: Dict[str, List[dict]] = field(default_factory=dict)
    last_mid: Dict[str, float] = field(default_factory=dict)


# base product trader

class BaseProductTrader:
    """Common lifecycle for a single product.

    Subclasses override the three hooks: ``fair_value``, ``compute_alpha``,
    ``compute_sizing``. Everything else (reservation math, adverse-fill shift,
    EOD guard, order placement) is inherited.
    """

    PRODUCT: str = ""

    def __init__(self, config, memory: ProductMemory) -> None:
        self.config = config
        self.memory = memory

    # ---- hooks (override in subclass) ----

    def fair_value(self, state: TradingState, snap: DepthSnapshot) -> float:
        raise NotImplementedError

    def compute_alpha(self, state: TradingState, snap: DepthSnapshot, fv: float) -> float:
        raise NotImplementedError

    def compute_sizing(
        self,
        state: TradingState,
        snap: DepthSnapshot,
        fv: float,
        alpha: float,
        position: int,
    ) -> Tuple[int, int, bool, bool]:
        """Return (buy_size, sell_size, allow_passive_buy, allow_passive_sell).
        Default: symmetric default size, both sides passive."""
        c = self.config
        return c.default_order_size, c.default_order_size, True, True

    def inventory_anchor(self, state: TradingState, position: int) -> int:
        """Reference inventory for the skew. Default 0 (symmetric around flat).
        Products with a known directional drift (e.g. pepper's +1/1000 trend)
        override this to lean long/short instead of being skewed back to zero."""
        return 0

    def run(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        position = state.position.get(self.PRODUCT, 0)
        snap = DepthSnapshot.from_depth(depth)

        fv = self.fair_value(state, snap)
        alpha = self.compute_alpha(state, snap, fv)
        adv_shift = self._adverse_fill_shift(snap.mid)

        anchor = self.inventory_anchor(state, position)
        reservation = (
            fv
            + alpha
            + adv_shift
            - inventory_adjustment(position - anchor, self.config.position_limit, self.config.inventory_skew_ticks)
        )

        bs, ss, apb, aps = self.compute_sizing(state, snap, fv, alpha, position)
        eod_pb, eod_ps, take_threshold = self._eod_adjust(state.timestamp, position)

        return self._make_orders(
            snap=snap,
            reservation_price=reservation,
            position=position,
            buy_size=bs,
            sell_size=ss,
            allow_passive_buy=apb and eod_pb,
            allow_passive_sell=aps and eod_ps,
            take_threshold=take_threshold,
        )

    def _adverse_fill_shift(self, current_mid: Optional[float]) -> float:
        """shared helper. If last tick's fills are underwater against current mid, shift reservation away."""
        if current_mid is None:
            return 0.0
        fills = self.memory.last_own_trades.get(self.PRODUCT) or []
        prev_mid = self.memory.last_mid.get(self.PRODUCT)
        if not fills or prev_mid is None:
            return 0.0
        shift = 0.0
        for f in fills:
            px = f.get("price")
            if px is None:
                continue
            if f.get("buyer") == "SUBMISSION" and current_mid < px:
                shift -= ADVERSE_SHIFT_TICKS
            if f.get("seller") == "SUBMISSION" and current_mid > px:
                shift += ADVERSE_SHIFT_TICKS
        return clip(shift, -ADVERSE_MAX_SHIFT, ADVERSE_MAX_SHIFT)

    def _eod_adjust(self, timestamp: int, position: int) -> Tuple[bool, bool, float]:
        """shared helpers. In the last ~3% of the day, bias quoting to flatten inventory."""
        allow_passive_buy = True
        allow_passive_sell = True
        take_threshold = self.config.take_threshold
        if timestamp >= LATE_TS:
            if position > 0:
                allow_passive_buy = False
                take_threshold = min(take_threshold, 0.5)
            elif position < 0:
                allow_passive_sell = False
                take_threshold = min(take_threshold, 0.5)
        return allow_passive_buy, allow_passive_sell, take_threshold

    def _make_orders(
        self,
        snap: DepthSnapshot,
        reservation_price: float,
        position: int,
        buy_size: int,
        sell_size: int,
        allow_passive_buy: bool,
        allow_passive_sell: bool,
        take_threshold: float,
    ) -> List[Order]:
        orders: List[Order] = []
        c = self.config
        product = self.PRODUCT

        buy_capacity = max(0, c.position_limit - position)
        sell_capacity = max(0, c.position_limit + position)

        best_bid = snap.best_bid
        best_ask = snap.best_ask
        best_bid_volume = snap.best_bid_volume
        best_ask_volume = snap.best_ask_volume

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

        # 2) Passive quotes.
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
            if candidate_bid <= reservation_price - c.min_edge_to_quote:
                passive_bid = candidate_bid
            else:
                passive_bid = int(reservation_price - 1)
            if candidate_ask >= reservation_price + c.min_edge_to_quote:
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


# ---------------------- osmium ----------------------

class OsmiumTrader(BaseProductTrader):
    PRODUCT = OSMIUM
    config: OsmiumConfig

    def fair_value(self, state: TradingState, snap: DepthSnapshot) -> float:
        c = self.config
        mid = snap.mid
        if mid is not None:
            if self.memory.osmium_fv_ema is None:
                self.memory.osmium_fv_ema = mid
            else:
                a = c.fv_ema_alpha
                self.memory.osmium_fv_ema = a * mid + (1 - a) * self.memory.osmium_fv_ema
        ema = self.memory.osmium_fv_ema if self.memory.osmium_fv_ema is not None else c.anchor_fv
        w = c.fv_anchor_weight
        return w * c.anchor_fv + (1 - w) * ema

    def compute_alpha(self, state: TradingState, snap: DepthSnapshot, fv: float) -> float:
        c = self.config
        alpha = 0.0
        if snap.imbalance is not None:
            alpha += c.imbalance_mult * snap.imbalance
        if snap.mid is not None:
            alpha += c.residual_mult * (snap.mid - fv)
        if snap.spread is not None:
            alpha += c.spread_mult * (snap.spread - c.spread_base)
        return clip(alpha, -c.alpha_clip, c.alpha_clip)

    def compute_sizing(
        self,
        state: TradingState,
        snap: DepthSnapshot,
        fv: float,
        alpha: float,
        position: int,
    ) -> Tuple[int, int, bool, bool]:
        c = self.config
        bs = c.default_order_size
        ss = c.default_order_size
        imb = snap.imbalance
        if imb is not None and abs(imb) >= 0.4:
            if imb > 0:
                bs = min(20, c.default_order_size + 4)
                ss = max(6, c.default_order_size - 2)
            else:
                ss = min(20, c.default_order_size + 4)
                bs = max(6, c.default_order_size - 2)
        return bs, ss, True, True


# ---------------------- pepper ----------------------

class PepperTrader(BaseProductTrader):
    PRODUCT = PEPPER
    config: PepperConfig

    def __init__(self, config: PepperConfig, memory: ProductMemory) -> None:
        super().__init__(config, memory)
        # Seed OLS fit from config defaults on first construction. Subsequent
        # load_memory() calls may overwrite these with persisted values.
        if self.memory.pepper_intercept == 0.0 and self.memory.pepper_slope == 0.0:
            self.memory.pepper_intercept = config.intercept_seed
            self.memory.pepper_slope = config.slope_seed

    def fair_value(self, state: TradingState, snap: DepthSnapshot) -> float:
        signal_mid = snap.popular_mid if snap.popular_mid is not None else snap.mid
        self._update_ols(state.timestamp, signal_mid)
        return self.memory.pepper_intercept + self.memory.pepper_slope * (state.timestamp / 1000.0)

    def inventory_anchor(self, state: TradingState, position: int) -> int:
        return self.config.target_inventory

    def compute_alpha(self, state: TradingState, snap: DepthSnapshot, fv: float) -> float:
        c = self.config
        residual = 0.0
        if snap.mid is not None:
            residual = snap.mid - fv
        return clip(c.residual_mult * residual, -c.alpha_clip, c.alpha_clip)

    def compute_sizing(
        self,
        state: TradingState,
        snap: DepthSnapshot,
        fv: float,
        alpha: float,
        position: int,
    ) -> Tuple[int, int, bool, bool]:
        c = self.config
        residual = 0.0
        if snap.mid is not None:
            residual = snap.mid - fv
        z_score = residual / c.residual_std if c.residual_std > 0 else 0.0

        bs = c.default_order_size
        ss = c.default_order_size
        apb = True
        aps = True
        if z_score >= c.strong_z:
            ss = min(16, c.default_order_size + 4)
            bs = max(4, c.default_order_size - 2)
            apb = False
        elif z_score <= -c.strong_z:
            bs = min(16, c.default_order_size + 4)
            ss = max(4, c.default_order_size - 2)
            aps = False
        return bs, ss, apb, aps

    def _update_ols(self, timestamp: int, signal_mid: Optional[float]) -> None:
        if signal_mid is None:
            return
        c = self.config
        m = self.memory
        x = timestamp / 1000.0
        y = float(signal_mid)

        n_buf = len(m.pepper_ols_xs)
        if n_buf > 0:
            # Outlier reject only once we have a fit to compare against.
            fit_y = m.pepper_intercept + m.pepper_slope * x
            if abs(y - fit_y) > c.outlier_sigma * c.residual_std:
                return

        m.pepper_ols_xs.append(x)
        m.pepper_ols_ys.append(y)
        if len(m.pepper_ols_xs) > c.ols_window:
            m.pepper_ols_xs = m.pepper_ols_xs[-c.ols_window:]
            m.pepper_ols_ys = m.pepper_ols_ys[-c.ols_window:]

        n = len(m.pepper_ols_xs)

        if n < c.ols_warmup:
            # Intercept-only fit: hold slope at seed, estimate intercept as
            # mean(y - slope * x) over the buffer. Avoids spurious trades at
            # session start from a stale seed intercept.
            slope = c.slope_seed
            intercept = (
                sum(yi - slope * xi for xi, yi in zip(m.pepper_ols_xs, m.pepper_ols_ys)) / n
            )
            m.pepper_slope = slope
            m.pepper_intercept = intercept
            return

        sx = sum(m.pepper_ols_xs)
        sy = sum(m.pepper_ols_ys)
        sxx = sum(xi * xi for xi in m.pepper_ols_xs)
        sxy = sum(xi * yi for xi, yi in zip(m.pepper_ols_xs, m.pepper_ols_ys))
        denom = n * sxx - sx * sx
        if denom <= 0:
            return
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        m.pepper_slope = slope
        m.pepper_intercept = intercept


# ---------------------- top-level trader ----------------------

class Trader:
    def __init__(self) -> None:
        self.memory = ProductMemory()
        self.product_traders: Dict[str, BaseProductTrader] = {
            OSMIUM: OsmiumTrader(OSMIUM_CONFIG, self.memory),
            PEPPER: PepperTrader(PEPPER_CONFIG, self.memory),
        }

    # ---- persistence ----

    def load_memory(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            raw = json.loads(trader_data)
        except Exception:
            return
        m = self.memory
        m.pepper_ols_xs = list(raw.get("pepper_ols_xs", m.pepper_ols_xs))
        m.pepper_ols_ys = list(raw.get("pepper_ols_ys", m.pepper_ols_ys))
        m.pepper_intercept = float(raw.get("pepper_intercept", m.pepper_intercept))
        m.pepper_slope = float(raw.get("pepper_slope", m.pepper_slope))
        m.osmium_fv_ema = raw.get("osmium_fv_ema", m.osmium_fv_ema)
        m.last_own_trades = raw.get("last_own_trades", m.last_own_trades)
        m.last_mid = raw.get("last_mid", m.last_mid)

    def dump_memory(self) -> str:
        c = PEPPER_CONFIG
        m = self.memory
        payload = {
            "pepper_ols_xs": m.pepper_ols_xs[-c.ols_window:],
            "pepper_ols_ys": m.pepper_ols_ys[-c.ols_window:],
            "pepper_intercept": m.pepper_intercept,
            "pepper_slope": m.pepper_slope,
            "osmium_fv_ema": m.osmium_fv_ema,
            "last_own_trades": m.last_own_trades,
            "last_mid": m.last_mid,
        }
        return json.dumps(payload, separators=(",", ":"))

    # ---- main dispatch ----

    def run(self, state: TradingState):
        self.load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            trader = self.product_traders.get(product)
            if trader is None:
                continue
            result[product] = trader.run(state, depth)

        # Snapshot current own_trades + mids for next tick's adverse-fill check.
        self.memory.last_own_trades = {
            p: [
                {
                    "price": t.price,
                    "quantity": t.quantity,
                    "buyer": getattr(t, "buyer", None),
                    "seller": getattr(t, "seller", None),
                }
                for t in state.own_trades.get(p, [])
            ]
            for p in self.product_traders
        }
        self.memory.last_mid = {}
        for p in self.product_traders:
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            bb, _ = _get_best_bid(depth)
            ba, _ = _get_best_ask(depth)
            m = _get_mid(bb, ba)
            if m is not None:
                self.memory.last_mid[p] = m

        return result, 0, self.dump_memory()
