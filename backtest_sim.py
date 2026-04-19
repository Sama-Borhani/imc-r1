from __future__ import annotations

import csv
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).parent
R1 = HERE.parent / "r1"
DATA_DIR = R1 / "data" / "round1"

# Always use r2's datamodel
sys.path.insert(0, str(HERE))
from datamodel import Order, OrderDepth, Trade, TradingState


def load_trader(module_path: Path):
    name = f"trader_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module's dir is on sys.path so its own imports (params, datamodel) resolve.
    mod_dir = str(module_path.parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    # Register in sys.modules so dataclasses' sys.modules lookup during class
    # creation (for resolving type annotations) succeeds.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.Trader()


def load_ticks(csv_path: Path) -> Dict[int, Dict[str, dict]]:
    """Returns {timestamp: {product: {bids: {px: vol}, asks: {px: vol}, mid: float}}}."""
    out: Dict[int, Dict[str, dict]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            t = int(row["timestamp"])
            prod = row["product"]
            bids: Dict[int, int] = {}
            asks: Dict[int, int] = {}
            for lvl in (1, 2, 3):
                bp = row.get(f"bid_price_{lvl}")
                bv = row.get(f"bid_volume_{lvl}")
                ap = row.get(f"ask_price_{lvl}")
                av = row.get(f"ask_volume_{lvl}")
                if bp and bv:
                    bids[int(bp)] = int(bv)
                if ap and av:
                    asks[-1 * 0 + int(ap)] = -int(av)  # sell volumes are negative in datamodel
            mid = float(row["mid_price"]) if row.get("mid_price") else None
            out.setdefault(t, {})[prod] = {"bids": bids, "asks": asks, "mid": mid}
    return out


def build_state(
    traderData: str,
    timestamp: int,
    tick: Dict[str, dict],
    position: Dict[str, int],
    own_trades: Dict[str, List[Trade]],
) -> TradingState:
    order_depths: Dict[str, OrderDepth] = {}
    for prod, v in tick.items():
        od = OrderDepth()
        od.buy_orders = dict(v["bids"])
        od.sell_orders = dict(v["asks"])
        order_depths[prod] = od
    return TradingState(
        traderData=traderData,
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades={},
        position=dict(position),
        observations=None,
    )


@dataclass
class FillRecord:
    product: str
    price: int
    qty: int                # +buy, -sell


def match_aggressive(orders: List[Order], depth_bids: Dict[int, int], depth_asks: Dict[int, int]) -> List[FillRecord]:
    """Fill orders that cross the resting book at the tick they were emitted."""
    fills: List[FillRecord] = []
    # Buys: qty > 0. Fill against asks if order.price >= best_ask.
    buys = sorted([o for o in orders if o.quantity > 0], key=lambda o: -o.price)
    sells = sorted([o for o in orders if o.quantity < 0], key=lambda o: o.price)

    asks_sorted = sorted(depth_asks.items())  # ascending price
    bids_sorted = sorted(depth_bids.items(), reverse=True)  # descending price

    for o in buys:
        qty_left = o.quantity
        for i, (ap, av) in enumerate(asks_sorted):
            if qty_left <= 0 or av >= 0:
                continue
            avail = -av  # asks store negative volumes
            if o.price >= ap and avail > 0:
                take = min(qty_left, avail)
                fills.append(FillRecord(o.symbol, ap, take))
                asks_sorted[i] = (ap, av + take)
                qty_left -= take

    for o in sells:
        qty_left = -o.quantity
        for i, (bp, bv) in enumerate(bids_sorted):
            if qty_left <= 0 or bv <= 0:
                continue
            if o.price <= bp and bv > 0:
                take = min(qty_left, bv)
                fills.append(FillRecord(o.symbol, bp, -take))
                bids_sorted[i] = (bp, bv - take)
                qty_left -= take

    return fills


def match_passive_against_next(
    orders: List[Order],
    depth_bids: Dict[int, int],
    depth_asks: Dict[int, int],
    next_bids: Dict[int, int],
    next_asks: Dict[int, int],
) -> List[FillRecord]:
    """Passive orders fill if next tick's opposing best touches/crosses ours.
    Only matches orders that did NOT cross the book at placement."""
    fills: List[FillRecord] = []
    best_ask_now = min(depth_asks.keys()) if depth_asks else None
    best_bid_now = max(depth_bids.keys()) if depth_bids else None

    for o in orders:
        if o.quantity > 0:  # buy
            # Was it crossing? If so, skip (already matched by match_aggressive).
            if best_ask_now is not None and o.price >= best_ask_now:
                continue
            # Passive buy fills if next tick ask touches/crosses our bid.
            if next_asks:
                next_best_ask = min(next_asks.keys())
                if next_best_ask <= o.price:
                    avail = -next_asks[next_best_ask]
                    take = min(o.quantity, max(0, avail))
                    if take > 0:
                        fills.append(FillRecord(o.symbol, o.price, take))
        else:  # sell
            if best_bid_now is not None and o.price <= best_bid_now:
                continue
            if next_bids:
                next_best_bid = max(next_bids.keys())
                if next_best_bid >= o.price:
                    avail = next_bids[next_best_bid]
                    take = min(-o.quantity, max(0, avail))
                    if take > 0:
                        fills.append(FillRecord(o.symbol, o.price, -take))
    return fills


def simulate(trader_path: Path, csv_path: Path) -> dict:
    trader = load_trader(trader_path)
    ticks = load_ticks(csv_path)
    timestamps = sorted(ticks.keys())

    products = set()
    for t in ticks.values():
        products.update(t.keys())

    position: Dict[str, int] = {p: 0 for p in products}
    position_limit = 80
    cash: Dict[str, float] = {p: 0.0 for p in products}
    own_trades: Dict[str, List[Trade]] = {p: [] for p in products}
    traderData = ""
    all_fills: List[Tuple[int, FillRecord]] = []

    # Track PnL curve (MtM at each tick)
    pnl_curve: List[Tuple[int, float]] = []
    max_abs_pos: Dict[str, int] = {p: 0 for p in products}

    for i, ts in enumerate(timestamps):
        tick = ticks[ts]
        state = build_state(traderData, ts, tick, position, own_trades)
        try:
            orders_by_prod, _, traderData = trader.run(state)
        except Exception as e:  # noqa: BLE001
            return {"error": f"trader exception at t={ts}: {e.__class__.__name__}: {e}"}

        tick_fills: Dict[str, List[FillRecord]] = {p: [] for p in products}

        # Match aggressive in this tick's book.
        for prod, orders in orders_by_prod.items():
            if prod not in tick:
                continue
            fills = match_aggressive(orders, tick[prod]["bids"], tick[prod]["asks"])
            tick_fills[prod].extend(fills)

        # Match passive against next tick's book.
        if i + 1 < len(timestamps):
            next_ts = timestamps[i + 1]
            next_tick = ticks[next_ts]
            for prod, orders in orders_by_prod.items():
                if prod not in tick or prod not in next_tick:
                    continue
                passives = [o for o in orders if not _crosses(o, tick[prod])]
                fills = match_passive_against_next(
                    passives,
                    tick[prod]["bids"], tick[prod]["asks"],
                    next_tick[prod]["bids"], next_tick[prod]["asks"],
                )
                tick_fills[prod].extend(fills)

        # Apply fills, respecting position limits.
        new_own = {p: [] for p in products}
        for prod, fills in tick_fills.items():
            for f in fills:
                # Respect position limits.
                if f.qty > 0 and position[prod] + f.qty > position_limit:
                    f = FillRecord(f.product, f.price, position_limit - position[prod])
                if f.qty < 0 and position[prod] + f.qty < -position_limit:
                    f = FillRecord(f.product, f.price, -position_limit - position[prod])
                if f.qty == 0:
                    continue
                position[prod] += f.qty
                cash[prod] -= f.price * f.qty   # buying (qty>0) reduces cash
                all_fills.append((ts, f))
                # Build own_trades entry for next tick's state
                trade = Trade(prod, f.price, f.qty, "SUBMISSION" if f.qty > 0 else None,
                              "SUBMISSION" if f.qty < 0 else None, ts)
                new_own[prod].append(trade)

        own_trades = new_own
        for p in products:
            max_abs_pos[p] = max(max_abs_pos[p], abs(position[p]))

        # Record MtM PnL
        mtm_total = 0.0
        for prod in products:
            mid = tick.get(prod, {}).get("mid")
            if mid is not None:
                mtm_total += cash[prod] + position[prod] * mid
        pnl_curve.append((ts, mtm_total))

    # Final PnL
    last_tick = ticks[timestamps[-1]]
    pnl_per_prod = {}
    total_pnl = 0.0
    for prod in sorted(products):
        mid = last_tick.get(prod, {}).get("mid")
        p = cash[prod] + (position[prod] * mid if mid is not None else 0)
        pnl_per_prod[prod] = round(p, 2)
        total_pnl += p

    # Drawdown
    peak = pnl_curve[0][1] if pnl_curve else 0
    max_dd = 0.0
    for _, v in pnl_curve:
        peak = max(peak, v)
        max_dd = min(max_dd, v - peak)

    return {
        "trader": str(trader_path.name),
        "data": str(csv_path.name),
        "ticks": len(timestamps),
        "pnl_per_product": pnl_per_prod,
        "total_pnl": round(total_pnl, 2),
        "final_position": {p: position[p] for p in sorted(products)},
        "max_abs_position": dict(max_abs_pos),
        "trades": len(all_fills),
        "max_drawdown": round(max_dd, 2),
    }


def _crosses(o: Order, tick: dict) -> bool:
    """True if this order crosses the resting book now (would match aggressively)."""
    if o.quantity > 0:
        best_ask = min(tick["asks"].keys()) if tick["asks"] else None
        return best_ask is not None and o.price >= best_ask
    else:
        best_bid = max(tick["bids"].keys()) if tick["bids"] else None
        return best_bid is not None and o.price <= best_bid


def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("trader", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    result = simulate(args.trader, args.csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
