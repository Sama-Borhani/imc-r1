import json
from typing import Dict, List
from datamodel import Order, TradingState, OrderDepth

# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SETTINGS
# ═══════════════════════════════════════════════════════════════════════
POSITION_LIMIT = 80
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ACO Parameters
ACO_EWMA_ALPHA  = 0.05    # Faster adaptation than 100-tick rolling mean
ACO_BASE_EDGE   = 2
ACO_SKEW_FACTOR = 3

# Timing
ENDGAME_TS = 999_500 

class Trader:
    def bid(self) -> int:
        return 1_000

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        data = json.loads(state.traderData) if state.traderData else {}

        # 1. ACO Fair Value (EWMA) - Fixes traderData size bug
        aco_fv = data.get("fv", 10000.0)
        if ACO in state.order_depths:
            d = state.order_depths[ACO]
            if d.buy_orders and d.sell_orders:
                mid = (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0
                aco_fv = (ACO_EWMA_ALPHA * mid) + (1.0 - ACO_EWMA_ALPHA) * aco_fv
        data["fv"] = aco_fv

        # 2. Context
        ts = state.timestamp
        is_endgame = (ts >= ENDGAME_TS)
        
        # Note: If traderData is cleared between days, is_last_day requires 
        # a different detection method (e.g., checking price levels).
        ticks = data.get("tt", 0) + 1
        data["tt"] = ticks
        is_last_day = (ticks > 20000) 

        # 3. Product Execution
        for product in [IPR, ACO]:
            if product not in state.order_depths: continue
            depth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == IPR:
                orders = self._trade_ipr(depth, pos, is_endgame, is_last_day, ts)
            else:
                orders = self._trade_aco(depth, pos, is_endgame, is_last_day, aco_fv)
            
            if orders:
                result[product] = orders

        return result, 0, json.dumps(data)

    def _trade_ipr(self, depth: OrderDepth, pos: int, end: bool, last: bool, ts: int) -> List[Order]:
        orders = []
        if end and last:
            # Aggressive closeout
            curr = pos
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if curr <= 0: break
                q = min(depth.buy_orders[bp], curr)
                orders.append(Order(IPR, int(bp), -q))
                curr -= q
            return orders

        # Passive Entry for IPR: Save the spread at start of day
        target = POSITION_LIMIT - pos
        if target > 0:
            if ts < 1000 and depth.buy_orders:
                # Try to join the best bid + 1 to save the 14-tick spread
                best_bid = max(depth.buy_orders.keys())
                orders.append(Order(IPR, best_bid + 1, target))
            else:
                # Fall back to aggressive if not filled or past initial ticks
                for ap in sorted(depth.sell_orders.keys()):
                    if target <= 0: break
                    q = min(abs(depth.sell_orders[ap]), target)
                    orders.append(Order(IPR, int(ap), q))
                    target -= q
        return orders

    def _trade_aco(self, depth: OrderDepth, pos: int, end: bool, last: bool, fv: float) -> List[Order]:
        orders = []
        if end and last:
            # Aggressive closeout
            if pos > 0:
                curr = pos
                for bp in sorted(depth.buy_orders.keys(), reverse=True):
                    if curr <= 0: break
                    q = min(depth.buy_orders[bp], curr)
                    orders.append(Order(ACO, int(bp), -q))
                    curr -= q
            elif pos < 0:
                curr = -pos
                for ap in sorted(depth.sell_orders.keys()):
                    if curr <= 0: break
                    q = min(abs(depth.sell_orders[ap]), curr)
                    orders.append(Order(ACO, int(ap), q))
                    curr -= q
            return orders

        # Skewed Market Making
        skew = (pos / POSITION_LIMIT) * ACO_SKEW_FACTOR
        buy_thresh, sell_thresh = fv - ACO_BASE_EDGE - skew, fv + ACO_BASE_EDGE - skew

        # Aggressive takes
        curr_pos = pos
        cap_long, cap_short = POSITION_LIMIT - curr_pos, POSITION_LIMIT + curr_pos
        for ap, av in sorted(depth.sell_orders.items()):
            if ap <= buy_thresh and cap_long > 0:
                q = min(abs(av), cap_long)
                orders.append(Order(ACO, int(ap), q))
                cap_long -= q; curr_pos += q
        for bp, bv in sorted(depth.buy_orders.items(), reverse=True):
            if bp >= sell_thresh and cap_short > 0:
                q = min(bv, cap_short)
                orders.append(Order(ACO, int(bp), -q))
                cap_short -= q; curr_pos -= q

        # Passive quoting
        if depth.buy_orders and depth.sell_orders:
            bb, ba = max(depth.buy_orders.keys()), min(depth.sell_orders.keys())
            if bb + 1 < ba - 1:
                if POSITION_LIMIT - curr_pos > 0:
                    orders.append(Order(ACO, bb + 1, POSITION_LIMIT - curr_pos))
                if POSITION_LIMIT + curr_pos > 0:
                    orders.append(Order(ACO, ba - 1, -(POSITION_LIMIT + curr_pos)))
        return orders
