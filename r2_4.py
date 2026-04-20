import json
from typing import Dict, List, Optional, Tuple
from datamodel import Order, TradingState, OrderDepth

# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SETTINGS
# ═══════════════════════════════════════════════════════════════════════
POSITION_LIMIT = 80
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ACO Parameters (Refined with Jump Protection)
ACO_EWMA_ALPHA   = 0.05
ACO_BASE_EDGE    = 2.0
ACO_SKEW_FACTOR  = 3.0
ACO_JUMP_IMB     = 0.50  # Signal for an imminent price jump
ACO_JUMP_SPREAD  = 10.0  # Narrow spread trigger
ACO_JUMP_BIAS    = 1.0   # Buffer to move quotes away from a jump

# IPR Parameters (Lead-Signal Refinements)
IPR_INTERCEPT_ALPHA = 0.08
IPR_PASSIVE_CLIP    = 20

# Timing
ENDGAME_TS = 999_500 

class Trader:
    def bid(self) -> int:
        """Determines extra market access (Round 2 only)."""
        return 1_000

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        data = json.loads(state.traderData) if state.traderData else {}

        # 1. ACO Fair Value (EWMA) - Replaces static 10k anchor
        aco_fv = data.get("fv", 10000.0)
        if ACO in state.order_depths:
            d = state.order_depths[ACO]
            mid = self._raw_mid(d)
            if mid is not None:
                aco_fv = (ACO_EWMA_ALPHA * mid) + (1.0 - ACO_EWMA_ALPHA) * aco_fv
        data["fv"] = aco_fv

        # 2. IPR Adaptive Intercept - Tracks trend starting point
        ipr_int = data.get("ipr_int", 12000.0)
        if IPR in state.order_depths:
            mid = self._raw_mid(state.order_depths[IPR])
            if mid is not None:
                candidate = mid - (state.timestamp / 1000.0)
                ipr_int = (IPR_INTERCEPT_ALPHA * candidate) + (1.0 - IPR_INTERCEPT_ALPHA) * ipr_int
        data["ipr_int"] = ipr_int

        # 3. Context & Persistence
        ts = state.timestamp
        ticks = data.get("tt", 0) + 1
        data["tt"] = ticks
        is_last_day = (ticks > 20000) 
        is_endgame = (ts >= ENDGAME_TS)

        # 4. Product Execution
        if ACO in state.order_depths:
            result[ACO] = self._trade_aco(state.order_depths[ACO], state.position.get(ACO, 0), is_endgame, is_last_day, aco_fv)
        
        if IPR in state.order_depths:
            result[IPR] = self._trade_ipr(state.order_depths[IPR], state.position.get(IPR, 0), is_endgame, is_last_day, ts)

        # Serialize with minimal whitespace to stay under traderData char limits
        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_ipr(self, depth: OrderDepth, pos: int, end: bool, last: bool, ts: int) -> List[Order]:
        orders = []
        if end and last:
            curr = pos
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if curr <= 0: break
                q = min(depth.buy_orders[bp], curr)
                orders.append(Order(IPR, int(bp), -q))
                curr -= q
            return orders

        bb, bv = self._best_bid(depth)
        ba, av = self._best_ask(depth)
        if bb is None or ba is None: return []

        raw_mid = 0.5 * (bb + ba)
        pop_mid = self._popular_mid(depth)
        lead = 0.0 if pop_mid is None else (pop_mid - raw_mid)
        imb = self._imbalance(bv, av)
        
        # High Conviction Signal: Requires stronger score to justify crossing 14-tick spread
        score = 0.8 * lead + 2.5 * imb
        target = POSITION_LIMIT - pos

        if target > 0:
            if score >= 2.0: # Refined threshold to reduce noise trades
                for ap, q in sorted(depth.sell_orders.items()):
                    if target <= 0: break
                    take = min(abs(q), target)
                    orders.append(Order(IPR, int(ap), take))
                    target -= take
            
            # Passive entry to capture drift while avoiding spread tax
            if target > 0 and bb + 1 < ba:
                post_qty = min(IPR_PASSIVE_CLIP, target)
                orders.append(Order(IPR, int(bb + 1), int(post_qty)))

        return orders

    def _trade_aco(self, depth: OrderDepth, pos: int, end: bool, last: bool, fv: float) -> List[Order]:
        orders = []
        if end and last:
            # Full liquidation for both directions
            if pos > 0:
                curr = pos
                for bp in sorted(depth.buy_orders.keys(), reverse=True):
                    if curr <= 0: break
                    q = min(depth.buy_orders[bp], curr)
                    orders.append(Order(ACO, int(bp), -q))
                    curr -= q
            elif pos < 0:
                curr = -pos
                for ap in sorted(depth.sell_orders.items()):
                    if curr <= 0: break
                    q = min(abs(depth.sell_orders[ap]), curr)
                    orders.append(Order(ACO, int(ap), q))
                    curr -= q
            return orders

        bb, bv = self._best_bid(depth)
        ba, av = self._best_ask(depth)
        if bb is None or ba is None: return []

        imb = self._imbalance(bv, av)
        spread = ba - bb
        jump_up = (spread <= ACO_JUMP_SPREAD and imb >= ACO_JUMP_IMB)
        jump_dn = (spread <= ACO_JUMP_SPREAD and imb <= -ACO_JUMP_IMB)

        skew = (pos / POSITION_LIMIT) * ACO_SKEW_FACTOR
        buy_thresh = fv - ACO_BASE_EDGE - skew
        sell_thresh = fv + ACO_BASE_EDGE - skew

        if jump_up: sell_thresh += ACO_JUMP_BIAS
        if jump_dn: buy_thresh -= ACO_JUMP_BIAS

        # Aggressive takes based on skewed thresholds
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

        # Passive quoting with safety guards for regime jumps
        if bb + 1 < ba - 1:
            if cap_long > 0 and not jump_dn:
                orders.append(Order(ACO, int(bb + 1), cap_long))
            if cap_short > 0 and not jump_up:
                orders.append(Order(ACO, int(ba - 1), -cap_short))

        return orders

    def _best_bid(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.buy_orders: return None, 0
        px = max(depth.buy_orders.keys())
        return int(px), int(depth.buy_orders[px])

    def _best_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.sell_orders: return None, 0
        px = min(depth.sell_orders.keys())
        return int(px), int(depth.sell_orders[px])

    def _raw_mid(self, depth: OrderDepth) -> Optional[float]:
        bb, _ = self._best_bid(depth)
        ba, _ = self._best_ask(depth)
        return 0.5 * (bb + ba) if bb and ba else None

    def _imbalance(self, bv: int, av: int) -> float:
        total = abs(bv) + abs(av)
        return (abs(bv) - abs(av)) / total if total > 0 else 0.0

    def _popular_mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders: return None
        pb = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        pa = max(depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]
        return 0.5 * (pb + pa)
