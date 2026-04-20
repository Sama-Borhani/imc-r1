import json
from typing import Dict, List
from datamodel import Order, TradingState, OrderDepth

# ──────────────────────────────────────────────────────────────────
# STRATEGY SETTINGS
# ──────────────────────────────────────────────────────────────────
POSITION_LIMIT = 80
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ACO Parameters
ACO_DEFAULT_FAIR = 10_000
ACO_WINDOW_SIZE = 200     # Rolling window for dynamic Fair Value
ACO_BASE_EDGE = 3
ACO_SKEW_SCALE = 4

# Timing
ENDGAME_START = 99_500    # Start closing positions in the last 500 ticks

class Trader:

    def bid(self) -> int:
        """Market Access Fee bid."""
        return 7_500

    def run(self, state: TradingState):
        """
        Processes each tick. Manages dynamic state via traderData.
        REQUIRED RETURN: (orders_dict, conversions_int, traderData_str)
        """
        result: Dict[str, List[Order]] = {}
        
        # 1. Parse and update rolling mid-price history for ACO
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                pass
        
        aco_history = trader_data.get("aco_history", [])
        
        # Get current mid for ACO to update window
        if ACO in state.order_depths:
            depth = state.order_depths[ACO]
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                mid = (best_bid + best_ask) / 2.0
                aco_history.append(mid)
                # Keep window size at ~200
                if len(aco_history) > ACO_WINDOW_SIZE:
                    aco_history.pop(0)

        # 2. Process each product
        for product in [IPR, ACO]:
            if product not in state.order_depths:
                continue

            depth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == IPR:
                # Keep existing aggressive trend-following
                orders = self._trade_ipr(depth, pos, state.timestamp)
            else:
                # Use dynamic FV and endgame logic
                aco_fair = sum(aco_history) / len(aco_history) if aco_history else ACO_DEFAULT_FAIR
                orders = self._trade_aco(depth, pos, state.timestamp, aco_fair)

            if orders:
                result[product] = orders

        # 3. Serialize data for next tick
        trader_data["aco_history"] = aco_history
        new_trader_data = json.dumps(trader_data)

        return result, 0, new_trader_data

    def _trade_ipr(self, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """
        Optimally captures linear trend by hitting asks immediately.
        """
        orders = []
        
        # Endgame: Close out long
        if timestamp >= ENDGAME_START:
            if pos > 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                orders.append(Order(IPR, int(best_bid), -pos))
            return orders

        # Normal: Buy until limit is full
        remaining = POSITION_LIMIT - pos
        if remaining > 0 and depth.sell_orders:
            for ask_price, ask_vol in sorted(depth.sell_orders.items()):
                if remaining <= 0: break
                qty = min(abs(ask_vol), remaining)
                orders.append(Order(IPR, int(ask_price), qty))
                remaining -= qty
        return orders

    def _trade_aco(self, depth: OrderDepth, pos: int, timestamp: int, fair_value: float) -> List[Order]:
        """
        Skewed Market Making with Dynamic Fair Value and Passive Quoting experiment.
        """
        orders = []

        # 1. Endgame Closeout
        if timestamp >= ENDGAME_START:
            if pos > 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                return [Order(ACO, int(best_bid), -pos)]
            if pos < 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                return [Order(ACO, int(best_ask), -pos)]
            return []

        # 2. Normal Skew Logic
        skew = (pos / POSITION_LIMIT) * ACO_SKEW_SCALE
        buy_threshold = fair_value - ACO_BASE_EDGE - skew
        sell_threshold = fair_value + ACO_BASE_EDGE - skew

        # Aggressive Taking
        cap_long = POSITION_LIMIT - pos
        for ask_price, ask_vol in sorted(depth.sell_orders.items()):
            if ask_price <= buy_threshold and cap_long > 0:
                qty = min(abs(ask_vol), cap_long)
                orders.append(Order(ACO, int(ask_price), qty))
                cap_long -= qty
                pos += qty

        cap_short = POSITION_LIMIT + pos
        for bid_price, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
            if bid_price >= sell_threshold and cap_short > 0:
                qty = min(bid_vol, cap_short)
                orders.append(Order(ACO, int(bid_price), -qty))
                cap_short -= qty
                pos -= qty

        # 3. Passive Quoting Experiment
        # Quote at spread edges if there is remaining capacity
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else int(fair_value - 1)
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else int(fair_value + 1)

        cap_long = POSITION_LIMIT - pos
        if cap_long > 0:
            # Place passive bid at best_bid + 1
            orders.append(Order(ACO, best_bid + 1, cap_long))
        
        cap_short = POSITION_LIMIT + pos
        if cap_short > 0:
            # Place passive ask at best_ask - 1
            orders.append(Order(ACO, best_ask - 1, -cap_short))

        return orders
