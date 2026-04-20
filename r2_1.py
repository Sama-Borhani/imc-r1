from __future__ import annotations
import json
from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import List, Dict

class Trader:
    def __init__(self):
        self.limits = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
        # Windows: OSMIUM is stable (long window), PEPPER is fast (short window)
        self.windows = {"ASH_COATED_OSMIUM": 100, "INTARIAN_PEPPER_ROOT": 20}

    def get_skew(self, pos: int, limit: int, intensity: float = 2.0) -> float:
        return -1 * (pos / limit) * intensity

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # PERSISTENCE: Load historical prices for both products
        all_prices = {"ASH_COATED_OSMIUM": [], "INTARIAN_PEPPER_ROOT": []}
        if state.traderData:
            try:
                all_prices = json.loads(state.traderData)
            except:
                pass

        for product in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            if product not in state.order_depths: continue
            
            order_depth = state.order_depths[product]
            current_pos = state.position.get(product, 0)
            limit = self.limits[product]
            
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if not best_bid or not best_ask: continue
                
            mid_price = (best_bid + best_ask) / 2

            # 1. Update Moving Fair Value
            prices = all_prices.get(product, [])
            prices.append(mid_price)
            if len(prices) > self.windows[product]:
                prices.pop(0)
            all_prices[product] = prices
            fv = sum(prices) / len(prices)

            # 2. Competitive Pricing
            # Skew pushes us to flatten; spread_buffer keeps us profitable
            skew = self.get_skew(current_pos, limit, intensity=2.0)
            my_bid = int(round(fv - 1 + skew))
            my_ask = int(round(fv + 1 + skew))

            # 3. Queue Leadership Guardrails
            # Instead of matching the bid, we try to BE the best bid (best_bid + 1)
            # But we NEVER cross the ask (best_ask - 1)
            my_bid = max(my_bid, best_bid)     # Try to stay at the front
            my_bid = min(my_bid, best_ask - 1) # But don't "take"
            
            my_ask = min(my_ask, best_ask)     # Try to stay at the front
            my_ask = max(my_ask, best_bid + 1) # But don't "take"

            # 4. Order Generation
            orders = []
            if current_pos < limit:
                orders.append(Order(product, my_bid, limit - current_pos))
            if current_pos > -limit:
                orders.append(Order(product, my_ask, -limit - current_pos))

            result[product] = orders

        traderData = json.dumps(all_prices)
        return result, conversions, traderData
