from __future__ import annotations
import json
from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import List, Dict

class Trader:
    def __init__(self):
        # Product configurations
        self.limits = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
        
        # OSMIUM: Fixed Fair Value (as per your analysis)
        self.osmium_fv = 10000
        
        # PEPPER: Simple Moving Average for Fair Value
        self.pepper_prices = []
        self.pepper_window = 20 # Faster than your previous 50

    def get_skew(self, pos, limit, intensity=2.0):
        """Calculates how many ticks to shift prices based on inventory."""
        return -1 * (pos / limit) * intensity

    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            # 1. Update Inventory and Fair Value
            current_pos = state.position.get(product, 0)
            limit = self.limits[product]
            
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is None or best_ask is None:
                continue
                
            mid_price = (best_bid + best_ask) / 2

            if product == "ASH_COATED_OSMIUM":
                fv = self.osmium_fv
            else: # PEPPER
                self.pepper_prices.append(mid_price)
                if len(self.pepper_prices) > self.pepper_window:
                    self.pepper_prices.pop(0)
                fv = sum(self.pepper_prices) / len(self.pepper_prices)

            # 2. Calculate Skewed Prices
            # If we are long, we lower our bid and lower our ask (to encourage selling)
            skew = self.get_skew(current_pos, limit, intensity=3.0)
            
            # Base spread: 1 tick for OSMIUM (stable), 2 for PEPPER (volatile)
            spread_buffer = 1 if product == "ASH_COATED_OSMIUM" else 2
            
            my_bid = round(fv - spread_buffer + skew)
            my_ask = round(fv + spread_buffer + skew)

            # 3. Ensure we don't cross the spread unless we intend to (Passive Only)
            # This prevents "leaking" by ensuring we stay at or behind top of book
            my_bid = min(my_bid, best_bid + 1 if current_pos < limit else best_bid)
            my_ask = max(my_ask, best_ask - 1 if current_pos > -limit else best_ask)

            # 4. Generate Orders (Passive Quoting)
            buy_vol = limit - current_pos
            sell_vol = -limit - current_pos # Negative for sells

            if buy_vol > 0:
                orders.append(Order(product, my_bid, buy_vol))
            if sell_vol < 0:
                orders.append(Order(product, my_ask, sell_vol))

            result[product] = orders

        return result
