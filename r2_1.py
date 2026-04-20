from __future__ import annotations
import json
from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import List, Dict

class Trader:
    def __init__(self):
        # Product configurations
        self.limits = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
        self.osmium_fv = 10000
        self.pepper_window = 20

    def get_skew(self, pos, limit, intensity=3.0):
        """Calculates price shift based on inventory to encourage flattening."""
        return -1 * (pos / limit) * intensity

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # PERSISTENCE: Retrieve pepper prices from traderData string
        pepper_prices = []
        if state.traderData:
            try:
                pepper_prices = json.loads(state.traderData)
            except:
                pepper_prices = []

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            current_pos = state.position.get(product, 0)
            limit = self.limits[product]
            
            # Get best market prices
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is None or best_ask is None:
                continue
                
            mid_price = (best_bid + best_ask) / 2

            # 1. Determine Fair Value (FV)
            if product == "ASH_COATED_OSMIUM":
                fv = self.osmium_fv
            else: # PEPPER
                pepper_prices.append(mid_price)
                if len(pepper_prices) > self.pepper_window:
                    pepper_prices.pop(0)
                fv = sum(pepper_prices) / len(pepper_prices)

            # 2. Calculate Skewed Quotes
            # High intensity skew (3.0) helps keep inventory near zero
            skew = self.get_skew(current_pos, limit, intensity=3.0)
            spread_buffer = 1 if product == "ASH_COATED_OSMIUM" else 2
            
            my_bid = int(round(fv - spread_buffer + skew))
            my_ask = int(round(fv + spread_buffer + skew))

            # 3. Passive Guardrails
            # Ensure we are always the 'maker' (passive), not the 'taker'
            my_bid = min(my_bid, best_bid) 
            my_ask = max(my_ask, best_ask)
            
            # Ensure my_bid is at least 1 tick below my_ask to avoid self-crossing
            if my_bid >= my_ask:
                my_bid = my_ask - 1

            # 4. Create Orders
            buy_vol = limit - current_pos
            sell_vol = -limit - current_pos # Negative for sells

            if buy_vol > 0:
                orders.append(Order(product, my_bid, buy_vol))
            if sell_vol < 0:
                orders.append(Order(product, my_ask, sell_vol))

            result[product] = orders

        # Serialize state for the next tick
        traderData = json.dumps(pepper_prices)

        return result, conversions, traderData
