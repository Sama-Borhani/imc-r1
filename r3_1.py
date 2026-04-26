import math
from typing import Dict, List
from datamodel import OrderDepth, UserId, TradingState, Order

# --- Math & Black-Scholes Utilities ---
def norm_cdf(x: float) -> float:
    # Fast approximation of the standard normal CDF
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    approx = 1.0 - 1.0 / math.sqrt(2 * math.pi) * math.exp(-x * x / 2.0) * \
             (0.319381530 * k - 0.356563782 * (k ** 2) + 1.781477937 * (k ** 3) - 
              1.821255978 * (k ** 4) + 1.330274429 * (k ** 5))
    if x < 0:
        return 1.0 - approx
    return approx

def norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)

def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0: return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0: return max(0.0, S - K)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm_cdf(_d1) - K * math.exp(-r * T) * norm_cdf(_d2)

def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0: return 1.0 if S > K else 0.0
    return norm_cdf(d1(S, K, T, r, sigma))

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0: return 0.0
    return S * math.sqrt(T) * norm_pdf(d1(S, K, T, r, sigma))

def implied_volatility_call(market_price: float, S: float, K: float, T: float, r: float) -> float:
    # Newton-Raphson method for IV
    if market_price <= max(0.0, S - K) or T <= 0:
        return 0.01 
    
    sigma = 0.2  # Initial guess
    for _ in range(15): # Capped iterations for performance
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-6:
            break
        diff = market_price - price
        if abs(diff) < 1e-3:
            break
        sigma += diff / vega
        sigma = max(0.001, min(sigma, 5.0)) # Bound volatility
    return sigma

class Trader:
    def __init__(self):
        self.underlying = "VELVETFRUIT_EXTRACT"
        # Focus strikes: Ignore deep ITM/OTM to avoid noise and save compute
        self.target_strikes = [5100, 5200, 5300, 5400, 5500]
        self.anchor_strike = 5300 
        self.r = 0.0  # Risk free rate
        
        self.iv_threshold = 0.015 # Trading signal threshold
        self.hydrogel_window = [] # For simple MA

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2.0
        return 0.0

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}
        
        # Calculate Time to Expiry (TTE) in years
        # Round 3 starts at 5 days. Each day is 1,000,000 timestamps.
        days_left = 5.0 - (state.timestamp / 1000000.0)
        T = max(0.0001, days_left / 252.0)

        # 1. Process Underlying Asset
        S = 0.0
        if self.underlying in state.order_depths:
            S = self.get_mid_price(state.order_depths[self.underlying])
            
        # 2. Market Making for Hydrogel Pack (Independent Strategy)
        if "HYDROGEL_PACK" in state.order_depths:
            hydro_depth = state.order_depths["HYDROGEL_PACK"]
            hydro_mid = self.get_mid_price(hydro_depth)
            
            if hydro_mid > 0:
                self.hydrogel_window.append(hydro_mid)
                if len(self.hydrogel_window) > 20:
                    self.hydrogel_window.pop(0)
                
                hydro_ma = sum(self.hydrogel_window) / len(self.hydrogel_window)
                hydro_pos = state.position.get("HYDROGEL_PACK", 0)
                hydro_orders = []
                
                # Mean reversion with inventory skew
                bid_price = math.floor(hydro_mid - 1)
                ask_price = math.ceil(hydro_mid + 1)
                
                if hydro_mid < hydro_ma: # Undervalued
                    bid_price = math.floor(hydro_mid) # Aggressive bid
                elif hydro_mid > hydro_ma: # Overvalued
                    ask_price = math.ceil(hydro_mid)  # Aggressive ask
                    
                bid_vol = 200 - hydro_pos
                ask_vol = -200 - hydro_pos
                
                if bid_vol > 0:
                    hydro_orders.append(Order("HYDROGEL_PACK", bid_price, bid_vol))
                if ask_vol < 0:
                    hydro_orders.append(Order("HYDROGEL_PACK", ask_price, ask_vol))
                    
                result["HYDROGEL_PACK"] = hydro_orders

        # 3. Process Options and Volatility Surface
        if S > 0:
            anchor_symbol = f"VEV_{self.anchor_strike}"
            anchor_iv = 0.15 # Default fallback
            
            # Calculate Baseline IV
            if anchor_symbol in state.order_depths:
                anchor_mid = self.get_mid_price(state.order_depths[anchor_symbol])
                if anchor_mid > 0:
                    anchor_iv = implied_volatility_call(anchor_mid, S, self.anchor_strike, T, self.r)
            
            portfolio_delta = 0.0
            
            # Evaluate relative value and generate option orders
            for strike in self.target_strikes:
                symbol = f"VEV_{strike}"
                if symbol not in state.order_depths: continue
                
                depth = state.order_depths[symbol]
                best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
                best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
                mid = self.get_mid_price(depth)
                
                if mid <= 0: continue
                
                # Calculate current IV and Delta
                current_iv = implied_volatility_call(mid, S, strike, T, self.r)
                current_delta = bs_call_delta(S, strike, T, self.r, current_iv)
                
                # Track total options delta based on current holdings
                pos = state.position.get(symbol, 0)
                portfolio_delta += pos * current_delta
                
                orders = []
                # Trading Logic: Mean reversion to baseline IV
                if current_iv > anchor_iv + self.iv_threshold and best_bid > 0:
                    # Overpriced - Sell to the market bid
                    vol_to_sell = max(-300 - pos, -depth.buy_orders.get(best_bid, 0))
                    if vol_to_sell < 0:
                        orders.append(Order(symbol, best_bid, vol_to_sell))
                        
                elif current_iv < anchor_iv - self.iv_threshold and best_ask > 0:
                    # Underpriced - Buy from the market ask
                    vol_to_buy = min(300 - pos, -depth.sell_orders.get(best_ask, 0))
                    if vol_to_buy > 0:
                        orders.append(Order(symbol, best_ask, vol_to_buy))
                        
                if orders:
                    result[symbol] = orders

            # 4. Execute Delta Hedge with Underlying
            # We want our total delta (options delta + underlying position) to be 0
            current_underlying_pos = state.position.get(self.underlying, 0)
            target_underlying_pos = -int(round(portfolio_delta))
            
            # Enforce underlying limits (-200 to 200)
            target_underlying_pos = max(-200, min(200, target_underlying_pos))
            
            delta_to_trade = target_underlying_pos - current_underlying_pos
            
            underlying_orders = []
            u_depth = state.order_depths[self.underlying]
            
            if delta_to_trade > 0:
                best_ask = min(u_depth.sell_orders.keys()) if u_depth.sell_orders else math.ceil(S)
                underlying_orders.append(Order(self.underlying, best_ask, delta_to_trade))
            elif delta_to_trade < 0:
                best_bid = max(u_depth.buy_orders.keys()) if u_depth.buy_orders else math.floor(S)
                underlying_orders.append(Order(self.underlying, best_bid, delta_to_trade))
                
            if underlying_orders:
                result[self.underlying] = underlying_orders

        # Return format expected by the engine
        conversions = 0
        trader_data = "DELTA_HEDGED_VOL_ARB_ACTIVE"
        return result, conversions, trader_data
