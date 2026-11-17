# src/inventory.py -- Inventory helpers used by notebooks/inventory_optimization.ipynb
"""Inventory helpers.

- eoq:           economic order quantity (Q)
- window_sum:    demand over the protection interval (next n days) for every day
- simulate_sS:   daily simulation of an (s, S) policy with lost sales
- kpis:          fill rate, stock-out days, average stock and costs after a warm-up

Daily sequence in simulate_sS (same order every day):
    1. receive orders that arrive today
    2. review: if inventory position (on hand + on order) <= s, order up to S = s + Q
    3. serve demand from stock on hand (unmet demand is lost)
    4. end-of-day stock is charged the holding cost
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit_year: float) -> int:
    """Economic order quantity: sqrt(2 * K * D / H), rounded up to a whole unit."""
    return int(math.ceil(math.sqrt(2 * order_cost * annual_demand / holding_cost_per_unit_year)))


def window_sum(values, n: int) -> np.ndarray:
    """Sum of values[t : t + n] for every t. Near the end of the array the window is shorter."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s[::-1].rolling(n, min_periods=1).sum()[::-1].to_numpy()


def simulate_sS(demand, reorder_point, order_qty: float, lead_time: int, review_period: int = 1) -> pd.DataFrame:
    """Simulate one product under an (s, S) policy with S = s + Q and lost sales.

    demand:        actual daily demand (array of length T)
    reorder_point: s for every day (array of length T); it can change from day to day
    order_qty:     Q (the EOQ)
    lead_time:     days between placing an order and receiving it
    Returns one row per day: received, ordered, on_hand (end of day), lost.
    Initial stock is s[0] + Q. An order that could not arrive before day T - 1 is not placed,
    because it can never serve demand inside the simulated window.
    """
    demand = np.asarray(demand, dtype=float)
    s = np.asarray(reorder_point, dtype=float)
    T = len(demand)
    on_hand, on_order = s[0] + order_qty, 0.0
    arrivals = np.zeros(T + lead_time + 1)
    received, ordered, stock, lost = (np.zeros(T) for _ in range(4))

    for t in range(T):
        on_hand += arrivals[t]                       # 1. receive
        on_order -= arrivals[t]
        received[t] = arrivals[t]
        if t % review_period == 0 and t + lead_time <= T - 1:   # 2. review
            position = on_hand + on_order
            if position <= s[t]:
                q = math.ceil(s[t] + order_qty - position)      # order up to S = s + Q
                arrivals[t + lead_time] += q
                on_order += q
                ordered[t] = q
        sold = min(on_hand, demand[t])               # 3. serve demand
        lost[t] = demand[t] - sold
        on_hand -= sold
        stock[t] = on_hand                           # 4. end-of-day stock
    return pd.DataFrame({"received": received, "ordered": ordered, "on_hand": stock, "lost": lost})


def kpis(sim: pd.DataFrame, demand, holding_per_unit_day: float, order_cost: float,
         stockout_cost_per_unit: float, warmup: int = 14) -> dict:
    """Service level and cost of one simulation. The first `warmup` days are ignored."""
    d = np.asarray(demand, dtype=float)[warmup:]
    s = sim.iloc[warmup:]
    n_orders = int((s["ordered"] > 0).sum())
    out = {
        "demand": d.sum(),
        "lost": s["lost"].sum(),
        "fill_rate": 1 - s["lost"].sum() / d.sum(),
        "stockout_days": int((s["lost"] > 0).sum()),
        "avg_on_hand": s["on_hand"].mean(),
        "orders": n_orders,
        "order_cost": n_orders * order_cost,
        "holding_cost": holding_per_unit_day * s["on_hand"].sum(),
        "stockout_cost": stockout_cost_per_unit * s["lost"].sum(),
    }
    out["total_cost"] = out["order_cost"] + out["holding_cost"] + out["stockout_cost"]
    return out
