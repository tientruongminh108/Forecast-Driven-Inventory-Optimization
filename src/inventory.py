# src/inventory.py -- Policy construction, simulation and optimisation helpers for the inventory notebook
"""Everything the inventory notebook computes that is not a table or a figure.

Design rules
- No module-level state: all data the functions need travels in a ForecastBundle (built once by build_bundle).
- protection_stats is memoised on (L, R, alphas) inside the bundle, so scenarios that share the same lead time
  and review period reuse the same Monte Carlo quantiles. Treat the returned arrays as read-only.
- simulate (full daily arrays), simulate_cost (scalar reference) and simulate_cost_batch (vectorised over
  scenarios) implement the SAME daily dynamics; tests/test_inventory.py checks that they agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import norm

# Policy labels and the key under which protection_stats stores each reorder-point family
NAIVE, CLAIRVOYANT = "Naive (no safety stock)", "Clairvoyant (perfect foresight)"
CLASSICAL, CVNORMAL, QUANTILE = "Classical (constant sigma)", "Normal (CV-scaled sigma)", "Quantile (multiplicative)"
DAYS_OF_COVER = "Days of cover (trailing mean)"
POLICIES = (CLASSICAL, CVNORMAL, QUANTILE)
KEY = {CLASSICAL: "classical", CVNORMAL: "cvnormal", QUANTILE: "quantile"}

DEFAULT_ALPHAS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)


@dataclass(frozen=True)
class SimConfig:
    """Simulation settings shared by every function in this module."""
    warmup: int = 30                    # days excluded from all metrics
    alphas: tuple = DEFAULT_ALPHAS      # service-level grid
    n_draws: int = 20000                # Monte Carlo draws for protection-interval demand
    seed: int = 42


@dataclass
class ForecastBundle:
    """Forecast hand-off data, validated and indexed once, plus the memo cache for protection_stats."""
    fc: pd.DataFrame
    ratio_q: pd.DataFrame
    scale: pd.Series                    # additive sigma of the training error, per fold
    cv: pd.Series                       # coefficient of variation of actual / fitted, per fold
    products: list
    folds: list
    horizon: int                        # days per fold
    lookahead: int                      # mean-forecast days beyond each fold
    n_days: int
    dates: np.ndarray
    act: dict                           # product -> actual demand over the simulation window
    promo: dict
    month: np.ndarray
    mu: dict                            # product -> fold -> mean forecast (test days followed by look-ahead days)
    econ: pd.DataFrame                  # per-product price and daily demand from the pre-window history
    cfg: SimConfig = field(default_factory=SimConfig)
    _cache: dict = field(default_factory=dict, repr=False)


# Setup

def _mean_blocks(fc, look, product, folds, lookahead):
    """Per fold: mean forecast for the test days followed by the look-ahead days (same fold's model)."""
    blocks = {}
    for k in folds:
        test_mu = fc[(fc["product_id"] == product) & (fc["fold"] == k)]["demand_mean"].to_numpy()
        ext_mu = look[(look["product_id"] == product) & (look["fold"] == k)].sort_values("date")["demand_mean"].to_numpy()
        if len(ext_mu) < lookahead:   # last fold: the data end, so carry the recent average forward
            ext_mu = np.concatenate([ext_mu, np.full(lookahead - len(ext_mu), test_mu[-7:].mean())])
        blocks[k] = np.concatenate([test_mu, ext_mu])
    return blocks


def build_bundle(raw: pd.DataFrame, fc: pd.DataFrame, ratio_q: pd.DataFrame, scale_tbl: pd.DataFrame,
                 look: pd.DataFrame, cfg: SimConfig | None = None) -> ForecastBundle:
    """Validate the forecasting hand-off files and build the ForecastBundle.

    raw: the raw dataset (for unit price and demand level before the simulation window).
    fc, ratio_q, scale_tbl, look: demand_forecasts, ratio_quantiles, training_error_scale, mean_forecast_lookahead.
    """
    cfg = cfg or SimConfig()
    fc = fc.sort_values(["product_id", "date"]).reset_index(drop=True)
    scale_tbl = scale_tbl.set_index("fold") if "fold" in scale_tbl.columns else scale_tbl
    products = sorted(fc["product_id"].unique())
    folds = sorted(fc["fold"].unique())
    horizon = int(fc.groupby(["product_id", "fold"]).size().iloc[0])
    lookahead = int(look.groupby(["fold", "product_id"]).size().max())
    n_days = fc["date"].nunique()

    dates = fc[fc["product_id"] == products[0]]["date"].to_numpy()
    for p in products:                                          # every product must cover the same dates in the same order
        if not (fc[fc["product_id"] == p]["date"].to_numpy() == dates).all():
            raise ValueError(f"Product {p} does not cover the same dates as product {products[0]}")

    history = raw[raw["date"] < fc["date"].min()]
    econ = history.groupby("product_id").agg(price=("price", "mean"), daily_demand=("demand", "mean"))

    return ForecastBundle(
        fc=fc, ratio_q=ratio_q, scale=scale_tbl["resid_std"], cv=scale_tbl["ratio_std"],
        products=products, folds=folds, horizon=horizon, lookahead=lookahead, n_days=n_days, dates=dates,
        act={p: fc[fc["product_id"] == p]["actual_demand"].to_numpy(float) for p in products},
        promo={p: fc[fc["product_id"] == p]["promotion"].to_numpy() for p in products},
        month=pd.DatetimeIndex(dates).month.to_numpy(),
        mu={p: _mean_blocks(fc, look, p, folds, lookahead) for p in products},
        econ=econ, cfg=cfg)


def unit_economics(econ: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per-product cost, holding cost per day, stock-out penalty and EOQ for one parameter set.

    params['cost_ratio'] may be a scalar or a Series indexed by product."""
    e = econ.copy()
    e["cost"] = params["cost_ratio"] * e["price"]
    e["h_day"] = params["holding_rate"] * e["cost"] / 365                 # holding cost per unit per day
    e["penalty"] = (e["price"] - e["cost"]) * (1 + params["goodwill"])    # cost per unit of lost sales
    e["annual_demand"] = e["daily_demand"] * 365
    e["Q"] = np.ceil(np.sqrt(2 * params["K"] * e["annual_demand"] / (params["holding_rate"] * e["cost"])))   # EOQ
    return e


# Demand over the protection interval

def window_sums(arr, n):
    """Sum of the next n values from each position (shorter at the end of the array)."""
    return pd.Series(arr[::-1]).rolling(n, min_periods=1).sum().to_numpy()[::-1]


def protection_stats(b: ForecastBundle, L: int, R: int, alphas=None) -> dict:
    """For every product and day: mean demand over the protection interval L + R, and reorder-point candidates.

    Memoised on (L, R, alphas): the result is shared between callers, so do not modify the arrays."""
    alphas = list(b.cfg.alphas if alphas is None else alphas)
    key = (int(L), int(R), tuple(float(a) for a in alphas))
    if key in b._cache:
        return b._cache[key]
    n = L + R
    if n > b.lookahead + 1:
        raise ValueError(f"Protection interval L + R = {n} needs a mean-forecast look-ahead of at least {n - 1} days, "
                         f"but the forecast file holds {b.lookahead}. Increase LOOKAHEAD in the forecasting notebook.")
    rng = np.random.default_rng(b.cfg.seed)
    z = norm.ppf(alphas)[:, None]
    parts = {p: {"mean": [], "quantile": [], "classical": [], "cvnormal": []} for p in b.products}
    for k in b.folds:
        lv = b.ratio_q[b.ratio_q["fold"] == k].sort_values("level")
        draws = np.interp(rng.random((b.cfg.n_draws, n)), lv["level"].to_numpy(), lv["ratio"].to_numpy())   # noise ratios
        for p in b.products:
            W = np.lib.stride_tricks.sliding_window_view(b.mu[p][k], n)[:b.horizon]      # (days, n) mean forecasts
            mean_sum = W.sum(axis=1)
            parts[p]["mean"].append(mean_sum)
            parts[p]["quantile"].append(np.quantile(draws @ W.T, alphas, axis=0))     # (alphas, days)
            parts[p]["classical"].append(mean_sum + z * b.scale[k] * np.sqrt(n))
            parts[p]["cvnormal"].append(mean_sum + z * b.cv[k] * np.sqrt((W ** 2).sum(axis=1)))   # sd of a sum of independent days with sd_t = CV x mean_t
    out = {p: {"mean": np.concatenate(v["mean"]),
               "quantile": np.concatenate(v["quantile"], axis=1),
               "classical": np.concatenate(v["classical"], axis=1),
               "cvnormal": np.concatenate(v["cvnormal"], axis=1),
               "clairvoyant": window_sums(b.act[p], n)} for p, v in parts.items()}
    b._cache[key] = out
    return out


# Simulators (identical daily dynamics)

def simulate(actual, s, Q, L, R, h, K):
    """Simulate one product under an (s, S = s + Q) policy with lost sales. Returns daily arrays."""
    n_days = len(actual)
    on_hand = s[0] + Q
    on_order = 0.0
    arrivals = np.zeros(n_days + L + 1)
    out = {k: np.zeros(n_days) for k in ("on_hand", "lost", "order_cost", "hold_cost", "received", "ordered")}
    out["initial"] = on_hand
    for i in range(n_days):
        on_hand += arrivals[i]
        on_order -= arrivals[i]
        out["received"][i] = arrivals[i]
        if i % R == 0 and i + L <= n_days - 1:                 # review day; the order must arrive within the horizon
            position = on_hand + on_order
            if position <= s[i]:
                q = np.ceil(s[i] + Q - position)
                arrivals[i + L] += q
                on_order += q
                out["order_cost"][i] = K
                out["ordered"][i] = q
        sales = min(on_hand, actual[i])
        out["lost"][i] = actual[i] - sales
        on_hand -= sales
        out["on_hand"][i] = on_hand
        out["hold_cost"][i] = h * on_hand
    return out


def simulate_cost(actual, s, Q, L, R, h, K, warmup=30):
    """Scalar reference: same dynamics as simulate(), but returns only (order cost, holding cost, lost units,
    mean on-hand) after the warm-up. Plain Python lists make it several times faster than simulate()."""
    n = len(actual)
    on_hand, on_order = s[0] + Q, 0.0
    arrivals = [0.0] * (n + L + 1)
    order_cost = hold = lost = stock = 0.0
    for i in range(n):
        a = arrivals[i]
        on_hand += a
        on_order -= a
        ordered = False
        if i % R == 0 and i + L <= n - 1:
            position = on_hand + on_order
            if position <= s[i]:
                q = math.ceil(s[i] + Q - position)
                arrivals[i + L] += q
                on_order += q
                ordered = True
        demand = actual[i]
        sold = on_hand if on_hand < demand else demand
        on_hand -= sold
        if i >= warmup:
            lost += demand - sold
            hold += h * on_hand
            stock += on_hand
            if ordered:
                order_cost += K
    return order_cost, hold, lost, stock / (n - warmup)


def simulate_cost_batch(actual, s, Q, L, R, h, K, warmup=30):
    """simulate_cost() for many scenarios at once.

    actual, s: arrays of shape (B, n_days), one row per scenario (a demand path combined with a reorder-point series).
    Q, h, K: scalars. Returns four arrays of length B: (order cost, holding cost, lost units, mean on-hand).
    The loop runs over days only; every day is one numpy operation across all B scenarios."""
    actual, s = np.asarray(actual, float), np.asarray(s, float)
    n_scen, n = actual.shape
    on_hand = s[:, 0] + Q
    on_order = np.zeros(n_scen)
    arrivals = np.zeros((n_scen, n + L + 1))
    order_cost = np.zeros(n_scen)
    hold = np.zeros(n_scen)
    lost = np.zeros(n_scen)
    stock = np.zeros(n_scen)
    for i in range(n):
        a = arrivals[:, i]
        on_hand = on_hand + a
        on_order = on_order - a
        ordered = None
        if i % R == 0 and i + L <= n - 1:
            position = on_hand + on_order
            ordered = position <= s[:, i]
            q = np.where(ordered, np.ceil(s[:, i] + Q - position), 0.0)
            arrivals[:, i + L] += q
            on_order = on_order + q
        demand = actual[:, i]
        sold = np.minimum(on_hand, demand)
        on_hand = on_hand - sold
        if i >= warmup:
            lost += demand - sold
            hold += h * on_hand
            stock += on_hand
            if ordered is not None:
                order_cost += K * ordered
    return order_cost, hold, lost, stock / (n - warmup)


# Policies on the real path

def policy_series(b: ForecastBundle, p, st):
    """Yield (policy, alpha, reorder-point series) for one product."""
    yield NAIVE, np.nan, st[p]["mean"]
    yield CLAIRVOYANT, np.nan, st[p]["clairvoyant"]
    for j, a in enumerate(b.cfg.alphas):
        yield CLASSICAL, a, st[p]["classical"][j]
        yield CVNORMAL, a, st[p]["cvnormal"][j]
        yield QUANTILE, a, st[p]["quantile"][j]


def evaluate(b: ForecastBundle, params: dict):
    """Simulate every policy and product under one parameter set on the real path.

    Returns (summary per policy x alpha x product, cost per fold block, daily arrays for the diagnostics)."""
    L, R = params["L"], params["R"]
    e = unit_economics(b.econ, params)
    st = protection_stats(b, L, R)
    ev = np.arange(b.n_days) >= b.cfg.warmup
    fold_of_day = np.arange(b.n_days) // b.horizon
    rows, blocks, daily = [], [], {}
    for p in b.products:
        for pol, alpha, s in policy_series(b, p, st):
            sim = simulate(b.act[p], s, e.loc[p, "Q"], L, R, e.loc[p, "h_day"], params["K"])
            day_cost = sim["order_cost"] + sim["hold_cost"] + e.loc[p, "penalty"] * sim["lost"]
            rows.append({
                "policy": pol, "alpha": alpha, "product": p,
                "demand": b.act[p][ev].sum(), "lost": sim["lost"][ev].sum(),
                "order_cost": sim["order_cost"][ev].sum(), "hold_cost": sim["hold_cost"][ev].sum(),
                "stockout_cost": e.loc[p, "penalty"] * sim["lost"][ev].sum(),
                "stockout_days": (sim["lost"][ev] > 0).sum(), "days": ev.sum(),
                "avg_on_hand": sim["on_hand"][ev].mean(), "orders": (sim["order_cost"][ev] > 0).sum(),
            })
            for k in b.folds:
                m = ev & (fold_of_day == k - 1)
                blocks.append({"policy": pol, "alpha": alpha, "product": p, "fold": k, "cost": day_cost[m].sum()})
            daily[(pol, alpha, p)] = {"lost": sim["lost"], "on_hand": sim["on_hand"]}
    summary = pd.DataFrame(rows)
    summary["total_cost"] = summary["order_cost"] + summary["hold_cost"] + summary["stockout_cost"]
    return summary, pd.DataFrame(blocks), daily


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Add up the products for each policy and alpha."""
    g = summary.assign(_a=summary["alpha"].fillna(-1)).groupby(["policy", "_a"])
    agg = g[["demand", "lost", "order_cost", "hold_cost", "stockout_cost", "total_cost",
             "stockout_days", "days", "avg_on_hand", "orders"]].sum().reset_index()
    agg["alpha"] = agg["_a"].replace(-1, np.nan)
    agg["fill_rate"] = 1 - agg["lost"] / agg["demand"]
    agg["stockout_day_share"] = agg["stockout_days"] / agg["days"]
    return agg.drop(columns=["_a", "stockout_days", "days"])


# Monte Carlo over simulated demand paths

def simulate_paths(b: ForecastBundle, n_paths: int, seed: int | None = None) -> dict:
    """Demand paths from the fitted model: each day's forecast mean times a noise ratio drawn from that fold's training distribution."""
    rng = np.random.default_rng(b.cfg.seed + 1 if seed is None else seed)
    out = {p: np.empty((n_paths, b.n_days)) for p in b.products}
    for k in b.folds:
        lv = b.ratio_q[b.ratio_q["fold"] == k].sort_values("level")
        sl = slice((k - 1) * b.horizon, k * b.horizon)
        for p in b.products:
            noise = np.interp(rng.random((n_paths, b.horizon)), lv["level"].to_numpy(), lv["ratio"].to_numpy())
            out[p][:, sl] = np.rint(b.mu[p][k][:b.horizon] * noise)
    return out


def mc_costs(b: ForecastBundle, paths: dict, params: dict, policies=POLICIES, q_mult: float = 1.0,
             n_paths: int | None = None, alphas=None) -> dict:
    """Replay every (policy, alpha, product) against each demand path.

    Returns arrays of shape (path, policy, alpha, product): total cost, average on-hand units and lost units.
    The paths x alphas scenarios of one product and policy run together in simulate_cost_batch."""
    alphas = list(b.cfg.alphas if alphas is None else alphas)
    e = unit_economics(b.econ, params)
    st = protection_stats(b, params["L"], params["R"], alphas)
    n_paths = n_paths or next(iter(paths.values())).shape[0]
    n_alpha = len(alphas)
    shape = (n_paths, len(policies), n_alpha, len(b.products))
    out = {k: np.zeros(shape) for k in ("cost", "inv", "lost")}
    for ip, p in enumerate(b.products):
        Q, h, pen = e.loc[p, "Q"] * q_mult, e.loc[p, "h_day"], e.loc[p, "penalty"]
        actual = np.repeat(paths[p][:n_paths], n_alpha, axis=0)         # rows: path-major, alpha-minor
        for ipol, pol in enumerate(policies):
            s = np.tile(st[p][KEY[pol]], (n_paths, 1))                  # the same row order
            oc, hc, ls, inv = simulate_cost_batch(actual, s, Q, params["L"], params["R"], h, params["K"], b.cfg.warmup)
            out["cost"][:, ipol, :, ip] = (oc + hc + pen * ls).reshape(n_paths, n_alpha)
            out["inv"][:, ipol, :, ip] = inv.reshape(n_paths, n_alpha)
            out["lost"][:, ipol, :, ip] = ls.reshape(n_paths, n_alpha)
    return out


def path_demand(b: ForecastBundle, paths: dict, n_paths: int | None = None) -> float:
    """Mean total demand per path after the warm-up (denominator of the Monte Carlo fill rate)."""
    n_paths = n_paths or next(iter(paths.values())).shape[0]
    return float(np.mean([sum(paths[p][r, b.cfg.warmup:].sum() for p in b.products) for r in range(n_paths)]))


def mc_table(b: ForecastBundle, mc: dict, paths: dict, policies=POLICIES, alphas=None) -> pd.DataFrame:
    """Per policy: the alpha with the lowest MEAN cost over the paths, and what that setting delivers."""
    alphas = list(b.cfg.alphas if alphas is None else alphas)
    total = mc["cost"].sum(axis=3)                                  # (path, policy, alpha)
    mean = total.mean(axis=0)
    demand = path_demand(b, paths, total.shape[0])
    rows = []
    for i, pol in enumerate(policies):
        j = int(mean[i].argmin())
        rows.append({"policy": pol, "alpha*": alphas[j], "mean cost": mean[i, j],
                     "std across paths": total[:, i, j].std(ddof=1),
                     "avg inventory (units)": mc["inv"].sum(axis=3).mean(axis=0)[i, j],
                     "fill rate": 1 - mc["lost"].sum(axis=3).mean(axis=0)[i, j] / demand})
    return pd.DataFrame(rows).set_index("policy")


def paired_summary(ref_cost, new_cost) -> dict:
    """Paired comparison of two cost vectors over the same paths. Positive diff = `new` is cheaper."""
    d = np.asarray(ref_cost) - np.asarray(new_cost)
    base = np.asarray(ref_cost).mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": d.mean(), "pct": 100 * d.mean() / base, "se": se, "se_pct": 100 * se / base,
            "share_new_cheaper": (d > 0).mean(), "paths": d}


def mc_paired(mc: dict, policies, ref, new) -> dict:
    """Paired difference cost(ref) - cost(new) over the paths, each at its own mean-cost-minimising alpha. Positive = `new` is cheaper."""
    total = mc["cost"].sum(axis=3)
    mean = total.mean(axis=0)
    ir, inw = policies.index(ref), policies.index(new)
    return paired_summary(total[:, ir, int(mean[ir].argmin())], total[:, inw, int(mean[inw].argmin())])


def nested_alpha(blocks: pd.DataFrame, pol: str) -> dict:
    """Choose alpha on folds 1-3 and score it on folds 4-6, and vice versa."""
    b = blocks[blocks["policy"] == pol].groupby(["alpha", "fold"])["cost"].sum().unstack("fold")
    first, second = [1, 2, 3], [4, 5, 6]
    a1, a2 = b[first].sum(axis=1).idxmin(), b[second].sum(axis=1).idxmin()
    return {"policy": pol, "alpha chosen on folds 1-3": a1, "alpha chosen on folds 4-6": a2,
            "cost, alpha chosen in-sample": b.sum(axis=1).min(),
            "cost, alpha chosen out-of-sample": b.loc[a1, second].sum() + b.loc[a2, first].sum()}


# Incumbent benchmark: days of cover on a trailing average (no forecast model)

def trailing_mean(demand, window: int = 28, first=None) -> np.ndarray:
    """Mean demand of the previous `window` days at each day (expanding at the start, never includes today).

    demand: array (n_days,) or (n_scenarios, n_days). Day 0 has no history: it takes `first` (default: its own value)."""
    d = np.atleast_2d(np.asarray(demand, float))
    n_days = d.shape[1]
    cs = np.concatenate([np.zeros((d.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)
    t = np.arange(n_days)
    start = np.maximum(0, t - window)
    tm = (cs[:, t] - cs[:, start]) / np.maximum(t - start, 1)
    tm[:, 0] = d[:, 0] if first is None else first
    return tm


def doc_costs(b: ForecastBundle, paths: dict, params: dict, k_grid: Sequence[float], window: int = 28,
              n_paths: int | None = None) -> dict:
    """Replay a days-of-cover rule s = (L + R + k) x trailing mean demand on each path, for every k in k_grid.

    Returns arrays of shape (path, k, product): total cost, average on-hand units and lost units."""
    e = unit_economics(b.econ, params)
    n = params["L"] + params["R"]
    n_paths = n_paths or next(iter(paths.values())).shape[0]
    ks = np.asarray(k_grid, float)
    out = {k_: np.zeros((n_paths, len(ks), len(b.products))) for k_ in ("cost", "inv", "lost")}
    for ip, p in enumerate(b.products):
        Q, h, pen = e.loc[p, "Q"], e.loc[p, "h_day"], e.loc[p, "penalty"]
        actual = paths[p][:n_paths]
        tm = trailing_mean(actual, window, first=b.mu[p][b.folds[0]][0])             # (n_paths, n_days)
        s = ((n + ks)[None, :, None] * tm[:, None, :]).reshape(n_paths * len(ks), -1)   # path-major, k-minor
        oc, hc, ls, inv = simulate_cost_batch(np.repeat(actual, len(ks), axis=0), s, Q, params["L"], params["R"],
                                              h, params["K"], b.cfg.warmup)
        out["cost"][:, :, ip] = (oc + hc + pen * ls).reshape(n_paths, len(ks))
        out["inv"][:, :, ip] = inv.reshape(n_paths, len(ks))
        out["lost"][:, :, ip] = ls.reshape(n_paths, len(ks))
    return out


def doc_real(b: ForecastBundle, params: dict, k_grid: Sequence[float], window: int = 28) -> pd.DataFrame:
    """The days-of-cover rule on the one real path, summed over products, one row per k."""
    e = unit_economics(b.econ, params)
    n = params["L"] + params["R"]
    rows = []
    demand = sum(b.act[p][b.cfg.warmup:].sum() for p in b.products)
    for k in k_grid:
        cost = hold = lost = inv = orders = 0.0
        for p in b.products:
            tm = trailing_mean(b.act[p], window, first=b.mu[p][b.folds[0]][0])[0]
            oc, hc, ls, iv = simulate_cost(b.act[p], (n + k) * tm, e.loc[p, "Q"], params["L"], params["R"],
                                           e.loc[p, "h_day"], params["K"], b.cfg.warmup)
            cost += oc + hc + e.loc[p, "penalty"] * ls
            lost += ls
            inv += iv
        rows.append({"k (extra days of cover)": k, "total_cost": cost, "avg_on_hand": inv, "fill_rate": 1 - lost / demand})
    return pd.DataFrame(rows)


# Allocation of service levels across products (binary MILP)

def mc_tables(b: ForecastBundle, mc: dict, params: dict, i_pol: int):
    """Expected cost, inventory value and lost units per (product, alpha) from the Monte Carlo runs."""
    e = unit_economics(b.econ, params)
    cost = mc["cost"][:, i_pol].mean(axis=0).T
    inv_val = (mc["inv"][:, i_pol].mean(axis=0) * e["cost"].to_numpy()).T
    lost = mc["lost"][:, i_pol].mean(axis=0).T
    return cost, inv_val, lost


def allocate(cost, inv_val, lost, budget=None, max_lost=None, minimise="cost"):
    """Pick one alpha per product (binary MILP). Returns the chosen alpha index per product, or None if infeasible."""
    P, J = cost.shape
    c = (cost if minimise == "cost" else inv_val).ravel()
    cons = [LinearConstraint(np.kron(np.eye(P), np.ones((1, J))), 1, 1)]          # exactly one alpha per product
    if budget is not None:
        cons.append(LinearConstraint(inv_val.reshape(1, -1), -np.inf, budget))     # inventory-value budget
    if max_lost is not None:
        cons.append(LinearConstraint(lost.reshape(1, -1), -np.inf, max_lost))      # service level
    res = milp(c, constraints=cons, integrality=np.ones(P * J), bounds=Bounds(0, 1))
    return res.x.reshape(P, J).argmax(axis=1) if res.success else None


def uniform_alpha(cost, inv_val, lost, budget=None, max_lost=None, minimise="cost"):
    """Same alpha for every product: the best feasible one (the policy a planner would use without the model)."""
    P, J = cost.shape
    best = None
    for j in range(J):
        if budget is not None and inv_val[:, j].sum() > budget:
            continue
        if max_lost is not None and lost[:, j].sum() > max_lost:
            continue
        v = (cost if minimise == "cost" else inv_val)[:, j].sum()
        if best is None or v < best[0]:
            best = (v, j)
    return None if best is None else np.full(P, best[1])


def total_of(table, idx):
    return table[np.arange(len(idx)), idx].sum()


def frontier(cost, inv_val, lost, alphas, n: int = 18) -> pd.DataFrame:
    """Cost versus inventory-value budget: optimal allocation (MILP) against one common alpha."""
    alphas = list(alphas)
    tot = inv_val.sum(axis=0)
    rows = []
    for budget in np.linspace(tot[alphas.index(0.70)], tot.max(), n):
        a_m, a_u = allocate(cost, inv_val, lost, budget=budget), uniform_alpha(cost, inv_val, lost, budget=budget)
        rows.append({"budget": budget, "MILP": total_of(cost, a_m), "one common alpha": total_of(cost, a_u)})
    f = pd.DataFrame(rows)
    f["gain (%)"] = 100 * (1 - f["MILP"] / f["one common alpha"])
    return f


def real_path(b: ForecastBundle, alpha_idx, params: dict, alphas):
    """Cost, inventory value and fill rate of an alpha-per-product choice on the one real path."""
    e, st = unit_economics(b.econ, params), protection_stats(b, params["L"], params["R"], alphas)
    cost = value = lost = demand = 0.0
    for ip, p in enumerate(b.products):
        oc, hc, ls, inv = simulate_cost(b.act[p], st[p]["quantile"][alpha_idx[ip]], e.loc[p, "Q"], params["L"], params["R"],
                                        e.loc[p, "h_day"], params["K"], b.cfg.warmup)
        cost += oc + hc + e.loc[p, "penalty"] * ls
        value += inv * e.loc[p, "cost"]
        lost += ls
        demand += b.act[p][b.cfg.warmup:].sum()
    return cost, value, 1 - lost / demand
