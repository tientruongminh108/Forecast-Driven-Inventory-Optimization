# src/reporting.py -- Planner-facing outputs: replenishment plan, run metadata and the Excel workbook
"""Turns the simulation results and the forward forecast into files a planner can act on.

Inputs the plan needs
- forward forecast (mean per day) and the noise distribution of the final model (from the forecasting notebook)
- the current inventory position per product (on hand, on order). The dataset has no reliable position, so
  load_position() uses an optional data/current_position.csv and otherwise falls back to a clearly flagged placeholder
- optional item master (MOQ, pack size) in data/item_master.csv
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PLAN_COLUMNS = [
    "as_of_date", "product_id", "status", "on_hand", "on_order", "inventory_position",
    "reorder_point", "order_up_to", "safety_stock", "mean_demand_protection",
    "order_flag", "suggested_qty", "order_qty_rounded", "expected_arrival_date",
    "days_of_cover", "p_stockout_in_lead_time", "eoq", "moq", "pack_size",
    "lead_time_days", "review_days", "alpha", "position_source",
]


# Inputs

def round_order_qty(q: float, moq: float = 1, pack: float = 1) -> float:
    """Round a suggested order quantity up to the pack size and lift it to the MOQ. Zero stays zero."""
    if q <= 0:
        return 0.0
    q = max(q, moq)
    return float(math.ceil(q / pack) * pack)


def load_item_master(path: Path | str, products) -> pd.DataFrame:
    """MOQ and pack size per product (1 and 1 when the file or a product is missing)."""
    base = pd.DataFrame({"product_id": list(products), "moq": 1.0, "pack_size": 1.0})
    path = Path(path)
    if not path.exists():
        return base.set_index("product_id")
    im = pd.read_csv(path)
    missing = {"product_id", "moq", "pack_size"} - set(im.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    return base.drop(columns=["moq", "pack_size"]).merge(im[["product_id", "moq", "pack_size"]], on="product_id", how="left") \
               .fillna({"moq": 1.0, "pack_size": 1.0}).set_index("product_id")


def load_position(path: Path | str, raw: pd.DataFrame, products) -> tuple[pd.DataFrame, str]:
    """Current inventory position per product and a label saying where it came from.

    path (csv with product_id, on_hand, on_order) is the ERP snapshot. Without it the dataset's `stock` column on the
    last date is used as on hand and on order is set to zero. That column is unrelated to demand in this dataset
    (see the EDA), so the result is a PLACEHOLDER and is labelled as such everywhere it is shown."""
    path = Path(path)
    if path.exists():
        pos = pd.read_csv(path)
        missing = {"product_id", "on_hand", "on_order"} - set(pos.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
        pos = pos.set_index("product_id").reindex(list(products))
        if pos[["on_hand", "on_order"]].isna().any().any():
            raise ValueError(f"{path.name} does not cover every product: {list(products)}")
        return pos[["on_hand", "on_order"]], f"ERP snapshot ({path.name})"
    last = raw[raw["date"] == raw["date"].max()].set_index("product_id")["stock"].reindex(list(products))
    pos = pd.DataFrame({"on_hand": last.astype(float), "on_order": 0.0})
    return pos, "PLACEHOLDER: dataset stock column on the last date, nothing on order"


# Plan

def protection_quantile(mu, levels, ratios, n: int, alpha: float, n_draws: int, seed: int, lead: int):
    """Mean, alpha-quantile of the demand over the next n days, and the draws of the first `lead` days' demand.

    Same method as the backtest: daily noise ratios are drawn from the final model's training distribution and
    multiplied by the daily mean forecasts."""
    if len(mu) < n:
        raise ValueError(f"Forward forecast has {len(mu)} days but the protection interval needs {n}.")
    rng = np.random.default_rng(seed)
    draws = np.interp(rng.random((n_draws, n)), levels, ratios) * np.asarray(mu[:n], float)
    total = draws.sum(axis=1)
    lead_total = draws[:, :lead].sum(axis=1) if lead > 0 else np.zeros(n_draws)
    return float(np.sum(mu[:n])), float(np.quantile(total, alpha)), lead_total


def build_replenishment_plan(fwd: pd.DataFrame, noise: pd.DataFrame, unit_econ: pd.DataFrame, params: dict, alpha: float,
                             position: pd.DataFrame, item_master: pd.DataFrame, position_source: str,
                             n_draws: int = 20000, seed: int = 42) -> pd.DataFrame:
    """One row per product: reorder point, order-up-to level, status and a suggested order for the as-of date.

    fwd: forward forecast (date, product_id, demand_mean). noise: ratio quantiles of the final model (level, ratio).
    unit_econ: output of inventory.unit_economics (needs column Q). position / item_master: indexed by product_id.
    Rules (daily review): REORDER if inventory position <= s; OVERSTOCK if position > S = s + Q; otherwise OK.
    suggested_qty = ceil(S - position) on a REORDER, then rounded to MOQ / pack size."""
    L, R = int(params["L"]), int(params["R"])
    n = L + R
    levels, ratios = noise["level"].to_numpy(), noise["ratio"].to_numpy()
    as_of = pd.Timestamp(fwd["date"].min())
    rows = []
    for p in sorted(fwd["product_id"].unique()):
        mu = fwd[fwd["product_id"] == p].sort_values("date")["demand_mean"].to_numpy()
        mean_sum, s, lead_total = protection_quantile(mu, levels, ratios, n, alpha, n_draws, seed, L)
        Q = float(unit_econ.loc[p, "Q"])
        on_hand, on_order = float(position.loc[p, "on_hand"]), float(position.loc[p, "on_order"])
        ip = on_hand + on_order
        S = s + Q
        status = "REORDER" if ip <= s else ("OVERSTOCK" if ip > S else "OK")
        suggested = float(math.ceil(S - ip)) if status == "REORDER" else 0.0
        moq, pack = float(item_master.loc[p, "moq"]), float(item_master.loc[p, "pack_size"])
        rows.append({
            "as_of_date": as_of.date(), "product_id": p, "status": status,
            "on_hand": on_hand, "on_order": on_order, "inventory_position": ip,
            "reorder_point": s, "order_up_to": S, "safety_stock": s - mean_sum, "mean_demand_protection": mean_sum,
            "order_flag": status == "REORDER", "suggested_qty": suggested,
            "order_qty_rounded": round_order_qty(suggested, moq, pack),
            "expected_arrival_date": (as_of + pd.Timedelta(days=L)).date() if status == "REORDER" else pd.NaT,
            "days_of_cover": on_hand / float(np.mean(mu[:7])),
            "p_stockout_in_lead_time": float((lead_total > ip).mean()) if L > 0 else 0.0,
            "eoq": Q, "moq": moq, "pack_size": pack, "lead_time_days": L, "review_days": R, "alpha": alpha,
            "position_source": position_source,
        })
    return pd.DataFrame(rows)[PLAN_COLUMNS]


# Run metadata

def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def run_metadata(root: Path, as_of_date, data_cutoff, policy: str, alpha: float, params: dict, position_source: str,
                 promo_assumption: str, forecast_model: str, n_draws: int, seed: int, extra: dict | None = None) -> dict:
    """Everything needed to reproduce or audit one plan."""
    p = {k: (v if np.isscalar(v) else {str(i): float(x) for i, x in v.items()}) for k, v in params.items()}
    meta = {
        "run_id": f"{pd.Timestamp(as_of_date):%Y%m%d}_{policy.split()[0].lower()}_a{alpha:g}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of_date).date()), "data_cutoff": str(pd.Timestamp(data_cutoff).date()),
        "forecast_model": forecast_model, "policy": policy, "alpha": float(alpha), "parameters": p,
        "position_source": position_source, "promotion_assumption": promo_assumption,
        "monte_carlo": {"draws": n_draws, "seed": seed},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "git_commit": _git_commit(root),
    }
    meta.update(extra or {})
    return meta


def save_json(obj: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


# Excel workbook

def _format_sheet(ws, df: pd.DataFrame, pct_cols=(), status_col: str | None = None) -> None:
    """Header style, freeze panes, filters, column widths, number formats and (optionally) status colours."""
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="1F3A5F")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), head_fill
        c.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "C2" if "product_id" in df.columns else "A2"
    ws.auto_filter.ref = ws.dimensions
    for j, col in enumerate(df.columns, start=1):
        letter = get_column_letter(j)
        width = max([len(str(col))] + [len(f"{v:,.1f}") if isinstance(v, float) else len(str(v)) for v in df[col].head(200)])
        ws.column_dimensions[letter].width = min(max(width + 2, 9), 46)
        if col in pct_cols:
            fmt = "0.0%"
        elif col == "alpha":
            fmt = "0.000"
        elif pd.api.types.is_float_dtype(df[col]):
            fmt = "#,##0.0"
        elif pd.api.types.is_integer_dtype(df[col]):
            fmt = "#,##0"
        else:
            fmt = None
        if fmt:
            for cell in ws[letter][1:]:
                cell.number_format = fmt
    if status_col and status_col in df.columns:
        letter = get_column_letter(list(df.columns).index(status_col) + 1)
        rng = f"{letter}2:{letter}{len(df) + 1}"
        for text, color in (("REORDER", "F8CBAD"), ("OVERSTOCK", "FFE699"), ("OK", "C6E0B4")):
            ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{text}"'],
                                                          fill=PatternFill("solid", bgColor=color, fgColor=color)))


def write_workbook(path: Path | str, summary_info: list[tuple[str, object]], summary_table: pd.DataFrame,
                   sheets: dict[str, pd.DataFrame], pct_cols: dict[str, Sequence[str]] | None = None,
                   status_sheets: Sequence[str] = ("Plan", "Exceptions")) -> Path:
    """Write the planner workbook: a Summary sheet (run info + KPI table) followed by one sheet per DataFrame."""
    from openpyxl.styles import Alignment, Font
    pct_cols = pct_cols or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame().to_excel(xl, sheet_name="Summary")
        ws = xl.sheets["Summary"]
        ws["A1"], ws["A1"].font = "Replenishment report", Font(bold=True, size=14)
        r = 3
        for key, val in summary_info:
            ws.cell(r, 1, key).font = Font(bold=True)
            cell = ws.cell(r, 2, val if isinstance(val, (int, float, str)) else str(val))
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if isinstance(val, str) and val.startswith("PLACEHOLDER"):
                cell.font = Font(bold=True, color="C00000")          # inputs that are not real data must be visible
            r += 1
        r += 1
        ws.cell(r, 1, "Policy comparison (Monte Carlo, base case)").font = Font(bold=True, size=12)
        r += 1
        summary_table.to_excel(xl, sheet_name="Summary", startrow=r - 1, index=False)
        for c in ws[r]:
            c.font = Font(bold=True)
        for j, col in enumerate(summary_table.columns, start=1):
            fmt = "0.0%" if ("fill" in col.lower() and "rate" in col.lower()) else \
                  ("#,##0.0" if pd.api.types.is_float_dtype(summary_table[col]) else None)
            if fmt:
                for i in range(len(summary_table)):
                    ws.cell(r + 1 + i, j).number_format = fmt
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 60
        for letter in "CDEFG":
            ws.column_dimensions[letter].width = 18
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
            _format_sheet(xl.sheets[name], df, pct_cols.get(name, ()), status_col="status" if name in status_sheets else None)
    return path
