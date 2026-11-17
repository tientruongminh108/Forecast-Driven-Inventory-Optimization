# src/utils.py -- Helper functions and configs
from __future__ import annotations
 
from pathlib import Path
from typing import Sequence
 
import numpy as np
import pandas as pd


# Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "inventory_demand_forecasting.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Constants
REQUIRED_COLUMNS = ["date", "product_id", "category", "price", "promotion", "stock", "demand"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Data loading and validation
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, parse dates, sort, and run basic sanity checks.
 
    Returns a DataFrame sorted by (product_id, date) with a clean RangeIndex.
    Raises ValueError if the data violates an assumption the notebooks rely on:
    required columns, no missing values, no duplicate (date, product_id), non-negative demand,
    the same dates for every product, and a continuous daily calendar.
    """
    df = pd.read_csv(path)
 
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
 
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
 
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values.")
    if df.duplicated(["date", "product_id"]).any():
        raise ValueError("Duplicate (date, product_id) rows found.")
    if (df["demand"] < 0).any():
        raise ValueError("Negative demand found.")
 
    n_dates_per_product = df.groupby("product_id")["date"].nunique()
    if n_dates_per_product.nunique() != 1:
        raise ValueError(f"Products have different numbers of dates:\n{n_dates_per_product}")

    # The rolling-window and look-ahead code works by position, so a calendar gap would silently
    # shift every window. Require one row per product for every day in the range.
    expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    if df["date"].nunique() != len(expected_days):
        missing = expected_days.difference(pd.DatetimeIndex(df["date"].unique()))
        raise ValueError(f"Dates are not a continuous daily range; {len(missing)} missing, e.g. {list(missing[:3].date)}")

    return df


# Features helpers
def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy of df with month, weekday, day-of-year and year columns."""
    out = df.copy()
    d = out[date_col].dt
    out["year"] = d.year
    out["month"] = d.month
    out["month_name"] = d.month_name().str[:3]      # English abbreviation regardless of the system locale (strftime is locale-dependent)
    out["day_of_week"] = d.dayofweek
    out["day_name"] = d.day_name()
    out["day_of_year"] = d.dayofyear
    return out


def add_fourier_terms(df: pd.DataFrame, order: int = 3, period: float = 365.25,
                      date_col: str = "date", prefix: str = "annual") -> pd.DataFrame:
    """Return a copy of df with sin/cos terms that encode a smooth annual cycle.
 
    Adds 2 * order columns: {prefix}_sin_1, {prefix}_cos_1, ..., {prefix}_sin_K, {prefix}_cos_K.
    The same calendar day gets the same values in every year.
    """
    out = df.copy()
    doy = out[date_col].dt.dayofyear.to_numpy()
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * doy / period
        out[f"{prefix}_sin_{k}"] = np.sin(angle)
        out[f"{prefix}_cos_{k}"] = np.cos(angle)
    return out


# Metrics
def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays and check that the two inputs have the same length."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|. NaN when the actuals sum to zero."""
    y_true, y_pred = _pair(y_true, y_pred)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def bias(y_true, y_pred) -> float:
    """Relative bias: (sum(pred) - sum(actual)) / sum(actual). Positive = over-forecast. NaN when the actuals sum to zero."""
    y_true, y_pred = _pair(y_true, y_pred)
    denom = y_true.sum()
    return float((y_pred.sum() - denom) / denom) if denom > 0 else float("nan")


def point_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, WAPE and bias in one dict (handy for building a results table)."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


# Saving results: one Excel workbook per notebook, written once the results are final
def save_workbook(sheets: dict[str, pd.DataFrame], name: str) -> Path:
    """Save DataFrames as the sheets of outputs/<name>.xlsx (one sheet per dict entry, in dict order) and return the path.

    Formatting is cosmetic only (bold header, frozen first row, column widths, number formats); the stored values
    are not rounded, so reading a sheet back gives the same numbers.
    """
    from datetime import datetime
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = OUTPUT_DIR / f"{name}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for j in range(1, ws.max_column + 1):
                column = ws[get_column_letter(j)]
                column[0].font = Font(bold=True)
                column[0].fill = PatternFill("solid", fgColor="DCE6F1")
                longest = len(str(column[0].value))
                for cell in column[1:]:
                    if isinstance(cell.value, datetime):
                        cell.number_format = "yyyy-mm-dd"
                        shown = 10
                    elif isinstance(cell.value, float):
                        small = cell.value != 0 and abs(cell.value) < 1      # ratios, per-day costs: show 4 decimals
                        cell.number_format = "0.0000" if small else "#,##0.00"
                        shown = len(f"{cell.value:,.4f}") if small else len(f"{cell.value:,.2f}")
                    else:
                        shown = len(str(cell.value))
                    longest = max(longest, shown)
                ws.column_dimensions[get_column_letter(j)].width = min(longest + 3, 60)
    return path


def load_sheet(book: str, sheet: str, date_cols: Sequence[str] = ("date",),
               required_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one sheet of outputs/<book>.xlsx; columns listed in date_cols are parsed as dates.

    required_cols: optional schema check. It raises ValueError naming the missing columns, so a notebook that
    reads a stale or hand-edited workbook fails with a clear message instead of a KeyError further down.
    """
    path = OUTPUT_DIR / f"{book}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the notebook that produces it first.")
    df = pd.read_excel(path, sheet_name=sheet)
    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}[{sheet}] is missing columns {missing}; re-run the notebook that produces it.")
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def save_figure(fig, name: str, dpi: int = 150, subdir: str | None = None, close: bool = False) -> Path:
    """Save a matplotlib figure as outputs/figures/[subdir/]name.png and return the path.

    close=True releases the figure afterwards (use it in loops that create many figures)."""
    folder = FIGURE_DIR / subdir if subdir else FIGURE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


# Plot style
def set_plot_style() -> None:
    """Apply one consistent plotting style across all notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    sns.set()
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})