# src/utils.py -- Helper functions and configs
from __future__ import annotations
 
from pathlib import Path
from typing import Iterator, Sequence
 
import numpy as np
import pandas as pd


# Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "inventory_demand_forecasting.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
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


def pinball_loss(y_true, y_pred, q: float) -> float:
    """Pinball (quantile) loss for quantile level q in (0, 1). Lower is better."""
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1.")
    y_true, y_pred = _pair(y_true, y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
 
 
def interval_coverage(y_true, lower, upper) -> float:
    """Share of actual values that fall inside [lower, upper]."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


# Time-series cross-validation
def rolling_origin_splits(df: pd.DataFrame, n_folds: int = 4, horizon: int = 60,
                          min_train_days: int = 365,
                          date_col: str = "date") -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold, train_idx, test_idx) for rolling-origin (expanding window) CV.
 
    - Splits are made by DATE, so every product shares the same cut-off dates.
    - The test windows are the last n_folds * horizon days, contiguous and
      non-overlapping; each fold trains on everything before its test window.
    - train_idx / test_idx are index labels of df (df needs a unique index,
      which load_data guarantees).
    - min_train_days guards against a first fold that has not seen a full
      annual cycle (important when the data has strong yearly seasonality).
    """
    dates = np.sort(df[date_col].unique())
    first_test_start = len(dates) - n_folds * horizon
    if first_test_start < min_train_days:
        raise ValueError(
            f"Only {first_test_start} training days before the first test window "
            f"(min_train_days={min_train_days}). Reduce n_folds or horizon."
        )
 
    for k in range(n_folds):
        test_start = first_test_start + k * horizon
        train_dates = dates[:test_start]
        test_dates = dates[test_start:test_start + horizon]
        train_idx = df.index[df[date_col].isin(train_dates)].to_numpy()
        test_idx = df.index[df[date_col].isin(test_dates)].to_numpy()
        yield k + 1, train_idx, test_idx


# Saving results
def _csv_path(name: str, subdir: str | None = None) -> Path:
    """Path of outputs/csv/[subdir/]name.csv (pure function: creates nothing)."""
    folder = CSV_DIR / subdir if subdir else CSV_DIR
    return folder / f"{name}.csv"


def save_output(df: pd.DataFrame, name: str, subdir: str | None = None) -> Path:
    """Save a DataFrame as outputs/csv/[subdir/]name.csv and return the path."""
    path = _csv_path(name, subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_output(name: str, subdir: str | None = None,
                date_cols: Sequence[str] = ("date",),
                required_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Load outputs/csv/[subdir/]name.csv; columns listed in date_cols are parsed as dates.

    required_cols: optional schema check. It raises ValueError naming the missing columns, so a notebook that
    reads a stale or hand-edited hand-off file fails with a clear message instead of a KeyError further down.
    Reading never creates directories.
    """
    path = _csv_path(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the notebook that produces it first.")
    df = pd.read_csv(path)
    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns {missing}; re-run the notebook that produces it.")
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