# src/utils.py
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def robust_location_scale(x: np.ndarray) -> Tuple[float, float]:
    """
    Returns (median, robust_sigma) where robust_sigma ~= std using MAD.
    robust_sigma = 1.4826 * MAD
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    robust_sigma = 1.4826 * mad
    return float(med), float(robust_sigma)


def robust_threshold(x: np.ndarray, k: float) -> float:
    med, rs = robust_location_scale(x)
    if not np.isfinite(med) or not np.isfinite(rs):
        return np.nan
    return float(med + k * rs)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("#", "", regex=False)
    )
    return df


def pick_time_column(columns) -> str:
    candidates = ["time", "recorded_at", "timestamp", "datetime"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Cannot find a time column. Available columns: {list(columns)}")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return (s - mu) * 0.0
    return (s - mu) / sd


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def as_bool_series(s: pd.Series) -> pd.Series:
    # Robust conversion for "True/False", 0/1, etc.
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"])