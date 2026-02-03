# src/data_loader.py
import os
import pandas as pd
from typing import Tuple, List

from .utils import standardize_columns, pick_time_column, safe_to_datetime


def load_raw_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    df = pd.read_csv(path)
    df = standardize_columns(df)
    return df


def prepare_time_sorted(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Identify time column, parse it, sort by time, and drop invalid timestamps.
    Returns (sorted_df, time_col).
    """
    time_col = pick_time_column(df.columns)
    df = df.copy()
    df[time_col] = safe_to_datetime(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df, time_col


def get_axis_columns(df: pd.DataFrame) -> List[str]:
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    if not axis_cols:
        raise ValueError(f"Cannot find axis columns. Columns: {df.columns.tolist()}")

    # Drop axes entirely NaN
    axis_cols = [c for c in axis_cols if df[c].notna().any()]

    # Sort by axis number if possible
    def axis_key(name: str) -> int:
        try:
            return int(name.split("_")[1])
        except Exception:
            return 10**9

    axis_cols = sorted(axis_cols, key=axis_key)
    return axis_cols