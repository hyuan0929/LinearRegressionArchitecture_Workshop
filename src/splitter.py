# src/splitter.py
from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_csv


REQUIRED_COLS = ["work_period", "mean_value", "peak_value", "period_start_time", "period_end_time"]


def _add_interval_id(df: pd.DataFrame, interval_size: int) -> pd.DataFrame:
    df = df.copy()
    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df = df.dropna(subset=["work_period"]).copy()
    df = df.sort_values("work_period").reset_index(drop=True)
    df["interval_id"] = ((df["work_period"] - 1) // int(interval_size)) + 1
    return df


def _standardize_from_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Tuple[str, str] = ("mean_value", "peak_value"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize using TRAIN statistics only to avoid data leakage.
    Creates *_z columns in both train and test.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in cols:
        mu = float(pd.to_numeric(train_df[c], errors="coerce").mean())
        sd = float(pd.to_numeric(train_df[c], errors="coerce").std(ddof=0))
        if (not np.isfinite(sd)) or sd == 0.0:
            train_df[f"{c}_z"] = 0.0
            test_df[f"{c}_z"] = 0.0
        else:
            train_df[f"{c}_z"] = (pd.to_numeric(train_df[c], errors="coerce") - mu) / sd
            test_df[f"{c}_z"] = (pd.to_numeric(test_df[c], errors="coerce") - mu) / sd

    return train_df, test_df


def split_period_csv_to_train_test(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Read period_csv (work-period summary) -> split into train/test -> save as preprocessed_* csv.
    Train/Test outputs contain:
      work_period, mean_value, peak_value, period_start_time, period_end_time, interval_id
      + optional mean_value_z, peak_value_z (if training.use_standardized = true)
    """
    period_csv = config["paths"]["period_csv"]
    out_train = config["paths"]["preprocessed_train_csv"]
    out_test = config["paths"]["preprocessed_test_csv"]

    interval_size = int(config["interval"]["interval_size"])
    test_ratio = float(config.get("split", {}).get("test_ratio", 0.2))
    random_state = int(config.get("split", {}).get("random_state", 42))
    by_interval = bool(config.get("split", {}).get("by_interval", True))
    use_z = bool(config.get("training", {}).get("use_standardized", True))

    if not os.path.exists(period_csv):
        raise FileNotFoundError(f"period_csv not found: {period_csv}")

    df = pd.read_csv(period_csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in period_csv: {missing}. Got: {df.columns.tolist()}")

    # Basic cleaning / typing
    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["peak_value"] = pd.to_numeric(df["peak_value"], errors="coerce")
    df["period_start_time"] = pd.to_datetime(df["period_start_time"], errors="coerce", utc=True)
    df["period_end_time"] = pd.to_datetime(df["period_end_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["work_period", "mean_value", "peak_value", "period_end_time"]).copy()

    # Add interval_id
    df = _add_interval_id(df, interval_size=interval_size)

    if by_interval:
        # Split by interval_id to avoid leakage across same interval
        intervals = df["interval_id"].dropna().unique()
        rng = np.random.default_rng(random_state)
        rng.shuffle(intervals)
        n_test = max(1, int(np.ceil(len(intervals) * test_ratio)))
        test_intervals = set(intervals[:n_test])

        test_df = df[df["interval_id"].isin(test_intervals)].copy()
        train_df = df[~df["interval_id"].isin(test_intervals)].copy()
    else:
        # Simple row-wise split (not recommended for time/interval data)
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n_test = int(np.ceil(len(df) * test_ratio))
        test_df = df.iloc[:n_test].copy()
        train_df = df.iloc[n_test:].copy()

    # Standardize using train stats only
    if use_z:
        train_df, test_df = _standardize_from_train(train_df, test_df, cols=("mean_value", "peak_value"))

    # Save
    ensure_dir(os.path.dirname(out_train))
    ensure_dir(os.path.dirname(out_test))
    save_csv(train_df, out_train)
    save_csv(test_df, out_test)

    return {
        "preprocessed_train_csv": out_train,
        "preprocessed_test_csv": out_test,
    }
