# src/thresholds.py
from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from .utils import robust_threshold, save_csv


def load_preprocessed_period_csv(path: str) -> pd.DataFrame:
    """
    Load TRAIN/TEST preprocessed period CSV.

    Required columns:
      work_period, mean_value, peak_value, period_start_time, period_end_time, interval_id
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed CSV not found: {path}")

    df = pd.read_csv(path)
    required = ["work_period", "mean_value", "peak_value", "period_start_time", "period_end_time", "interval_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Got: {df.columns.tolist()}")

    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["peak_value"] = pd.to_numeric(df["peak_value"], errors="coerce")
    df["interval_id"] = pd.to_numeric(df["interval_id"], errors="coerce")

    df["period_start_time"] = pd.to_datetime(df["period_start_time"], errors="coerce", utc=True)
    df["period_end_time"] = pd.to_datetime(df["period_end_time"], errors="coerce", utc=True)

    df = df.dropna(subset=["work_period", "mean_value", "peak_value", "interval_id", "period_end_time"]).copy()
    df = df.sort_values(["interval_id", "work_period"]).reset_index(drop=True)
    return df


def fit_thresholds_on_train(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit alert thresholds using TRAIN only to avoid data leakage.

    Inputs (from config):
      - paths.preprocessed_train_csv
      - paths.theta_table_csv
      - paths.threshold_csv
      - alerts.k
      - alerts.slope_source  ('scratch' or 'sklearn')

    Output:
      - Writes a single-row CSV to threshold_csv
      - Returns thresholds as a dict
    """
    train_csv = config["paths"]["preprocessed_train_csv"]
    theta_csv = config["paths"]["theta_table_csv"]
    threshold_csv = config["paths"]["threshold_csv"]

    k = float(config["alerts"]["k"])
    slope_source = str(config["alerts"]["slope_source"]).lower().strip()

    train_df = load_preprocessed_period_csv(train_csv)
    theta_df = pd.read_csv(theta_csv)

    # Level thresholds (TRAIN distribution only)
    mean_alert = robust_threshold(train_df["mean_value"].to_numpy(), k)
    peak_alert = robust_threshold(train_df["peak_value"].to_numpy(), k)

    # Slope thresholds from TRAIN theta table
    if slope_source == "scratch":
        mean_slope_col = "scratch_mean_theta1"
        peak_slope_col = "scratch_peak_theta1"
    else:
        mean_slope_col = "sklearn_mean_theta1"
        peak_slope_col = "sklearn_peak_theta1"

    mean_slope_thr = robust_threshold(np.abs(theta_df[mean_slope_col].to_numpy()), k)
    peak_slope_thr = robust_threshold(np.abs(theta_df[peak_slope_col].to_numpy()), k)

    thresholds = {
        "mean_alert_threshold": float(mean_alert),
        "peak_alert_threshold": float(peak_alert),
        "mean_slope_threshold": float(mean_slope_thr),
        "peak_slope_threshold": float(peak_slope_thr),
        "slope_source": slope_source,
        "k": float(k),
        "mean_slope_col": mean_slope_col,
        "peak_slope_col": peak_slope_col,
        "train_csv": train_csv,
        "theta_csv": theta_csv,
    }

    save_csv(pd.DataFrame([thresholds]), threshold_csv)
    return thresholds


def load_thresholds(threshold_csv: str) -> Dict[str, Any]:
    """
    Load thresholds saved by fit_thresholds_on_train().
    """
    if not os.path.exists(threshold_csv):
        raise FileNotFoundError(f"threshold_csv not found: {threshold_csv}. Run fit_thresholds_on_train() first.")

    df = pd.read_csv(threshold_csv)
    if df.empty:
        raise ValueError(f"threshold_csv is empty: {threshold_csv}")

    return df.iloc[0].to_dict()
