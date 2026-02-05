# src/alerts.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .utils import robust_threshold, save_csv, ensure_dir


def build_interval_summary_from_period(period_csv: str, interval_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build interval-level summary from a period-level CSV.

    period_csv must contain:
      - work_period
      - mean_value
      - peak_value
      - period_start_time
      - period_end_time

    Returns:
      interval_summary_df, period_level_df
    """
    if not os.path.exists(period_csv):
        raise FileNotFoundError(f"Period CSV not found: {period_csv}")

    df = pd.read_csv(period_csv)

    required = ["work_period", "mean_value", "peak_value", "period_start_time", "period_end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in period csv: {missing}. Got: {df.columns.tolist()}")

    # Ensure types
    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["peak_value"] = pd.to_numeric(df["peak_value"], errors="coerce")
    df["period_start_time"] = pd.to_datetime(df["period_start_time"], errors="coerce", utc=True)
    df["period_end_time"] = pd.to_datetime(df["period_end_time"], errors="coerce", utc=True)

    df = df.dropna(subset=["work_period", "mean_value", "peak_value", "period_end_time"]).copy()
    df = df.sort_values("work_period").reset_index(drop=True)

    # Interval id (every N work periods)
    df["interval_id"] = ((df["work_period"] - 1) // int(interval_size)) + 1

    # Interval summary: start/end + mean level + peak level
    interval_summary = (
        df.groupby("interval_id")
        .agg(
            interval_start_time=("period_start_time", "min"),
            interval_end_time=("period_end_time", "max"),
            interval_mean_level=("mean_value", "mean"),
            interval_peak_level=("peak_value", "max"),
            n_periods=("work_period", "count"),
        )
        .reset_index()
        .sort_values("interval_id")
        .reset_index(drop=True)
    )

    return interval_summary, df


def generate_alert_thresholds(
    period_level_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    k: float,
    slope_source: str,
) -> Dict[str, Any]:
    """
    Compute robust thresholds:
      - mean/peak level thresholds from period-level distribution
      - slope thresholds from interval theta table distribution (abs(theta1))
    """
    mean_alert = robust_threshold(period_level_df["mean_value"].to_numpy(), k)
    peak_alert = robust_threshold(period_level_df["peak_value"].to_numpy(), k)

    slope_source = slope_source.lower().strip()
    if slope_source == "scratch":
        mean_slope_col = "scratch_mean_theta1"
        peak_slope_col = "scratch_peak_theta1"
    else:
        mean_slope_col = "sklearn_mean_theta1"
        peak_slope_col = "sklearn_peak_theta1"

    mean_slope_thr = robust_threshold(np.abs(theta_df[mean_slope_col].to_numpy()), k)
    peak_slope_thr = robust_threshold(np.abs(theta_df[peak_slope_col].to_numpy()), k)

    return {
        "mean_alert_threshold": float(mean_alert),
        "peak_alert_threshold": float(peak_alert),
        "mean_slope_threshold": float(mean_slope_thr),
        "peak_slope_threshold": float(peak_slope_thr),
        "slope_source": slope_source,
        "k": float(k),
        "mean_slope_col": mean_slope_col,
        "peak_slope_col": peak_slope_col,
    }


def build_results_csv(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Alerts/Testing step:
      - Reads TEST period CSV (you should point config["paths"]["period_csv"] to TEST here)
      - Uses trained theta table from TRAIN
      - Outputs only anomaly intervals to results_csv
      - If no anomalies: writes an empty CSV (headers only) and prints message
    """
    period_csv = config["paths"]["period_csv"]
    theta_csv = config["paths"]["theta_table_csv"]
    results_csv = config["paths"]["results_csv"]
    threshold_csv = config["paths"]["threshold_csv"]

    interval_size = int(config["interval"]["interval_size"])
    k = float(config["alerts"]["k"])
    predict_days = int(config["alerts"]["predict_offset_days"])
    slope_source = str(config["alerts"]["slope_source"]).lower()

    interval_summary, period_level_df = build_interval_summary_from_period(period_csv, interval_size=interval_size)
    theta_df = pd.read_csv(theta_csv)

    thresholds = generate_alert_thresholds(period_level_df, theta_df, k=k, slope_source=slope_source)
    mean_slope_col = thresholds["mean_slope_col"]
    peak_slope_col = thresholds["peak_slope_col"]

    # Merge slopes into interval table
    merged = interval_summary.merge(
        theta_df[["interval_id", mean_slope_col, peak_slope_col]],
        on="interval_id",
        how="left",
    )

    # Flags based on slope thresholds
    merged["mean_slope_flag"] = merged[mean_slope_col].abs() > thresholds["mean_slope_threshold"]
    merged["peak_slope_flag"] = merged[peak_slope_col].abs() > thresholds["peak_slope_threshold"]

    # Level thresholds
    merged["aging_alert"] = merged["mean_slope_flag"] & (merged["interval_mean_level"] > thresholds["mean_alert_threshold"])
    merged["fault_alert"] = merged["peak_slope_flag"] & (merged["interval_peak_level"] > thresholds["peak_alert_threshold"])

    # Failure type label
    def failure_type(row) -> str:
        if row["aging_alert"] and row["fault_alert"]:
            return "Aging + Fault"
        if row["fault_alert"]:
            return "Fault"
        if row["aging_alert"]:
            return "Aging"
        return "No Alert"

    merged["failure_type"] = merged.apply(failure_type, axis=1)
    merged["predicted_failure_time"] = merged["interval_end_time"] + pd.Timedelta(days=predict_days)

    # Reason
    def build_reason(r) -> str:
        reasons = []
        if r["aging_alert"]:
            reasons.append("Possible aging: mean slope & mean level exceed thresholds.")
        if r["fault_alert"]:
            reasons.append("Possible fault: peak slope & peak level exceed thresholds.")
        return " | ".join(reasons) if reasons else "No alert."

    merged["alert_reason"] = merged.apply(build_reason, axis=1)

    # Save thresholds (audit)
    thr_df = pd.DataFrame([{
        **thresholds,
        "period_csv": period_csv,
        "theta_csv": theta_csv,
    }])
    save_csv(thr_df, threshold_csv)

    out_cols = [
        "interval_id",
        "interval_start_time",
        "interval_end_time",
        "predicted_failure_time",
        "failure_type",
        "alert_reason",
    ]
    results_df = merged[out_cols].sort_values("interval_id").reset_index(drop=True)

    # Keep ONLY anomaly rows
    anomaly_df = results_df[results_df["failure_type"] != "No Alert"].copy()

    # Always write a CSV:
    ensure_dir(os.path.dirname(results_csv))
    if anomaly_df.empty:
        empty = pd.DataFrame(columns=out_cols)
        empty.to_csv(results_csv, index=False)
        print("No anomalies detected. Saved an empty results file:", results_csv)
        return empty

    anomaly_df.to_csv(results_csv, index=False)
    print(f"Anomalies detected: {len(anomaly_df)}. Saved:", results_csv)
    return anomaly_df
