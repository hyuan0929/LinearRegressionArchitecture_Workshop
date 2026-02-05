# src/alerts.py
from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd

from .utils import ensure_dir
from .thresholds import load_preprocessed_period_csv, load_thresholds


def build_interval_summary_from_period_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build interval-level summary from a period-level dataframe (TEST side).

    Expected columns:
      work_period, mean_value, peak_value, period_start_time, period_end_time, interval_id
    """
    summary = (
        df.groupby("interval_id")
        .agg(
            interval_start_time=("period_start_time", "min"),
            interval_end_time=("period_end_time", "max"),
            interval_mean_level=("mean_value", "mean"),
            interval_peak_level=("peak_value", "max"),
            n_periods=("work_period", "count"),
            start_work_period=("work_period", "min"),
            end_work_period=("work_period", "max"),
        )
        .reset_index()
        .sort_values("interval_id")
        .reset_index(drop=True)
    )
    return summary


def detect_alerts_on_test(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Testing step (TEST only):
      - Reads TEST preprocessed CSV
      - Reads TRAIN theta table (slopes per interval)
      - Reads thresholds CSV (fit from TRAIN)
      - Writes results CSV:
          * ONLY anomaly intervals (failure_type != 'No Alert')
          * If none: writes an empty CSV (headers only) and prints message
    """
    test_csv = config["paths"]["preprocessed_test_csv"]
    theta_csv = config["paths"]["theta_table_csv"]
    threshold_csv = config["paths"]["threshold_csv"]
    results_csv = config["paths"]["results_csv"]

    predict_days = int(config["alerts"]["predict_offset_days"])

    # Load TEST period-level data and summarize to interval level
    test_df = load_preprocessed_period_csv(test_csv)
    interval_summary = build_interval_summary_from_period_df(test_df)

    # Load TRAIN theta table and TRAIN thresholds
    theta_df = pd.read_csv(theta_csv)
    thr = load_thresholds(threshold_csv)

    mean_slope_col = str(thr["mean_slope_col"])
    peak_slope_col = str(thr["peak_slope_col"])

    # Merge slopes onto TEST intervals (some intervals may be missing in theta table)
    merged = interval_summary.merge(
        theta_df[["interval_id", mean_slope_col, peak_slope_col]],
        on="interval_id",
        how="left",
    )

    # Trend flags
    merged["mean_slope_flag"] = merged[mean_slope_col].abs() > float(thr["mean_slope_threshold"])
    merged["peak_slope_flag"] = merged[peak_slope_col].abs() > float(thr["peak_slope_threshold"])

    # Level + trend logic
    merged["aging_alert"] = merged["mean_slope_flag"] & (merged["interval_mean_level"] > float(thr["mean_alert_threshold"]))
    merged["fault_alert"] = merged["peak_slope_flag"] & (merged["interval_peak_level"] > float(thr["peak_alert_threshold"]))

    def failure_type(r) -> str:
        if bool(r["aging_alert"]) and bool(r["fault_alert"]):
            return "Aging + Fault"
        if bool(r["fault_alert"]):
            return "Fault"
        if bool(r["aging_alert"]):
            return "Aging"
        return "No Alert"

    merged["failure_type"] = merged.apply(failure_type, axis=1)
    merged["predicted_failure_time"] = merged["interval_end_time"] + pd.Timedelta(days=predict_days)

    def reason(r) -> str:
        reasons = []
        if bool(r["aging_alert"]):
            reasons.append("Possible aging: mean slope & mean level exceed TRAIN thresholds.")
        if bool(r["fault_alert"]):
            reasons.append("Possible fault: peak slope & peak level exceed TRAIN thresholds.")
        return " | ".join(reasons) if reasons else "No alert."

    merged["alert_reason"] = merged.apply(reason, axis=1)

    out_cols = [
        "interval_id",
        "start_work_period",
        "end_work_period",
        "interval_start_time",
        "interval_end_time",
        "predicted_failure_time",
        "failure_type",
        "alert_reason",
    ]
    results_df = merged[out_cols].sort_values("interval_id").reset_index(drop=True)

    # Keep only anomalies
    anomaly_df = results_df[results_df["failure_type"] != "No Alert"].copy()

    # Always write a CSV file
    ensure_dir(os.path.dirname(results_csv))
    if anomaly_df.empty:
        pd.DataFrame(columns=out_cols).to_csv(results_csv, index=False)
        print(f"No anomalies detected. Wrote empty results file: {results_csv}")
        return anomaly_df

    anomaly_df.to_csv(results_csv, index=False)
    print(f"Anomalies detected: {len(anomaly_df)}. Saved: {results_csv}")
    return anomaly_df
