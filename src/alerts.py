# src/alerts.py
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from .utils import robust_threshold, save_csv, ensure_dir


def build_interval_summary_from_period(period_csv: str, interval_size: int) -> pd.DataFrame:
    """
    period_csv must contain:
    work_period, mean_value, peak_value, period_start_time, period_end_time
    """
    df = pd.read_csv(period_csv)

    req = ["work_period", "mean_value", "peak_value", "period_start_time", "period_end_time"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in period csv: {missing}")

    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["peak_value"] = pd.to_numeric(df["peak_value"], errors="coerce")
    df["period_start_time"] = pd.to_datetime(df["period_start_time"], errors="coerce", utc=True)
    df["period_end_time"] = pd.to_datetime(df["period_end_time"], errors="coerce", utc=True)

    df = df.dropna(subset=["work_period", "mean_value", "peak_value", "period_end_time"]).copy()
    df = df.sort_values("work_period").reset_index(drop=True)

    df["interval_id"] = ((df["work_period"] - 1) // interval_size) + 1

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
    return interval_summary, df  # return df also for global thresholds


def generate_alert_thresholds(period_level_df: pd.DataFrame, theta_df: pd.DataFrame, k: float, slope_source: str):
    """
    - mean/peak level thresholds from period-level raw distributions
    - slope thresholds from theta table |theta1|
    """
    mean_alert = robust_threshold(period_level_df["mean_value"].to_numpy(), k)
    peak_alert = robust_threshold(period_level_df["peak_value"].to_numpy(), k)

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
    Build experiments/results.csv with:
    interval start/end time, predicted_failure_time (end+14 days), failure_type.
    Uses:
    - period_csv for time range & levels
    - theta_table_csv for slopes
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

    # merge slopes
    merged = interval_summary.merge(theta_df[["interval_id", mean_slope_col, peak_slope_col]], on="interval_id", how="left")

    # flags
    merged["mean_slope_flag"] = merged[mean_slope_col].abs() > thresholds["mean_slope_threshold"]
    merged["peak_slope_flag"] = merged[peak_slope_col].abs() > thresholds["peak_slope_threshold"]

    merged["aging_alert"] = merged["mean_slope_flag"] & (merged["interval_mean_level"] > thresholds["mean_alert_threshold"])
    merged["fault_alert"] = merged["peak_slope_flag"] & (merged["interval_peak_level"] > thresholds["peak_alert_threshold"])

    def failure_type(row):
        if row["aging_alert"] and row["fault_alert"]:
            return "Aging + Fault"
        if row["fault_alert"]:
            return "Fault"
        if row["aging_alert"]:
            return "Aging"
        return "No Alert"

    merged["failure_type"] = merged.apply(failure_type, axis=1)
    merged["predicted_failure_time"] = merged["interval_end_time"] + pd.Timedelta(days=predict_days)

    def build_reason(r):
        reasons = []
        if r["aging_alert"]:
            reasons.append("Possible system aging: mean slope & mean level exceed thresholds.")
        if r["fault_alert"]:
            reasons.append("Possible failure: peak slope & peak level exceed thresholds.")
        return " | ".join(reasons) if reasons else "No alert."

    merged["alert_reason"] = merged.apply(build_reason, axis=1)

    # save thresholds
    thr_df = pd.DataFrame([{
        **thresholds,
        "period_csv": period_csv,
        "theta_csv": theta_csv,
    }])
    save_csv(thr_df, threshold_csv)

    # save results
    ensure_dir(os.path.dirname(results_csv))
    results_df = merged[[
        "interval_id",
        "interval_start_time",
        "interval_end_time",
        "predicted_failure_time",
        "failure_type",
        "alert_reason",
    ]].sort_values("interval_id").reset_index(drop=True)

    results_df.to_csv(results_csv, index=False)
    return results_df