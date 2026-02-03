# src/preprocessing.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .data_loader import prepare_time_sorted, get_axis_columns
from .utils import zscore, save_csv


def compute_overall_force(df: pd.DataFrame, axis_cols: list[str]) -> pd.Series:
    """
    Overall force per record (row).
    Default: mean(abs(axis_i)) across axes.
    """
    return df[axis_cols].abs().mean(axis=1)


def merge_short_false_gaps(mask: np.ndarray, gap_n: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and (not out[j]):
                j += 1
            left_true = (i - 1 >= 0 and out[i - 1])
            right_true = (j < n and out[j])
            if left_true and right_true and (j - i) <= gap_n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def remove_short_true_runs(mask: np.ndarray, min_n: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_n:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def label_true_runs(mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(mask), dtype=int)
    run_id = 0
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            run_id += 1
            j = i
            while j < n and mask[j]:
                j += 1
            labels[i:j] = run_id
            i = j
        else:
            i += 1
    return labels


def detect_work_periods(
    raw_df: pd.DataFrame,
    smooth_seconds: float,
    max_gap_multiplier: float,
    min_gap_seconds: float,
    min_work_seconds: float,
    thresh_quantile: float,
    thresh_scale: float,
) -> Tuple[pd.DataFrame, str]:
    """
    Detect work vs rest and label work_period on raw telemetry records.
    Returns df_with_work_period and time_col.
    """
    df, time_col = prepare_time_sorted(raw_df)
    axis_cols = get_axis_columns(df)

    # sampling interval estimate
    dt_sec = df[time_col].diff().dt.total_seconds()
    median_dt = float(dt_sec.dropna().median()) if dt_sec.dropna().size else 1.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0

    force = compute_overall_force(df, axis_cols)

    # smooth
    win = max(3, int(np.ceil(smooth_seconds / median_dt)))
    force_smooth = force.rolling(win, min_periods=1).mean()

    # adaptive threshold
    qv = float(np.nanpercentile(force_smooth, thresh_quantile * 100))
    thr = 0.05 if qv <= 0 else thresh_scale * qv

    is_work = (force_smooth > thr).to_numpy()

    # break across big gaps
    max_gap_seconds = max_gap_multiplier * median_dt
    big_gap = (dt_sec.fillna(0) > max_gap_seconds).to_numpy()
    is_work = is_work & (~big_gap)

    # merge short rest and remove short work
    gap_n = max(1, int(np.ceil(min_gap_seconds / median_dt)))
    min_n = max(2, int(np.ceil(min_work_seconds / median_dt)))

    is_work = merge_short_false_gaps(is_work, gap_n=gap_n)
    is_work = remove_short_true_runs(is_work, min_n=min_n)

    # label
    work_labels = label_true_runs(is_work)

    out = df.copy()
    out["overall_force"] = force
    out["work_period"] = work_labels
    return out, time_col


def summarize_work_periods(df_with_work: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Build period-level table:
    work_period, mean_value, peak_value, period_start_time, period_end_time
    """
    df_work = df_with_work[df_with_work["work_period"] > 0].copy()
    if df_work.empty:
        raise ValueError("No work periods detected. Try adjusting thresholds.")

    summary = (
        df_work.groupby("work_period")
        .agg(
            mean_value=("overall_force", "mean"),
            peak_value=("overall_force", "max"),
            period_start_time=(time_col, "min"),
            period_end_time=(time_col, "max"),
        )
        .reset_index()
        .sort_values("work_period")
        .reset_index(drop=True)
    )
    return summary


def build_preprocessed_period_table(
    period_df: pd.DataFrame,
    interval_size: int,
    add_zscore: bool = True,
) -> pd.DataFrame:
    """
    Create preprocessed table for modeling:
    - interval_id (every N work_periods)
    - mean_value_z / peak_value_z (optional)
    """
    df = period_df.copy()
    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["peak_value"] = pd.to_numeric(df["peak_value"], errors="coerce")

    df = df.dropna(subset=["work_period", "mean_value", "peak_value"]).copy()
    df = df.sort_values("work_period").reset_index(drop=True)

    df["interval_id"] = ((df["work_period"] - 1) // interval_size) + 1

    if add_zscore:
        df["mean_value_z"] = zscore(df["mean_value"])
        df["peak_value_z"] = zscore(df["peak_value"])

    return df


def run_preprocessing_pipeline(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end preprocessing:
    raw -> work_period detection -> period summary -> preprocessed table
    Saves output CSVs based on config paths.
    Returns (period_summary_df, preprocessed_df)
    """
    raw_csv = config["paths"]["raw_csv"]
    period_csv = config["paths"]["period_csv"]
    preprocessed_csv = config["paths"]["preprocessed_csv"]

    wp = config["work_period_detection"]
    interval_size = int(config["interval"]["interval_size"])
    add_z = bool(config["training"]["use_standardized"])

    import pandas as pd
    from .data_loader import load_raw_csv

    raw_df = load_raw_csv(raw_csv)

    df_with_work, time_col = detect_work_periods(
        raw_df=raw_df,
        smooth_seconds=float(wp["smooth_seconds"]),
        max_gap_multiplier=float(wp["max_gap_multiplier"]),
        min_gap_seconds=float(wp["min_gap_seconds"]),
        min_work_seconds=float(wp["min_work_seconds"]),
        thresh_quantile=float(wp["thresh_quantile"]),
        thresh_scale=float(wp["thresh_scale"]),
    )

    period_summary = summarize_work_periods(df_with_work, time_col)
    save_csv(period_summary, period_csv)

    preprocessed = build_preprocessed_period_table(period_summary, interval_size=interval_size, add_zscore=add_z)
    save_csv(preprocessed, preprocessed_csv)

    return period_summary, preprocessed