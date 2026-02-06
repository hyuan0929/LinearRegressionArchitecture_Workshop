# src/stream_alert_thresholds.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_stream_alert_thresholds(csv_path: str) -> dict:
    """
    Read thresholds for streaming popup alert.

    Supported formats:
      A) One row with columns:
         mean_slope_threshold, peak_slope_threshold
      B) One row with column:
         slope_threshold (shared for both)
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Stream alert thresholds file not found: {csv_path}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Stream alert thresholds file is empty: {csv_path}")

    if {"mean_slope_threshold", "peak_slope_threshold"}.issubset(df.columns):
        row = df.iloc[0]
        return {
            "mean_slope_threshold": float(row["mean_slope_threshold"]),
            "peak_slope_threshold": float(row["peak_slope_threshold"]),
        }

    if "slope_threshold" in df.columns:
        thr = float(df.iloc[0]["slope_threshold"])
        return {"mean_slope_threshold": thr, "peak_slope_threshold": thr}

    raise ValueError(
        "Expected columns: (mean_slope_threshold, peak_slope_threshold) OR (slope_threshold)."
    )
