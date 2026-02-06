"""
EDA_RobotDashboard.py

Streams robot telemetry data from Neon PostgreSQL and shows a live plot.
NEW: After each work period ends, compute mean/peak of overall_force.
When there are 10 work periods available (current + previous 9),
run regression on the last 10 mean series and peak series.
If slope > thresholds in data/models/alert_thresholds.csv -> popup alert.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Popup (Windows desktop friendly)
import tkinter as tk
from tkinter import messagebox


# -----------------------------
# Database configuration
# -----------------------------
# IMPORTANT: Use sslmode=require (already included). Add connect_timeout to avoid hanging forever.
CONN_STR = (
    "postgresql://neondb_owner:npg_Sh8bV3HjZvkd@ep-plain-scene-ahmzh8by-pooler.c-3.us-east-1.aws.neon.tech/"
    "neondb?sslmode=require"
)

# Default table for streaming
DEFAULT_SOURCE_TABLE = "robot_data"

# Output CSV used to init DB once (if table empty)
DEFAULT_RAW_CSV = "data/raw/RMBR4-2_export_test.csv"

# Alert thresholds file
ALERT_THRESHOLDS_CSV = "data/models/alert_thresholds.csv"


def _make_engine(conn_str: str):
    """
    Create SQLAlchemy engine with a connection timeout
    so the program won't hang forever if the network blocks DB connections.
    """
    return create_engine(
        conn_str,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},  # seconds
    )


ENGINE = _make_engine(CONN_STR)


# -----------------------------
# Helper: Popup alert
# -----------------------------
def popup_alert(title: str, msg: str) -> None:
    """Show a warning popup dialog."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, msg)
    root.destroy()


# -----------------------------
# Helper: thresholds
# -----------------------------
def load_alert_thresholds(csv_path: str = ALERT_THRESHOLDS_CSV) -> dict:
    """
    Read thresholds from CSV.

    Supported formats:
    A) One row with:
       mean_slope_threshold, peak_slope_threshold
    B) One row with:
       slope_threshold   (used for both mean and peak)
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Alert thresholds file not found: {csv_path}")

    df = pd.read_csv(p)

    if {"mean_slope_threshold", "peak_slope_threshold"}.issubset(df.columns):
        row = df.iloc[0].to_dict()
        return {
            "mean_slope_threshold": float(row["mean_slope_threshold"]),
            "peak_slope_threshold": float(row["peak_slope_threshold"]),
        }

    if "slope_threshold" in df.columns:
        thr = float(df.iloc[0]["slope_threshold"])
        return {
            "mean_slope_threshold": thr,
            "peak_slope_threshold": thr,
        }

    raise ValueError(
        "alert_thresholds.csv must contain either "
        "['mean_slope_threshold','peak_slope_threshold'] or ['slope_threshold']"
    )


# -----------------------------
# Helper: regression slope
# -----------------------------
def regression_slope(y: List[float]) -> float:
    """
    Simple linear regression slope (theta1) for y against x = 1..N.
    """
    y_arr = np.asarray(y, dtype=float)
    x = np.arange(1, len(y_arr) + 1, dtype=float)

    x_mean = x.mean()
    y_mean = y_arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0

    slope = (((x - x_mean) * (y_arr - y_mean)).sum()) / denom
    return float(slope)


# -----------------------------
# DB utilities
# -----------------------------
def is_table_empty(table_name: str, engine=ENGINE) -> bool:
    """
    Return True if table does not exist or has zero rows.
    Uses fast query: SELECT 1 ... LIMIT 1
    """
    try:
        if not inspect(engine).has_table(table_name):
            return True
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1")).first() is None
    except SQLAlchemyError:
        # safest default
        return True


def _preprocess_robot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns and parse recorded_at.
    Keeps axis_1..axis_8 typically, drops extra axis columns if present.
    """
    df = df.copy()
    df.columns = [col.lower().strip().replace(" ", "_").replace("#", "") for col in df.columns]

    # Rename time column to recorded_at (common in some exports)
    if "time" in df.columns and "recorded_at" not in df.columns:
        df.rename(columns={"time": "recorded_at"}, inplace=True)

    # Drop unused axis columns (keep axis_1 ~ axis_8)
    cols_to_drop = ["axis_9", "axis_10", "axis_11", "axis_12", "axis_13", "axis_14"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")

    return df


def init_data_once(
    source_table: str = DEFAULT_SOURCE_TABLE,
    csv_path: str = DEFAULT_RAW_CSV,
) -> None:
    """
    Initialize DB table only once:
      - If table missing or empty -> load CSV -> preprocess -> write to DB
      - Otherwise do nothing
    """
    if not is_table_empty(source_table):
        return

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _preprocess_robot_df(df)

    if "recorded_at" not in df.columns:
        raise ValueError("CSV must contain a time column 'recorded_at' (or 'time').")

    # Replace ONLY because table is empty/missing
    df.to_sql(source_table, ENGINE, if_exists="replace", index=False)
    print(f"✅ Initialized '{source_table}' with {len(df)} rows from {csv_path!r}.")


# -----------------------------
# Online work period detection
# -----------------------------
class OnlineWorkPeriodDetector:
    """
    Online work period detection using overall_force >= work_threshold.

    - When entering work: start collecting points
    - When leaving work: finalize if enough points -> return summary dict
    """

    def __init__(self, work_threshold: float, min_points: int = 10):
        self.work_threshold = float(work_threshold)
        self.min_points = int(min_points)

        self.in_work = False
        self.forces: List[float] = []
        self.start_time: Optional[pd.Timestamp] = None
        self.end_time: Optional[pd.Timestamp] = None

    def update(self, ts: pd.Timestamp, overall_force: float) -> Optional[dict]:
        is_work = overall_force >= self.work_threshold

        # Start a work period
        if (not self.in_work) and is_work:
            self.in_work = True
            self.forces = [overall_force]
            self.start_time = ts
            self.end_time = ts
            return None

        # Continue a work period
        if self.in_work and is_work:
            self.forces.append(overall_force)
            self.end_time = ts
            return None

        # End a work period
        if self.in_work and (not is_work):
            self.in_work = False

            result = None
            if len(self.forces) >= self.min_points:
                result = {
                    "mean_value": float(np.mean(self.forces)),
                    "peak_value": float(np.max(self.forces)),
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "n_points": len(self.forces),
                }

            # reset
            self.forces = []
            self.start_time = None
            self.end_time = None
            return result

        return None


# -----------------------------
# Streaming simulator
# -----------------------------
@dataclass
class StreamConfig:
    source_table: str = DEFAULT_SOURCE_TABLE
    delay: float = 0.1
    max_xticks: int = 50
    window_len: int = 200

    # Work period detection parameters (tune these)
    work_threshold: float = 0.2
    min_work_points: int = 10

    # Alert cooldown (seconds): prevent too many popups
    alert_cooldown_sec: float = 10.0


class StreamingSimulator:
    """
    Stream robot telemetry data from online DB table.

    NEW:
    - Detect work periods online based on overall_force
    - When a work period ends: compute mean/peak, keep last 10,
      run regression, compare slope to thresholds, popup alert if exceeded.
    """

    def __init__(self, db_conn_str: str, config: StreamConfig):
        self.config = config
        self.engine = _make_engine(db_conn_str)

        self.current_offset = 0
        self._buffer_row: Optional[pd.DataFrame] = None

        # Load alert thresholds
        self.alert_thr = load_alert_thresholds(ALERT_THRESHOLDS_CSV)
        self._last_alert_time = 0.0

        # Count total rows once
        cnt = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {self.config.source_table};",
            self.engine
        )
        self.total_rows = int(cnt.iloc[0]["n"])
        if self.total_rows <= 0:
            raise RuntimeError(f"Source table '{self.config.source_table}' is empty.")

        # Fetch one row to detect schema/axis columns
        self._buffer_row = pd.read_sql_query(
            f"SELECT * FROM {self.config.source_table} ORDER BY recorded_at LIMIT 1 OFFSET 0;",
            self.engine
        )
        self._buffer_row = _preprocess_robot_df(self._buffer_row)

        # Detect axis columns (axis_1..axis_8)
        self.axis_cols: List[str] = [
            col for col in self._buffer_row.columns
            if col.startswith("axis_")
            and col.split("_")[1].isdigit()
            and 1 <= int(col.split("_")[1]) <= 8
        ]
        self.axis_cols = sorted(self.axis_cols, key=lambda x: int(x.split("_")[1]))

        if "recorded_at" not in self._buffer_row.columns:
            raise ValueError("Source table must contain a 'recorded_at' column.")

        if not self.axis_cols:
            raise ValueError("No axis columns found (expected axis_1..axis_8).")

        # Work period detector & buffers for last 10 periods
        self.wp_detector = OnlineWorkPeriodDetector(
            work_threshold=self.config.work_threshold,
            min_points=self.config.min_work_points
        )
        self.last_mean: List[float] = []
        self.last_peak: List[float] = []
        self.work_period_count = 0

        # Plot init
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.x_data: List[pd.Timestamp] = []
        self.y_data_dict: Dict[str, List[float]] = {col: [] for col in self.axis_cols}

        print(f"✅ Loaded {self.total_rows} rows from DB table '{self.config.source_table}'.")
        print(f"✅ Detected Y-axis columns: {self.axis_cols}")
        print(f"✅ Thresholds: {self.alert_thr}")

    def _get_next_row(self) -> pd.DataFrame:
        """Fetch exactly one record from source_table, ordered by recorded_at."""
        if self._buffer_row is not None:
            row = self._buffer_row
            self._buffer_row = None
            return row

        row = pd.read_sql_query(
            f"SELECT * FROM {self.config.source_table} ORDER BY recorded_at LIMIT 1 OFFSET {self.current_offset};",
            self.engine
        )
        if row.empty:
            return row
        return _preprocess_robot_df(row)

    def _maybe_alert(self, mean_slope: float, peak_slope: float, finished: dict) -> None:
        """
        Compare slopes to thresholds and popup alert if exceeded.
        Includes cooldown to reduce spam.
        """
        mean_thr = float(self.alert_thr["mean_slope_threshold"])
        peak_thr = float(self.alert_thr["peak_slope_threshold"])

        fault_types = []
        if mean_slope > mean_thr:
            fault_types.append("Equipment Aging")
        if peak_slope > peak_thr:
            fault_types.append("Equipment Fault")

        if not fault_types:
            return

        now = time.time()
        if (now - self._last_alert_time) < float(self.config.alert_cooldown_sec):
            return
        self._last_alert_time = now

        # Predicted failure time: 2 weeks after abnormal interval end
        end_time = finished["end_time"]
        predicted_failure_time = None
        try:
            predicted_failure_time = pd.to_datetime(end_time) + pd.Timedelta(days=14)
        except Exception:
            predicted_failure_time = "N/A"

        msg = (
            f"Abnormal trend detected!\n\n"
            f"Abnormal interval (work period):\n"
            f"  Start: {finished['start_time']}\n"
            f"  End:   {finished['end_time']}\n"
            f"  Points:{finished['n_points']}\n\n"
            f"Prediction:\n"
            f"  Possible failure time: {predicted_failure_time}\n"
            f"  Type: {', '.join(fault_types)}\n\n"
            f"Slopes (last 10 work periods):\n"
            f"  Mean slope: {mean_slope:.6f}  (thr={mean_thr:.6f})\n"
            f"  Peak slope: {peak_slope:.6f}  (thr={peak_thr:.6f})\n"
        )
        popup_alert("Robot Alert", msg)

    def nextDataPoint(self):
        if self.current_offset >= self.total_rows:
            print("All data points have been streamed.")
            return None

        row = self._get_next_row()
        if row.empty:
            print("No more rows.")
            return None

        # Timestamp
        ts = pd.to_datetime(row["recorded_at"].values[0])

        # Overall force: mean(abs(axis_i)) across axes
        overall_force = float(row[self.axis_cols].abs().mean(axis=1).iloc[0])

        # Update work-period detector
        finished = self.wp_detector.update(ts, overall_force)
        if finished is not None:
            self.work_period_count += 1
            self.last_mean.append(float(finished["mean_value"]))
            self.last_peak.append(float(finished["peak_value"]))

            # Keep last 10
            self.last_mean = self.last_mean[-10:]
            self.last_peak = self.last_peak[-10:]

            # Run regression when we have 10 work periods
            if len(self.last_mean) == 10 and len(self.last_peak) == 10:
                mean_slope = regression_slope(self.last_mean)
                peak_slope = regression_slope(self.last_peak)

                # Trigger popup if exceeded thresholds
                self._maybe_alert(mean_slope, peak_slope, finished)

        # Update plot buffers
        self.x_data.append(ts)
        self.x_data = self.x_data[-self.config.window_len:]

        for col in self.axis_cols:
            self.y_data_dict[col].append(float(row[col].values[0]))
            self.y_data_dict[col] = self.y_data_dict[col][-self.config.window_len:]

        # Redraw plot
        self.ax.clear()
        for col in self.axis_cols:
            self.ax.plot(self.x_data, self.y_data_dict[col], label=col, linewidth=1)

        self.ax.set_title(f"Streaming Robot Axis Data ({self.current_offset + 1}/{self.total_rows})")
        self.ax.set_xlabel("recorded_at")
        self.ax.set_ylabel("Axis Values")

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        if len(self.x_data) > self.config.max_xticks:
            step = max(1, len(self.x_data) // self.config.max_xticks)
            self.ax.set_xticks(self.x_data[::step])

        self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

        # advance offset and sleep
        self.current_offset += 1
        time.sleep(self.config.delay)

        return row


def main():
    # Ensure running from project root
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")

    # 1) Initialize DB table once (only if empty/missing)
    init_data_once(source_table=DEFAULT_SOURCE_TABLE, csv_path=DEFAULT_RAW_CSV)

    # 2) Stream from online DB
    ss = StreamingSimulator(
        db_conn_str=CONN_STR,
        config=StreamConfig(
            source_table=DEFAULT_SOURCE_TABLE,
            delay=0.1,
            work_threshold=0.2,      # tune based on your data
            min_work_points=10,      # tune based on your sampling rate
            alert_cooldown_sec=10.0, # avoid popup spam
        ),
    )

    while True:
        result = ss.nextDataPoint()
        if result is None:
            break


if __name__ == "__main__":
    main()
