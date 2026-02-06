# src/stream_streaming.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .stream_db import make_stream_engine, normalize_robot_df
from .stream_regression import stream_slope
from .stream_popup import show_stream_popup


class StreamWorkPeriodDetector:
    """
    Online work-period detection using overall_force >= work_threshold.
    When leaving work, return a summary dict (mean/peak/start/end).
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

        # Enter a work period
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

        # Exit a work period
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


@dataclass
class StreamConfig:
    source_table: str
    delay: float = 0.1
    max_xticks: int = 50
    window_len: int = 200

    # Work period detector
    work_threshold: float = 0.2
    min_work_points: int = 10

    # Popup spam control
    alert_cooldown_sec: float = 10.0


class StreamingSimulator:
    """
    Stream robot telemetry rows from DB and draw a live plot.

    After each finished work period:
      - compute mean/peak overall_force
      - keep last 10 work periods (current + previous 9)
      - run regression slopes
      - if slope > thresholds -> popup alert
    """

    def __init__(self, db_conn_str: str, config: StreamConfig, alert_thresholds: dict):
        self.config = config
        self.engine = make_stream_engine(db_conn_str, connect_timeout=10)

        self.alert_thr = alert_thresholds
        self._last_alert_time = 0.0

        self.current_offset = 0
        self._buffer_row: Optional[pd.DataFrame] = None

        # Count total rows (once)
        cnt = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {self.config.source_table};",
            self.engine
        )
        self.total_rows = int(cnt.iloc[0]["n"])
        if self.total_rows <= 0:
            raise RuntimeError(f"Source table '{self.config.source_table}' is empty.")

        # Probe one row to detect schema + axis columns
        self._buffer_row = pd.read_sql_query(
            f"SELECT * FROM {self.config.source_table} ORDER BY recorded_at LIMIT 1 OFFSET 0;",
            self.engine
        )
        self._buffer_row = normalize_robot_df(self._buffer_row)

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

        # Work period detector and last-10 buffers
        self.wp = StreamWorkPeriodDetector(
            work_threshold=self.config.work_threshold,
            min_points=self.config.min_work_points,
        )
        self.last_mean: List[float] = []
        self.last_peak: List[float] = []

        # Plot init
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.x_data: List[pd.Timestamp] = []
        self.y_data_dict: Dict[str, List[float]] = {col: [] for col in self.axis_cols}

        print(f"✅ Loaded {self.total_rows} rows from '{self.config.source_table}'.")
        print(f"✅ Axis columns: {self.axis_cols}")
        print(f"✅ Stream thresholds: {self.alert_thr}")

    def _get_next_row(self) -> pd.DataFrame:
        """Fetch exactly one row from DB by offset."""
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
        return normalize_robot_df(row)

    def _maybe_popup(self, mean_slope: float, peak_slope: float, finished: dict) -> None:
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

        # Predicted failure time = 14 days after end_time (per your description)
        try:
            predicted_failure_time = pd.to_datetime(finished["end_time"]) + pd.Timedelta(days=14)
        except Exception:
            predicted_failure_time = "N/A"

        msg = (
            f"Abnormal trend detected!\n\n"
            f"Work period finished:\n"
            f"  Start: {finished['start_time']}\n"
            f"  End:   {finished['end_time']}\n"
            f"  Points: {finished['n_points']}\n\n"
            f"Prediction:\n"
            f"  Possible failure time: {predicted_failure_time}\n"
            f"  Type: {', '.join(fault_types)}\n\n"
            f"Slopes (last 10 work periods):\n"
            f"  Mean slope: {mean_slope:.6f} (thr={mean_thr:.6f})\n"
            f"  Peak slope: {peak_slope:.6f} (thr={peak_thr:.6f})\n"
        )
        show_stream_popup("Robot Alert", msg)

    def nextDataPoint(self):
        """Stream one row, update plot, update detector, maybe popup."""
        if self.current_offset >= self.total_rows:
            print("All data points have been streamed.")
            return None

        row = self._get_next_row()
        if row.empty:
            return None

        ts = pd.to_datetime(row["recorded_at"].values[0])

        # overall_force = mean(abs(axis_i)) across axes
        overall_force = float(row[self.axis_cols].abs().mean(axis=1).iloc[0])

        # Work-period update
        finished = self.wp.update(ts, overall_force)
        if finished is not None:
            self.last_mean.append(float(finished["mean_value"]))
            self.last_peak.append(float(finished["peak_value"]))

            self.last_mean = self.last_mean[-10:]
            self.last_peak = self.last_peak[-10:]

            if len(self.last_mean) == 10 and len(self.last_peak) == 10:
                mean_slope = stream_slope(self.last_mean)
                peak_slope = stream_slope(self.last_peak)
                self._maybe_popup(mean_slope, peak_slope, finished)

        # Plot update buffers
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

        self.current_offset += 1
        time.sleep(self.config.delay)
        return row
