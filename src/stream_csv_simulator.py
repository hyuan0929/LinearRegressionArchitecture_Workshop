# src/stream_csv_simulator.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import text

from .stream_db import make_stream_engine, normalize_robot_df


def detect_axis_cols(df: pd.DataFrame, max_axis: int = 8) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith("axis_"):
            parts = c.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                if 1 <= k <= max_axis:
                    cols.append(c)
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


@dataclass
class CSVStreamConfig:
    csv_path: str
    db_conn_str: str
    table_name: str

    delay: float = 2.0
    window_len: int = 200
    max_xticks: int = 50

    # if table doesn't exist, create it using CSV header/schema
    create_table_if_missing: bool = True


class StreamingSimulator:
    """
    Simulate streaming from a CSV.

    Each nextDataPoint():
      1) Read next record from in-memory DataFrame
      2) Insert it into the database table
      3) Update a real-time plot
      4) Sleep for a configured delay
    """

    def __init__(self, config: CSVStreamConfig):
        self.config = config

        # 1) Load entire CSV into memory + normalize columns (re-use your function)
        raw = pd.read_csv(self.config.csv_path)
        self.df = normalize_robot_df(raw)

        if "recorded_at" not in self.df.columns:
            raise ValueError("CSV must contain 'recorded_at' (or 'time') column after normalization.")

        self.df = self.df.dropna(subset=["recorded_at"]).sort_values("recorded_at").reset_index(drop=True)
        self.total_rows = len(self.df)
        self.current_index = 0

        # 2) Detect axis columns
        self.axis_cols = detect_axis_cols(self.df, max_axis=8)
        if not self.axis_cols:
            raise ValueError("No axis columns found (expected axis_1..axis_8).")

        # 3) DB engine (re-use your function)
        self.engine = make_stream_engine(self.config.db_conn_str, connect_timeout=10)

        # 4) Ensure table exists (optional but useful for demo)
        if self.config.create_table_if_missing:
            self._ensure_table_exists()

        # 5) Plot init
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.x_data: List[pd.Timestamp] = []
        self.y_data: Dict[str, List[float]] = {c: [] for c in self.axis_cols}

        print(f"✅ CSV loaded: {self.total_rows} rows")
        print(f"✅ Axis columns: {self.axis_cols}")
        print(f"✅ Streaming into DB table: {self.config.table_name}")

    def _ensure_table_exists(self) -> None:
        """
        Create table schema if missing:
        Try SELECT 1; if fails, create schema using df.head(0).to_sql(replace).
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"SELECT 1 FROM {self.config.table_name} LIMIT 1;"))
        except Exception:
            # schema-only create
            self.df.head(0).to_sql(self.config.table_name, self.engine, if_exists="replace", index=False)
            print(f"🛠️ Created table '{self.config.table_name}' (schema only).")

    def _insert_one_row(self, row_df: pd.DataFrame) -> None:
        """
        Insert exactly one row (as a 1-row DataFrame) into DB table.
        """
        row_df.to_sql(
            self.config.table_name,
            self.engine,
            if_exists="append",
            index=False,
            method="multi",
        )

    def _update_plot(self, ts: pd.Timestamp, row_df: pd.DataFrame) -> None:
        self.x_data.append(ts)
        self.x_data = self.x_data[-self.config.window_len :]

        for c in self.axis_cols:
            self.y_data[c].append(float(row_df[c].iloc[0]))
            self.y_data[c] = self.y_data[c][-self.config.window_len :]

        self.ax.clear()
        for c in self.axis_cols:
            self.ax.plot(self.x_data, self.y_data[c], label=c, linewidth=1)

        self.ax.set_title(f"Streaming Robot Axis Data ({self.current_index + 1}/{self.total_rows})")
        self.ax.set_xlabel("recorded_at")
        self.ax.set_ylabel("Axis Values")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        if len(self.x_data) > self.config.max_xticks:
            step = max(1, len(self.x_data) // self.config.max_xticks)
            self.ax.set_xticks(self.x_data[::step])

        self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def nextDataPoint(self) -> Optional[pd.DataFrame]:
        """
        Stream one record. Returns the 1-row DataFrame, or None when finished.
        """
        if self.current_index >= self.total_rows:
            print("✅ All data points have been streamed.")
            return None

        # 1) Read next record
        row_df = self.df.iloc[[self.current_index]].copy()

        # 2) Insert into DB
        self._insert_one_row(row_df)

        # 3) Update plot
        ts = pd.to_datetime(row_df["recorded_at"].iloc[0])
        self._update_plot(ts, row_df)

        # 4) Advance + delay
        self.current_index += 1
        time.sleep(self.config.delay)
        return row_df
