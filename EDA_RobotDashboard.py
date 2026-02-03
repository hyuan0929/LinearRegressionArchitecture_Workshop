import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError


# -----------------------------
# Database configuration
# -----------------------------
TABLE_NAME = "workshop_robot_data"
CONN_STR = (
    "postgresql://neondb_owner:npg_Sh8bV3HjZvkd@ep-plain-scene-ahmzh8by-pooler.c-3.us-east-1.aws.neon.tech/"
    "neondb?sslmode=require"
)
ENGINE = create_engine(CONN_STR)



def is_table_empty(table_name: str, engine=ENGINE) -> bool:
    """Return True if the table does not exist or has zero rows.

    Keep this intentionally simple:
    - if table missing -> empty
    - else run SELECT 1 ... LIMIT 1
    """
    try:
        if not inspect(engine).has_table(table_name):
            return True
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1")).first() is None
    except SQLAlchemyError:
        # safest default for assignment flows
        return True

def _preprocess_robot_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = [col.lower().strip().replace(" ", "_").replace("#", "") for col in df.columns]

    # Rename time column
    if "time" in df.columns and "recorded_at" not in df.columns:
        df.rename(columns={"time": "recorded_at"}, inplace=True)

    # Drop unused axis columns (keep axis_1 ~ axis_8)
    cols_to_drop = ["axis_9", "axis_10", "axis_11", "axis_12", "axis_13", "axis_14"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Ensure datetime type
    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    return df


def init_data_once(
    source_table: str = "robot_data",
    csv_path: str = "data/raw/RMBR4-2_export_test.csv",
) -> None:
    """Initialize DB data only once.

    Logic:
    1) Check is_table_empty(source_table)
    2) Only if empty -> load CSV -> preprocess -> write to DB
    """
    if not is_table_empty(source_table):
        return

    df = pd.read_csv(csv_path)
    df = _preprocess_robot_csv(df)

    # Use 'replace' here ONLY because we are sure it's the first init
    df.to_sql(source_table, ENGINE, if_exists="replace", index=False)
    print(f"Initialized '{source_table}' with {len(df)} rows from {csv_path!r}.")


@dataclass
class StreamConfig:
    source_table: str = "robot_data"          # where we read from (online DB)
    delay: float = 0.1
    max_xticks: int = 50
    window_len: int = 40


class StreamingSimulator:
    """Stream robot telemetry data FROM the online DB.

    Each call to nextDataPoint():
      1) Reads ONE row from source_table (already in DB)
      2) Updates the chart
    """

    def __init__(self, db_conn_str: str, config: StreamConfig):
        self.config = config
        self.engine = create_engine(db_conn_str)
        self.current_offset = 0
        self._buffer_row: Optional[pd.DataFrame] = None

        # Count total rows once (so we know when to stop)
        cnt = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {self.config.source_table};",
            self.engine
        )
        self.total_rows = int(cnt.iloc[0]["n"])

        if self.total_rows <= 0:
            raise RuntimeError(f"Source table '{self.config.source_table}' is empty.")

        # Fetch ONE row up-front to (a) validate schema and (b) detect axis columns.
        self._buffer_row = pd.read_sql_query(
            f"SELECT * FROM {self.config.source_table} ORDER BY recorded_at LIMIT 1 OFFSET 0;",
            self.engine
        )
        self._buffer_row = _preprocess_robot_csv(self._buffer_row)

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

        # Plot init

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.x_data: List[pd.Timestamp] = []
        self.y_data_dict: Dict[str, List[float]] = {col: [] for col in self.axis_cols}

        print(f"Loaded {self.total_rows} rows from DB table '{self.config.source_table}'.")
        print(f"Detected Y-axis columns: {self.axis_cols}")

    def _get_next_row(self) -> pd.DataFrame:
        """Fetch exactly ONE record from source_table, in a stable order."""
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
        return _preprocess_robot_csv(row)


    def nextDataPoint(self):
        if self.current_offset >= self.total_rows:
            print("All data points have been streamed.")
            return None

        row = self._get_next_row()  # keep as DataFrame for to_sql

        # 2) update plot
        ts = pd.to_datetime(row["recorded_at"].values[0])
        self.x_data.append(ts)
        self.x_data = self.x_data[-self.config.window_len :]

        for col in self.axis_cols:
            self.y_data_dict[col].append(float(row[col].values[0]))
            self.y_data_dict[col] = self.y_data_dict[col][-self.config.window_len :]

        self.ax.clear()
        for col in self.axis_cols:
            self.ax.plot(self.x_data, self.y_data_dict[col], label=col, linewidth=1)

        self.ax.set_title(
            f"Streaming Robot Axis Data ({self.current_offset + 1}/{self.total_rows})"
        )
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

        # 3) advance index and sleep
        self.current_offset += 1
        time.sleep(self.config.delay)

        return row


def main():
    # 1) init source table once (only if empty)
    init_data_once(source_table="robot_data", csv_path="data/raw/RMBR4-2_export_test.csv")

    # 2) stream FROM online DB table 'robot_data'
    ss = StreamingSimulator(
        db_conn_str=CONN_STR,
        config=StreamConfig(
            source_table="robot_data",            
            delay=0.1,
        ),
    )

    while True:
        result = ss.nextDataPoint()
        if result is None:
            break


if __name__ == "__main__":
    main()
