# DataStreamVisualization_main.py
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import yaml

from src.stream_db import make_stream_engine, init_stream_table_from_csv
from src.stream_alert_thresholds import read_stream_alert_thresholds
from src.stream_streaming import StreamingSimulator, StreamConfig


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_root() -> None:
    """
    Ensure the script runs from project root.
    - If launched from notebooks/, go up one level.
    - Also ensure root is in sys.path so 'src' is importable.
    """
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")

    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def run_stream_dashboard(config: Dict[str, Any]) -> None:
    """
    Run streaming visualization + popup alerts.

    Expected YAML keys:
      database.connstr
      paths.stream_threshold_csv   (or fallback to paths.threshold_csv)
      stream.source_table
      stream.raw_csv
      stream.delay
      stream.plot_window_len
      stream.max_xticks
      stream.work_threshold
      stream.min_work_points
      stream.alert_cooldown_sec
      stream.init_from_csv_if_empty
    """
    connstr = config["database"]["connstr"]

    # Stream section (with sane defaults)
    stream_cfg = config.get("stream", {})
    source_table = str(stream_cfg.get("source_table", config["database"].get("source_table", "robot_data")))
    raw_csv = str(stream_cfg.get("raw_csv", config["paths"].get("raw_csv", "data/raw/RMBR4-2_export_test.csv")))

    delay = float(stream_cfg.get("delay", 0.1))
    window_len = int(stream_cfg.get("plot_window_len", 200))
    max_xticks = int(stream_cfg.get("max_xticks", 50))

    work_threshold = float(stream_cfg.get("work_threshold", 0.2))
    min_work_points = int(stream_cfg.get("min_work_points", 10))
    alert_cooldown_sec = float(stream_cfg.get("alert_cooldown_sec", 10.0))

    init_from_csv_if_empty = bool(stream_cfg.get("init_from_csv_if_empty", True))

    # Thresholds file (prefer stream_threshold_csv, fallback to threshold_csv)
    thresholds_csv = (
        config.get("paths", {}).get("stream_threshold_csv")
        or config.get("paths", {}).get("threshold_csv")
        or "data/models/alert_thresholds_stream.csv"
    )

    print("=== Data Stream Visualization (Neon -> Live Plot + Popup Alerts) ===")
    print(f"DB table: {source_table}")
    print(f"Raw CSV (for init if empty): {raw_csv}")
    print(f"Stream thresholds CSV: {thresholds_csv}")
    print()

    # 1) Make engine + optionally init table once
    engine = make_stream_engine(connstr, connect_timeout=10)

    if init_from_csv_if_empty:
        try:
            n = init_stream_table_from_csv(engine, source_table, raw_csv)
            print(f"✅ Table '{source_table}' ready. Rows = {n}\n")
        except Exception as e:
            print(f"⚠️ Could not init table from CSV (continuing): {e}\n")

    # 2) Load streaming thresholds
    thr = read_stream_alert_thresholds(thresholds_csv)

    # 3) Run streaming simulator
    sim = StreamingSimulator(
        db_conn_str=connstr,
        config=StreamConfig(
            source_table=source_table,
            delay=delay,
            window_len=window_len,
            max_xticks=max_xticks,
            work_threshold=work_threshold,
            min_work_points=min_work_points,
            alert_cooldown_sec=alert_cooldown_sec,
        ),
        alert_thresholds=thr,
    )

    # 4) Stream loop
    while True:
        row = sim.nextDataPoint()
        if row is None:
            break

    print("\n✅ Streaming finished.")


def main() -> None:
    ensure_project_root()

    parser = argparse.ArgumentParser(
        description="Data Stream Visualization (Neon PostgreSQL -> Live Plot + Work-Period Trend Alerts)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Optional flag: allow disabling stream via YAML (stream.enabled: false)
    if "stream" in config and isinstance(config["stream"], dict):
        if config["stream"].get("enabled") is False:
            print("stream.enabled is false in config. Nothing to run.")
            return

    run_stream_dashboard(config)


if __name__ == "__main__":
    main()
