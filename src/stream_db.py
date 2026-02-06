# src/stream_db.py
from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError


def make_stream_engine(conn_str: str, connect_timeout: int = 10):
    """
    Create SQLAlchemy engine with timeout for streaming dashboard.
    Prevents hanging forever when DB/network is slow.
    """
    return create_engine(
        conn_str,
        pool_pre_ping=True,
        connect_args={"connect_timeout": connect_timeout},
    )


def stream_table_empty(engine, table_name: str) -> bool:
    """Return True if table missing or has zero rows."""
    try:
        if not inspect(engine).has_table(table_name):
            return True
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1")).first() is None
    except SQLAlchemyError:
        return True


def normalize_robot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns + parse recorded_at.
    Keep axis_1..axis_8, drop extra axis columns if present.
    """
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_").replace("#", "") for c in df.columns]

    if "time" in df.columns and "recorded_at" not in df.columns:
        df.rename(columns={"time": "recorded_at"}, inplace=True)

    drop_cols = ["axis_9", "axis_10", "axis_11", "axis_12", "axis_13", "axis_14"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")

    return df


def init_stream_table_from_csv(
    engine,
    table_name: str,
    csv_path: str,
) -> int:
    """
    Initialize table from local CSV if table is missing/empty.
    Uses if_exists="replace" ONLY on first init.
    Returns the row count after init (or existing count).
    """
    if not stream_table_empty(engine, table_name):
        with engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name};")).scalar())

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_robot_df(df)

    if "recorded_at" not in df.columns:
        raise ValueError("CSV must contain 'recorded_at' (or 'time') column.")

    df.to_sql(table_name, engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name};")).scalar())
