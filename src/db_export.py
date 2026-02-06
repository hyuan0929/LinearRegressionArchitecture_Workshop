from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine


def export_postgres_table_to_csv(
    connstr: str,
    table_name: str,
    output_csv: str,
) -> None:
    """
    Export a PostgreSQL table to CSV.

    Parameters
    ----------
    connstr : str
        PostgreSQL connection string (e.g. Neon)
    table_name : str
        Table name to export
    output_csv : str
        Path to output CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    engine = create_engine(connstr)

    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, engine)

    df.to_csv(output_csv, index=False)
    print(f"✅ Exported table '{table_name}' to {output_csv}")
