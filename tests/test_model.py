# tests/test_model.py
import pandas as pd
from src.model import build_interval_theta_table


def test_interval_theta_table(config):
    theta_df = build_interval_theta_table(config)

    assert not theta_df.empty

    theta_cols = [
        "scratch_mean_theta1",
        "scratch_peak_theta1",
        "sklearn_mean_theta1",
        "sklearn_peak_theta1",
    ]

    for col in theta_cols:
        assert theta_df[col].notna().all()