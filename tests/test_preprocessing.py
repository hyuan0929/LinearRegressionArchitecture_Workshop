# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import run_preprocessing_pipeline


def test_preprocessing_outputs(config):
    run_preprocessing_pipeline(config)

    df = pd.read_csv(config["paths"]["processed_csv"])

    required_cols = [
        "interval_id",
        "work_period",
        "mean_value_z",
        "peak_value_z"
    ]

    for col in required_cols:
        assert col in df.columns

    assert df["interval_id"].min() >= 1