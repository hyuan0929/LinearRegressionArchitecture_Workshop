# tests/test_evaluation.py
import pandas as pd
from src.evaluation import evaluate_all_intervals


def test_evaluation_metrics(config):
    df = evaluate_all_intervals(config)

    metric_cols = [
        "scratch_mean_rmse",
        "scratch_peak_rmse",
        "sklearn_mean_r2",
        "sklearn_peak_r2",
    ]

    for col in metric_cols:
        assert col in df.columns