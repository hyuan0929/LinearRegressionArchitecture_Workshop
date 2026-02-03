# tests/test_alerts.py
import pandas as pd
from src.alerts import build_results_csv


def test_alert_generation(config):
    results = build_results_csv(config)

    assert not results.empty
    assert "failure_type" in results.columns

    valid_types = {"No Alert", "Aging", "Fault", "Aging + Fault"}
    assert set(results["failure_type"]).issubset(valid_types)