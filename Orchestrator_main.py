# Orchestrator_main.py
from __future__ import annotations

import argparse
import os
import yaml

from src.utils import ensure_dir
from src.preprocessing import run_preprocessing_pipeline
from src.splitter import split_period_csv_to_train_test
from src.model import build_interval_theta_table
from src.evaluation import evaluate_all_intervals
from src.alerts import build_results_csv


def load_config(path: str) -> dict:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    End-to-end pipeline:

    1) Preprocess raw -> period_csv (work-period summary)
    2) Split period_csv -> preprocessed_train_csv / preprocessed_test_csv
    3) Train interval regression on TRAIN only -> theta_table_csv
    4) Evaluate models on TRAIN only -> evaluated_csv (+ plots)
    5) Run alerts/testing on TEST only -> results_csv
       - Output only anomaly intervals
       - If no anomalies: write empty file and print message
    """
    parser = argparse.ArgumentParser(description="Linear Regression MLOps Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["preprocess", "split", "train", "evaluate", "alerts", "all"],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure folders exist
    ensure_dir(config["paths"]["raw_dir"])
    ensure_dir(config["paths"]["preprocessed_dir"])
    ensure_dir(config["paths"]["models_dir"])
    ensure_dir(config["paths"]["experiments_dir"])

    if args.step in ["preprocess", "all"]:
        print("=== Step: Preprocess (raw -> period summary) ===")
        run_preprocessing_pipeline(config)
        print("Preprocessing done.\n")

    if args.step in ["split", "all"]:
        print("=== Step: Split (period_csv -> train/test preprocessed) ===")
        out = split_period_csv_to_train_test(config)
        print("Split done. Saved:")
        print(" -", out["preprocessed_train_csv"])
        print(" -", out["preprocessed_test_csv"])
        print()

    if args.step in ["train", "all"]:
        print("=== Step: Model (train interval theta table on TRAIN only) ===")
        build_interval_theta_table(config)
        print("Model training done. Saved:", config["paths"]["theta_table_csv"], "\n")

    if args.step in ["evaluate", "all"]:
        print("=== Step: Evaluation (TRAIN only) ===")
        evaluate_all_intervals(config)
        print("Evaluation done.\n")

    if args.step in ["alerts", "all"]:
        print("=== Step: Alerts/Testing (TEST only) ===")
        build_results_csv(config)
        print("Alerts/testing done. Saved:", config["paths"]["results_csv"], "\n")

    print("✅ Pipeline finished.")


if __name__ == "__main__":
    main()
