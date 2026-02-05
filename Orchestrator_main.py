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

# NEW: thresholds are fit on TRAIN only
from src.thresholds import fit_thresholds_on_train

# NEW: alerts are detected on TEST only
from src.alerts import detect_alerts_on_test


def load_config(path: str) -> dict:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_dirs(config: dict) -> None:
    """Create folders declared in config (if missing)."""
    ensure_dir(config["paths"]["raw_dir"])
    ensure_dir(config["paths"]["preprocessed_dir"])
    ensure_dir(config["paths"]["models_dir"])

    # results_csv may be nested under data/experiments or experiments/
    results_csv = config["paths"].get("results_csv", "")
    if results_csv:
        ensure_dir(os.path.dirname(results_csv))

    threshold_csv = config["paths"].get("threshold_csv", "")
    if threshold_csv:
        ensure_dir(os.path.dirname(threshold_csv))


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear Regression MLOps Pipeline (Train/Test Split + Alerts)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["preprocess", "split", "train", "evaluate", "alerts", "all"],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)

    # ----------------------------------------
    # Step 1: Preprocess raw -> period summary
    # ----------------------------------------
    if args.step in ["preprocess", "all"]:
        print("=== Step: Preprocess (raw -> period summary) ===")
        run_preprocessing_pipeline(config)
        print("Preprocessing done.\n")

    # ---------------------------------------------------
    # Step 2: Split period summary -> TRAIN/TEST preprocessed
    # ---------------------------------------------------
    if args.step in ["split", "all"]:
        print("=== Step: Split (period_csv -> train/test preprocessed) ===")
        out = split_period_csv_to_train_test(config)
        print("Split done. Saved:")
        print(" -", out["preprocessed_train_csv"])
        print(" -", out["preprocessed_test_csv"])
        print()

    # ---------------------------------------------------
    # Step 3: Train model (theta table) on TRAIN only
    # ---------------------------------------------------
    if args.step in ["train", "all"]:
        print("=== Step: Model (train interval theta table on TRAIN only) ===")
        build_interval_theta_table(config)  # must read paths.preprocessed_train_csv internally
        print("Model training done. Saved:", config["paths"]["theta_table_csv"], "\n")

    # ---------------------------------------------------
    # Step 4: Evaluation on TRAIN only
    # ---------------------------------------------------
    if args.step in ["evaluate", "all"]:
        print("=== Step: Evaluation (TRAIN only) ===")
        evaluate_all_intervals(config)  # must read paths.preprocessed_train_csv internally
        print("Evaluation done.\n")

    # ---------------------------------------------------
    # Step 5: Alerts/Testing
    #   - Fit thresholds on TRAIN
    #   - Detect anomalies on TEST
    #   - Output ONLY anomaly intervals to results_csv
    #   - If none: write empty CSV and print message
    # ---------------------------------------------------
    if args.step in ["alerts", "all"]:
        print("=== Step: Alerts/Testing (TEST only) ===")

        # 5.1 Fit thresholds using TRAIN only (no leakage)
        fit_thresholds_on_train(config)

        # 5.2 Detect anomalies using TEST only
        detect_alerts_on_test(config)

        print("Alerts/Testing done.\n")

    print("✅ Pipeline finished.")


if __name__ == "__main__":
    main()
