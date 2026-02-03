# Orchestrator_main.py
import argparse
import os
import yaml

from src.utils import ensure_dir
from src.preprocessing import run_preprocessing_pipeline
from src.model import build_interval_theta_table
from src.evaluation import evaluate_all_intervals
from src.alerts import build_results_csv


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Linear Regression MLOps Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to YAML config")
    parser.add_argument("--step", type=str, default="all",
                        choices=["preprocess", "train", "evaluate", "alerts", "all"],
                        help="Which pipeline step to run")
    args = parser.parse_args()

    config = load_config(args.config)

    # ensure folders
    ensure_dir(config["paths"]["raw_dir"])
    ensure_dir(config["paths"]["preprocessed_dir"])
    ensure_dir(config["paths"]["models_dir"])
    ensure_dir(config["paths"]["experiments_dir"])

    if args.step in ["preprocess", "all"]:
        print("=== Step: Preprocess ===")
        run_preprocessing_pipeline(config)
        print("Preprocessing done.\n")

    if args.step in ["train", "all"]:
        print("=== Step: Model (theta table) ===")
        build_interval_theta_table(config)
        print("Model training done.\n")

    if args.step in ["evaluate", "all"]:
        print("=== Step: Evaluation ===")
        evaluate_all_intervals(config)
        print("Evaluation done.\n")

    if args.step in ["alerts", "all"]:
        print("=== Step: Alerts & Experiment Tracking ===")
        build_results_csv(config)
        print("Alerts/results.csv done.\n")

    print("✅ Pipeline finished.")


if __name__ == "__main__":
    main()