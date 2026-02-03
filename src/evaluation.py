# src/evaluation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import ensure_dir, save_csv


def predict(theta0: float, theta1: float, x: np.ndarray) -> np.ndarray:
    return theta0 + theta1 * x


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_scratch: np.ndarray,
    y_sklearn: np.ndarray,
    interval_id: int,
    target_name: str,
    out_dir: str,
) -> str:
    ensure_dir(out_dir)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="Data", color="black")
    plt.plot(x, y_scratch, label="Scratch", linewidth=2)
    plt.plot(x, y_sklearn, label="Sklearn", linestyle="--", linewidth=2)
    plt.xlabel("Work Period")
    plt.ylabel(target_name)
    plt.title(f"Interval {interval_id} – {target_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = f"interval_{interval_id}_{target_name}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath)
    plt.close()
    return fpath


def evaluate_all_intervals(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Read preprocessed data + theta table and generate:
    - evaluated theta table with RMSE/MAE/R2 for mean & peak (scratch & sklearn)
    - plots per interval and target
    Output: interval_theta_table_evaluated.csv and plots/*
    """
    data_path = config["paths"]["preprocessed_csv"]
    theta_path = config["paths"]["theta_table_csv"]
    evaluated_out = config["paths"]["evaluated_csv"]

    plot_dir = os.path.join(config["paths"]["models_dir"], "plots")
    ensure_dir(plot_dir)

    use_z = bool(config["training"]["use_standardized"])
    x_col = "work_period"
    group_col = "interval_id"
    mean_target = "mean_value_z" if use_z else "mean_value"
    peak_target = "peak_value_z" if use_z else "peak_value"

    df = pd.read_csv(data_path)
    theta_df = pd.read_csv(theta_path)

    need = [group_col, x_col, mean_target, peak_target]
    df = df.dropna(subset=need).copy()

    results = []

    for _, row in theta_df.iterrows():
        interval_id = int(row["interval_id"])
        g = df[df[group_col] == interval_id].sort_values(x_col)
        if len(g) < 2:
            continue

        x = g[x_col].to_numpy(dtype=float)

        # mean
        y_mean = g[mean_target].to_numpy(dtype=float)
        y_mean_scratch = predict(row["scratch_mean_theta0"], row["scratch_mean_theta1"], x)
        y_mean_sklearn = predict(row["sklearn_mean_theta0"], row["sklearn_mean_theta1"], x)
        m_scratch = compute_metrics(y_mean, y_mean_scratch)
        m_sklearn = compute_metrics(y_mean, y_mean_sklearn)

        plot_regression(x, y_mean, y_mean_scratch, y_mean_sklearn, interval_id, "mean", plot_dir)

        # peak
        y_peak = g[peak_target].to_numpy(dtype=float)
        y_peak_scratch = predict(row["scratch_peak_theta0"], row["scratch_peak_theta1"], x)
        y_peak_sklearn = predict(row["sklearn_peak_theta0"], row["sklearn_peak_theta1"], x)
        p_scratch = compute_metrics(y_peak, y_peak_scratch)
        p_sklearn = compute_metrics(y_peak, y_peak_sklearn)

        plot_regression(x, y_peak, y_peak_scratch, y_peak_sklearn, interval_id, "peak", plot_dir)

        results.append({
            "interval_id": interval_id,

            "scratch_mean_rmse": m_scratch["rmse"],
            "scratch_mean_mae": m_scratch["mae"],
            "scratch_mean_r2": m_scratch["r2"],

            "sklearn_mean_rmse": m_sklearn["rmse"],
            "sklearn_mean_mae": m_sklearn["mae"],
            "sklearn_mean_r2": m_sklearn["r2"],

            "scratch_peak_rmse": p_scratch["rmse"],
            "scratch_peak_mae": p_scratch["mae"],
            "scratch_peak_r2": p_scratch["r2"],

            "sklearn_peak_rmse": p_sklearn["rmse"],
            "sklearn_peak_mae": p_sklearn["mae"],
            "sklearn_peak_r2": p_sklearn["r2"],
        })

    metrics_df = pd.DataFrame(results)
    final_df = theta_df.merge(metrics_df, on="interval_id", how="left")

    save_csv(final_df, evaluated_out)
    return final_df