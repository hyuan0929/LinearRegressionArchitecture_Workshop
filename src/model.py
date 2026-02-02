import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_yaml_config, load_csv, require_columns

def hypothesis(theta0: float, theta1: float, x: np.ndarray) -> np.ndarray:
    return theta0 + theta1 * x

def gradient_descent_1d(x: np.ndarray, y: np.ndarray, lr: float, iters: int):
    x = x.astype(float)
    y = y.astype(float)

    m = len(x)
    if m == 0:
        raise ValueError("Empty training set.")

    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iters):
        y_pred = hypothesis(theta0, theta1, x)
        error = y_pred - y

        grad_theta0 = (2.0 / m) * np.sum(error)
        grad_theta1 = (2.0 / m) * np.sum(error * x)

        theta0 -= lr * grad_theta0
        theta1 -= lr * grad_theta1

    return float(theta0), float(theta1)

def fit_scratch_with_centering(x: np.ndarray, y: np.ndarray, lr: float, iters: int):
    mean_x = float(np.mean(x))
    x_center = x - mean_x
    theta0_c, theta1_c = gradient_descent_1d(x_center, y, lr, iters)
    theta0 = theta0_c - theta1_c * mean_x
    theta1 = theta1_c
    return float(theta0), float(theta1)

def fit_sklearn(x: np.ndarray, y: np.ndarray):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return float(model.intercept_), float(model.coef_[0])

def train_per_interval(cfg: dict) -> pd.DataFrame:
    data_path = cfg["data"]["preprocessed_path"]
    out_path = cfg["outputs"]["theta_table_path"]

    lr = float(cfg["model"]["learning_rate"])
    iters = int(cfg["model"]["iterations"])
    predictor = cfg["model"]["predictor_feature"]

    mean_target = cfg["model"]["targets"]["mean"]
    peak_target = cfg["model"]["targets"]["peak"]

    df = load_csv(data_path)

    require_columns(df, ["interval_id", predictor, mean_target, peak_target], name="preprocessed")

    df[predictor] = pd.to_numeric(df[predictor], errors="coerce")
    df[mean_target] = pd.to_numeric(df[mean_target], errors="coerce")
    df[peak_target] = pd.to_numeric(df[peak_target], errors="coerce")
    df = df.dropna(subset=["interval_id", predictor, mean_target, peak_target]).copy()

    rows = []
    for interval_id, g in df.groupby("interval_id"):
        g = g.sort_values(predictor)
        x = g[predictor].to_numpy(dtype=float)

        y_mean = g[mean_target].to_numpy(dtype=float)
        y_peak = g[peak_target].to_numpy(dtype=float)

        s_mean_t0, s_mean_t1 = fit_scratch_with_centering(x, y_mean, lr, iters)
        s_peak_t0, s_peak_t1 = fit_scratch_with_centering(x, y_peak, lr, iters)

        k_mean_t0, k_mean_t1 = fit_sklearn(x, y_mean)
        k_peak_t0, k_peak_t1 = fit_sklearn(x, y_peak)

        rows.append({
            "interval_id": interval_id,
            "n_points": len(g),

            "scratch_mean_theta0": s_mean_t0,
            "scratch_mean_theta1": s_mean_t1,
            "scratch_peak_theta0": s_peak_t0,
            "scratch_peak_theta1": s_peak_t1,

            "sklearn_mean_theta0": k_mean_t0,
            "sklearn_mean_theta1": k_mean_t1,
            "sklearn_peak_theta0": k_peak_t0,
            "sklearn_peak_theta1": k_peak_t1,
        })

    theta_table = pd.DataFrame(rows).sort_values("interval_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    theta_table.to_csv(out_path, index=False)
    return theta_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    theta_table = train_per_interval(cfg)
    print(f"Saved theta table to: {cfg['outputs']['theta_table_path']}")
    print(theta_table.head(10))

if __name__ == "__main__":
    main()