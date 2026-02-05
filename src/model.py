# src/model.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.linear_model import LinearRegression
from .utils import save_csv


def hypothesis(theta0: float, theta1: float, x: np.ndarray) -> np.ndarray:
    return theta0 + theta1 * x


def mse_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def gradient_descent_1d(x: np.ndarray, y: np.ndarray, lr: float, iters: int):
    x = x.astype(float)
    y = y.astype(float)
    m = len(x)
    if m == 0:
        raise ValueError("Empty training set.")

    theta0, theta1 = 0.0, 0.0
    history = []

    for _ in range(iters):
        y_pred = hypothesis(theta0, theta1, x)
        error = y_pred - y

        grad_theta0 = (2.0 / m) * np.sum(error)
        grad_theta1 = (2.0 / m) * np.sum(error * x)

        theta0 -= lr * grad_theta0
        theta1 -= lr * grad_theta1

        history.append(mse_cost(y, y_pred))

    return float(theta0), float(theta1), history


def sklearn_fit_1d(x: np.ndarray, y: np.ndarray):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    theta0 = float(model.intercept_)
    theta1 = float(model.coef_[0])
    return theta0, theta1


def build_interval_theta_table(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Train interval-level regression using TRAIN preprocessed data only.
    Output: interval theta table (scratch + sklearn) for mean & peak.
    """
    preprocessed_train_csv = config["paths"]["preprocessed_train_csv"]
    theta_out_csv = config["paths"]["theta_table_csv"]

    lr = float(config["training"]["learning_rate"])
    iters = int(config["training"]["iterations"])
    use_z = bool(config["training"]["use_standardized"])

    x_col = "work_period"
    group_col = "interval_id"

    y_mean = "mean_value_z" if use_z else "mean_value"
    y_peak = "peak_value_z" if use_z else "peak_value"

    df = pd.read_csv(preprocessed_train_csv)
    need = [group_col, x_col, y_mean, y_peak, "period_start_time", "period_end_time"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train preprocessed data: {missing}")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_mean] = pd.to_numeric(df[y_mean], errors="coerce")
    df[y_peak] = pd.to_numeric(df[y_peak], errors="coerce")
    df = df.dropna(subset=[group_col, x_col, y_mean, y_peak]).copy()

    rows: List[dict] = []

    for interval_id, g in df.groupby(group_col):
        g = g.sort_values(x_col).copy()
        x = g[x_col].to_numpy(dtype=float)

        # Save interval work_period range (required by your statement)
        start_wp = int(g[x_col].min())
        end_wp = int(g[x_col].max())

        # center x for GD stability
        x_center = x - x.mean()
        mean_x = float(x.mean())

        # mean target
        y1 = g[y_mean].to_numpy(dtype=float)
        s0_m, s1_m, _ = gradient_descent_1d(x_center, y1, lr=lr, iters=iters)
        s0_m_unc = float(s0_m - s1_m * mean_x)
        s1_m_unc = float(s1_m)
        sk0_m, sk1_m = sklearn_fit_1d(x, y1)

        # peak target
        y2 = g[y_peak].to_numpy(dtype=float)
        s0_p, s1_p, _ = gradient_descent_1d(x_center, y2, lr=lr, iters=iters)
        s0_p_unc = float(s0_p - s1_p * mean_x)
        s1_p_unc = float(s1_p)
        sk0_p, sk1_p = sklearn_fit_1d(x, y2)

        rows.append({
            "interval_id": int(interval_id),
            "start_work_period": start_wp,
            "end_work_period": end_wp,
            "n_periods": int(len(g)),

            "scratch_mean_theta0": s0_m_unc,
            "scratch_mean_theta1": s1_m_unc,
            "scratch_peak_theta0": s0_p_unc,
            "scratch_peak_theta1": s1_p_unc,

            "sklearn_mean_theta0": sk0_m,
            "sklearn_mean_theta1": sk1_m,
            "sklearn_peak_theta0": sk0_p,
            "sklearn_peak_theta1": sk1_p,

            "learning_rate": lr,
            "iterations": iters,
            "target_space": "z" if use_z else "raw",
        })

    theta_df = pd.DataFrame(rows).sort_values("interval_id").reset_index(drop=True)
    save_csv(theta_df, theta_out_csv)
    return theta_df
