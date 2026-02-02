import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import load_yaml_config, load_csv, require_columns

def predict(theta0, theta1, x):
    return theta0 + theta1 * x

def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def plot_regression(x, y, y_scratch, y_sklearn, interval_id, target_name, plot_dir):
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
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()

def evaluate(cfg: dict) -> pd.DataFrame:
    data_path = cfg["data"]["preprocessed_path"]
    theta_path = cfg["outputs"]["theta_table_path"]
    out_path = cfg["outputs"]["evaluated_table_path"]
    plot_dir = cfg["outputs"]["plot_dir"]

    predictor = cfg["model"]["predictor_feature"]
    mean_target = cfg["model"]["targets"]["mean"]
    peak_target = cfg["model"]["targets"]["peak"]

    df = load_csv(data_path)
    theta_df = load_csv(theta_path)

    require_columns(df, ["interval_id", predictor, mean_target, peak_target], name="preprocessed")
    require_columns(theta_df, ["interval_id"], name="theta_table")

    df = df.dropna(subset=["interval_id", predictor, mean_target, peak_target]).copy()
    df = df.sort_values(["interval_id", predictor])

    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for _, trow in theta_df.iterrows():
        interval_id = trow["interval_id"]
        g = df[df["interval_id"] == interval_id].sort_values(predictor)
        if len(g) < 2:
            continue

        x = g[predictor].to_numpy(dtype=float)

        y_mean = g[mean_target].to_numpy(dtype=float)
        y_peak = g[peak_target].to_numpy(dtype=float)

        # Mean predictions
        y_mean_scratch = predict(trow["scratch_mean_theta0"], trow["scratch_mean_theta1"], x)
        y_mean_sklearn = predict(trow["sklearn_mean_theta0"], trow["sklearn_mean_theta1"], x)

        m_scratch = compute_metrics(y_mean, y_mean_scratch)
        m_sklearn = compute_metrics(y_mean, y_mean_sklearn)

        plot_regression(x, y_mean, y_mean_scratch, y_mean_sklearn, interval_id, "mean", plot_dir)

        # Peak predictions
        y_peak_scratch = predict(trow["scratch_peak_theta0"], trow["scratch_peak_theta1"], x)
        y_peak_sklearn = predict(trow["sklearn_peak_theta0"], trow["sklearn_peak_theta1"], x)

        p_scratch = compute_metrics(y_peak, y_peak_scratch)
        p_sklearn = compute_metrics(y_peak, y_peak_sklearn)

        plot_regression(x, y_peak, y_peak_scratch, y_peak_sklearn, interval_id, "peak", plot_dir)

        rows.append({
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

    metrics_df = pd.DataFrame(rows)
    final_df = theta_df.merge(metrics_df, on="interval_id", how="left")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    final_df = evaluate(cfg)

    print(f"Saved evaluated table to: {cfg['outputs']['evaluated_table_path']}")
    print(f"Plots saved to: {cfg['outputs']['plot_dir']}")
    print(final_df.head(10))

if __name__ == "__main__":
    main()