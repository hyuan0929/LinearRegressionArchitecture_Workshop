import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_yaml_config, load_csv, require_columns

def preprocess(cfg: dict) -> pd.DataFrame:
    input_path = cfg["data"]["period_summary_path"]
    output_path = cfg["data"]["preprocessed_path"]

    interval_size = int(cfg["preprocessing"]["interval_size"])
    test_size = float(cfg["preprocessing"]["test_size"])
    fill_method = str(cfg["preprocessing"]["fill_method"]).lower()
    do_standardize = bool(cfg["preprocessing"]["standardize_features"])

    feature_cols = ["mean_value", "peak_value"]
    time_cols = ["period_start_time", "period_end_time"]

    df = load_csv(input_path)

    require_columns(df, ["work_period"] + feature_cols + time_cols, name="period_summary")

    # Parse times
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # Drop rows with invalid time
    df = df.dropna(subset=["period_end_time"]).copy()

    # Ensure numeric
    df["work_period"] = pd.to_numeric(df["work_period"], errors="coerce")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["work_period"] + feature_cols).copy()

    # Sort by time (time-aware)
    df = df.sort_values("period_end_time").reset_index(drop=True)

    # Create interval_id (every N work periods)
    df["interval_id"] = ((df["work_period"] - 1) // interval_size) + 1

    # Handle missing values in features
    if fill_method == "ffill":
        df[feature_cols] = df[feature_cols].ffill().bfill()
    elif fill_method == "mean":
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    else:  # median
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    if df[feature_cols].isna().any().any():
        raise ValueError("Missing values remain after filling. Check input data.")

    # Train/test split (time-aware)
    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough periods ({n}) to split reliably.")

    split_idx = int(np.floor((1 - test_size) * n))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Standardize features (fit on train, transform both)
    if do_standardize:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        test_scaled = scaler.transform(test_df[feature_cols])

        for i, col in enumerate(feature_cols):
            train_df[f"{col}_z"] = train_scaled[:, i]
            test_df[f"{col}_z"] = test_scaled[:, i]
    else:
        for col in feature_cols:
            train_df[f"{col}_z"] = train_df[col]
            test_df[f"{col}_z"] = test_df[col]

    processed_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    processed_df["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    return processed_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    df = preprocess(cfg)
    print(f"Saved preprocessed data to: {cfg['data']['preprocessed_path']}")
    print(df.head(10))

if __name__ == "__main__":
    main()