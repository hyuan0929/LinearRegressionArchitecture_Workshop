import os
import pandas as pd

def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")
    return cfg

def require_columns(df: pd.DataFrame, cols: list[str], name: str = "dataframe") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}. Available: {df.columns.tolist()}")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)