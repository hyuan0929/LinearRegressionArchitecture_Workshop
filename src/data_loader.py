from pathlib import Path
import pandas as pd
import yaml


def load_experiment_config(config_path: str | Path) -> dict:
    """
    Load experiment configuration from YAML file.
    This supports config-driven experiment architecture.
    """
    config_path = Path(config_path)

    print(f"[ConfigLoader] Loading config from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[ConfigLoader] Config loaded successfully.")
    return config


def load_csv_dataset(dataset_cfg: dict) -> pd.DataFrame:
    """
    Load a CSV dataset based on configuration.
    This is architecture-level logic (not full data validation).
    """

    csv_path = Path(dataset_cfg["path"])

    print(f"[DataLoader] Expecting CSV at: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. "
            f"This is expected at this stage if data is not downloaded yet."
        )

    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Loaded dataset with shape: {df.shape}")

    return df


if __name__ == "__main__":
    print("data_loader module loaded successfully.")

    # Minimal architecture test (no real data required)
    config = load_experiment_config("configs/experiment_config.yaml")

    datasets_cfg = config["data"]["datasets"]

    print("[DataLoader] Available datasets in config:")
    for name, cfg in datasets_cfg.items():
        print(f" - {name}: source_type = {cfg.get('source_type')}")
