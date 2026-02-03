# tests/conftest.py
import yaml
import pytest


@pytest.fixture(scope="session")
def config():
    with open("configs/experiment_config.yaml", "r") as f:
        return yaml.safe_load(f)