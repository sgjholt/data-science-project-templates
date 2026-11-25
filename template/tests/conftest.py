from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"


def make_cfg(tmp_path: Path, overrides: list[str] | None = None) -> DictConfig:
    """Compose the real project config, redirecting every write into tmp_path."""
    base = [
        f"paths.raw_dir={tmp_path / 'raw'}",
        f"paths.processed_dir={tmp_path / 'processed'}",
        f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}",
        f"mlflow.artifact_dir={tmp_path / 'mlruns' / 'artifacts'}",
        "mlflow.log_model=false",
    ]
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=base + (overrides or []))


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    return make_cfg(tmp_path, ["model.estimator.n_estimators=10"])
