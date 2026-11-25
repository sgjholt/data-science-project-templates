"""MLflow helpers: point at the configured backend and flatten Hydra configs into params."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf


def setup_mlflow(cfg: DictConfig) -> str:
    """Set tracking URI + experiment. Returns the experiment id."""
    uri: str = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)

    name = cfg.mlflow.experiment_name
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    artifact_dir = Path(cfg.mlflow.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(name, artifact_location=artifact_dir.as_uri())


def flatten(d: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in d.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from flatten(value, f"{full_key}.")
        else:
            yield full_key, value


def config_to_params(cfg: DictConfig) -> dict[str, Any]:
    """{'model.estimator.n_estimators': 200, ...} - one MLflow param per config leaf."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatten(container))
