"""Entrypoint: one command -> one run -> one folder.

    uv run src/main.py                                   # defaults from conf/config.yaml
    uv run src/main.py model=logistic_regression          # swap a config group
    uv run src/main.py model.estimator.n_estimators=50    # override a single value
    uv run src/main.py -m model=random_forest,logistic_regression seed=0,1,2   # sweep

Each run writes outputs/<date>/<time>/ containing the resolved config (.hydra/),
run.log, metrics and predictions; the same files are logged to MLflow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import skops.io
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import compat  # noqa: F401  (must precede @hydra.main use on Python 3.14)
import ingest
import model
import post_processing
import pre_processing
from logger import setup_logging
from tracking import config_to_params, setup_mlflow


def run_pipeline(cfg: DictConfig, run_dir: Path) -> dict[str, float]:
    """The pipeline itself, independent of Hydra so it can be called from tests/notebooks."""
    raw_path = ingest.run(cfg)
    splits = pre_processing.run(cfg, raw_path)
    fitted = model.run(cfg, splits.X_train, splits.y_train)
    metrics = post_processing.run(fitted, splits.X_test, splits.y_test, run_dir)

    mlflow.log_metrics(metrics)
    if cfg.mlflow.log_model:
        # MLflow serialises sklearn models with skops (no arbitrary-code pickle). We built this
        # object ourselves, so its types are trusted; loaders still refuse anything else.
        trusted = skops.io.get_untrusted_types(data=skops.io.dumps(fitted))
        with warnings.catch_warnings():
            # Integer feature columns trigger a generic "may contain NaN later" hint.
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            mlflow.sklearn.log_model(
                fitted,
                name="model",
                input_example=splits.X_test.head(5),
                skops_trusted_types=trusted,
            )
    return metrics


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    setup_logging(run_dir, level=cfg.log_level)
    logger.info("Run dir: {}", run_dir)
    logger.debug("Config:\n{}", OmegaConf.to_yaml(cfg))

    experiment_id = setup_mlflow(cfg)
    overrides = hydra_cfg.job.override_dirname or "defaults"
    with mlflow.start_run(experiment_id=experiment_id, run_name=overrides) as active:
        mlflow.set_tags({"run_dir": str(run_dir), "overrides": overrides})
        mlflow.log_params(config_to_params(cfg))
        try:
            metrics = run_pipeline(cfg, run_dir)
        except Exception:
            logger.exception("Pipeline failed")
            raise
        finally:
            # Resolved config (.hydra/), run.log, metrics, predictions -> MLflow.
            mlflow.log_artifacts(str(run_dir), artifact_path="run")
        logger.success("MLflow run {} finished", active.info.run_id)

    # Returning a scalar lets Hydra sweepers (e.g. Optuna) optimise it.
    return metrics["f1_macro"]


if __name__ == "__main__":
    main()
