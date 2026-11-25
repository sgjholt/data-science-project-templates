"""Stage 3 - model: build and fit the estimator described by `cfg.model`.

The estimator class and its hyperparameters live in conf/model/*.yaml and are
created with `hydra.utils.instantiate`, so adding a new model is a new YAML file,
not a code change.
"""

from __future__ import annotations

import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(cfg: DictConfig) -> Pipeline:
    steps = []
    if cfg.model.scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", instantiate(cfg.model.estimator)))
    return Pipeline(steps)


def run(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = build(cfg)
    logger.info("Fitting {}", pipeline)
    pipeline.fit(X_train, y_train)
    return pipeline
