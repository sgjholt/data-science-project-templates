"""Stage 2 - pre-processing: clean raw data and split it into train/test sets.

Keep fitted transforms (scalers, encoders) out of this stage - they belong inside
the model Pipeline so they are fit on training data only (no leakage) and are
saved together with the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().dropna()
    if dropped := before - len(df):
        logger.warning("Dropped {} duplicate/NaN rows", dropped)
    return df


def split(df: pd.DataFrame, cfg: DictConfig) -> Splits:
    target = cfg.data.target
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
        stratify=y if cfg.split.stratify else None,
    )
    return Splits(X_train, X_test, y_train, y_test)


def run(cfg: DictConfig, raw_path: Path) -> Splits:
    df = clean(pd.read_csv(raw_path))
    splits = split(df, cfg)

    out_dir = Path(cfg.paths.processed_dir) / cfg.data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.X_train.assign(**{cfg.data.target: splits.y_train}).to_csv(
        out_dir / "train.csv", index=False
    )
    splits.X_test.assign(**{cfg.data.target: splits.y_test}).to_csv(
        out_dir / "test.csv", index=False
    )
    logger.info("Split train={} test={} -> {}", len(splits.X_train), len(splits.X_test), out_dir)
    return splits
