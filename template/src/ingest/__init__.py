"""Stage 1 - ingest: pull raw data from its source into data/raw/.

The example loads a bundled scikit-learn dataset via the Hydra-instantiated
`cfg.data.loader`. Replace `load_raw` with your own source (SQL, API, S3, ...),
keeping the contract: return a DataFrame and write it to data/raw/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig


def load_raw(cfg: DictConfig) -> pd.DataFrame:
    bunch = instantiate(cfg.data.loader)
    df = bunch.frame if hasattr(bunch, "frame") else pd.DataFrame(bunch)
    return df


def run(cfg: DictConfig) -> Path:
    """Fetch the raw dataset and persist it. Returns the raw file path."""
    out = Path(cfg.paths.raw_dir) / f"{cfg.data.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg)
    df.to_csv(out, index=False)
    logger.info("Ingested {} rows x {} cols -> {}", *df.shape, out)
    return out
