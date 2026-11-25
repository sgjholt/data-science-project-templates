"""Stage 4 - post-processing: evaluate the model and write run outputs.

Everything written to `run_dir` is uploaded to MLflow as run artifacts by main.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def run(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, run_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        run_dir / "predictions.csv", index=False
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(classification_report(y_test, y_pred))

    for name, value in metrics.items():
        logger.info("{} = {:.4f}", name, value)
    return metrics
