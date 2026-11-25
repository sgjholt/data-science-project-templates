from __future__ import annotations

import subprocess
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from conftest import PROJECT_ROOT, make_cfg

import ingest
import model
import post_processing
import pre_processing
from main import run_pipeline
from tracking import config_to_params, setup_mlflow


def test_ingest_writes_raw_csv(cfg):
    path = ingest.run(cfg)
    df = pd.read_csv(path)
    assert cfg.data.target in df.columns
    assert len(df) > 0


def test_split_is_stratified_and_reproducible(cfg):
    df = pd.read_csv(ingest.run(cfg))
    a = pre_processing.split(df, cfg)
    b = pre_processing.split(df, cfg)
    assert a.X_test.index.equals(b.X_test.index)
    assert len(a.X_test) == pytest.approx(len(df) * cfg.split.test_size, abs=1)
    train_dist = a.y_train.value_counts(normalize=True).sort_index()
    test_dist = a.y_test.value_counts(normalize=True).sort_index()
    assert (train_dist - test_dist).abs().max() < 0.05


@pytest.mark.parametrize("model_name", ["random_forest", "logistic_regression"])
def test_every_model_config_trains(tmp_path, model_name):
    cfg = make_cfg(tmp_path, [f"model={model_name}"])
    splits = pre_processing.run(cfg, ingest.run(cfg))
    fitted = model.run(cfg, splits.X_train, splits.y_train)
    metrics = post_processing.run(fitted, splits.X_test, splits.y_test, tmp_path)
    assert metrics["accuracy"] > 0.8
    assert (tmp_path / "metrics.json").exists()


def test_config_to_params_flattens(cfg):
    params = config_to_params(cfg)
    assert params["model.estimator._target_"].startswith("sklearn.")
    assert params["seed"] == cfg.seed


def test_run_pipeline_logs_to_mlflow(tmp_path):
    cfg = make_cfg(tmp_path, ["model.estimator.n_estimators=10", "mlflow.log_model=true"])
    experiment_id = setup_mlflow(cfg)
    with mlflow.start_run(experiment_id=experiment_id) as active:
        metrics = run_pipeline(cfg, tmp_path)
    logged = mlflow.get_run(active.info.run_id).data.metrics
    assert logged["f1_macro"] == pytest.approx(metrics["f1_macro"])

    loaded = mlflow.sklearn.load_model(f"runs:/{active.info.run_id}/model")
    X_test = pd.read_csv(tmp_path / "processed" / cfg.data.name / "test.csv")
    assert len(loaded.predict(X_test.drop(columns=[cfg.data.target]))) == len(X_test)


def test_cli_end_to_end(tmp_path):
    """Exercise the real entrypoint exactly as a user would."""
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "src/main.py",
            f"hydra.run.dir={run_dir}",
            f"paths.raw_dir={tmp_path / 'raw'}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
            f"mlflow.artifact_dir={tmp_path / 'artifacts'}",
            "model.estimator.n_estimators=10",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ["run.log", "metrics.json", "predictions.csv", ".hydra/config.yaml"]:
        assert (run_dir / name).exists(), name
