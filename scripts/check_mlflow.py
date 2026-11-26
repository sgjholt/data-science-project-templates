"""Smoke-check a generated project's MLflow tracking through a running MLflow server.

Run from the generated project's root with the project's own interpreter:

    uv run python <template>/scripts/check_mlflow.py --expect-runs 3            # starts `mlflow ui`
    uv run python <template>/scripts/check_mlflow.py --url http://localhost:5000 --expect-runs 1

Checks, all via the server's HTTP API (what the web UI itself uses):
  - the UI page is served
  - the project's experiment exists and has exactly --expect-runs FINISHED runs
  - each run has the expected params, metrics and tags
  - each run's artifacts (resolved config, run.log, metrics.json) are listed and downloadable,
    and the downloaded metrics.json matches the logged metrics
  - at least one model was logged; when the server is local, it is loaded back and predicts
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REQUIRED_PARAMS = ["seed", "data.name", "model.name", "split.test_size", "model.estimator._target_"]
REQUIRED_METRICS = ["accuracy", "f1_macro"]
REQUIRED_TAGS = ["run_dir", "overrides"]
REQUIRED_ARTIFACTS = ["run/metrics.json", "run/run.log", "run/.hydra/config.yaml"]


class CheckError(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckError(msg)
    print(f"  ok  {msg}")


def http(url: str, payload: dict | None = None) -> bytes:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def api(base: str, endpoint: str, payload: dict | None = None, **query: str) -> dict:
    url = f"{base}/api/2.0/mlflow/{endpoint}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return json.loads(http(url, payload))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_up(base: str, proc: subprocess.Popen | None, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise CheckError(f"mlflow server exited early with code {proc.returncode}")
        try:
            http(f"{base}/health")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    raise CheckError(f"mlflow server at {base} not healthy after {timeout}s")


def list_artifacts(base: str, run_id: str, path: str = "") -> list[str]:
    """Recursive artifact listing, as file paths relative to the run's artifact root."""
    resp = api(base, "artifacts/list", run_id=run_id, path=path)
    files = []
    for f in resp.get("files", []):
        if f.get("is_dir"):
            files += list_artifacts(base, run_id, f["path"])
        else:
            files.append(f["path"])
    return files


def run_checks(base: str, experiment_name: str, expect_runs: int, load_model: bool) -> None:
    print(f"==> MLflow checks against {base}")
    html = http(f"{base}/").decode(errors="replace").lower()
    check("<html" in html, "UI page is served")

    experiment = api(base, "experiments/get-by-name", experiment_name=experiment_name)["experiment"]
    exp_id = experiment["experiment_id"]
    check(True, f"experiment '{experiment_name}' exists (id={exp_id})")

    runs = api(base, "runs/search", {"experiment_ids": [exp_id], "max_results": 1000}).get("runs", [])
    check(len(runs) == expect_runs, f"{len(runs)} runs logged (expected {expect_runs})")

    for run in runs:
        info, data = run["info"], run["data"]
        run_id, name = info["run_id"], info.get("run_name", "?")
        print(f"  -- run {name} ({run_id})")
        check(info["status"] == "FINISHED", "status FINISHED")

        params = {p["key"]: p["value"] for p in data.get("params", [])}
        missing = [k for k in REQUIRED_PARAMS if k not in params]
        check(not missing, f"params present ({len(params)} logged, missing={missing})")

        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        bad = [k for k in REQUIRED_METRICS if not 0.0 <= metrics.get(k, -1) <= 1.0]
        check(not bad, f"metrics in [0, 1]: { {k: metrics.get(k) for k in REQUIRED_METRICS} }")

        tags = {t["key"]: t["value"] for t in data.get("tags", [])}
        missing = [k for k in REQUIRED_TAGS if k not in tags]
        check(not missing, f"tags present (missing={missing})")

        files = list_artifacts(base, run_id)
        missing = [a for a in REQUIRED_ARTIFACTS if a not in files]
        check(not missing, f"artifacts listed ({len(files)} files, missing={missing})")

        # Same endpoint the UI uses to display/download an artifact.
        query = urllib.parse.urlencode({"run_uuid": run_id, "path": "run/metrics.json"})
        downloaded = json.loads(http(f"{base}/get-artifact?{query}"))
        diff = {k: (downloaded.get(k), metrics.get(k)) for k in REQUIRED_METRICS
                if abs(downloaded.get(k, -1) - metrics.get(k, -2)) > 1e-12}
        check(not diff, "downloaded metrics.json matches logged metrics")

    models = api(base, "logged-models/search", {"experiment_ids": [exp_id]}).get("models", [])
    check(len(models) >= 1, f"{len(models)} logged model(s) registered on the experiment")

    if load_model:
        import mlflow
        import mlflow.sklearn
        import pandas as pd

        mlflow.set_tracking_uri(base)
        source_run = next(m["info"]["source_run_id"] for m in models)
        model = mlflow.sklearn.load_model(f"runs:/{source_run}/model")
        test_csv = next(Path("data/processed").glob("*/test.csv"))
        df = pd.read_csv(test_csv)
        X = df.drop(columns=["target"])
        preds = model.predict(X)
        check(len(preds) == len(df), f"model from run {source_run} loads and predicts {len(preds)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="Use an already-running server instead of starting one")
    parser.add_argument("--expect-runs", type=int, required=True)
    parser.add_argument("--experiment", help="Default: [project].name from pyproject.toml")
    parser.add_argument("--no-model-load", action="store_true",
                        help="Skip loading the model locally (e.g. server runs in a container)")
    args = parser.parse_args()

    experiment = args.experiment or tomllib.loads(Path("pyproject.toml").read_text())["project"]["name"]

    proc = None
    base = args.url
    if base is None:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
               "sqlite:///mlruns/mlflow.db", "--host", "127.0.0.1", "--port", str(port)]
        print("==> starting:", " ".join(cmd))
        proc = subprocess.Popen(cmd, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})
    try:
        wait_until_up(base, proc)
        run_checks(base.rstrip("/"), experiment, args.expect_runs,
                   load_model=not args.no_model_load)
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("==> MLflow checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
