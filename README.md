# data-science-project-templates

[Copier](https://copier.readthedocs.io) template for config-driven, reproducible data science
projects: **one command, one run, one folder.**

| Tool | Role | In the generated project |
|---|---|---|
| [Hydra](https://hydra.cc) | config | `conf/` groups for data + model, CLI overrides, `-m` sweeps, resolved config saved per run |
| [MLflow](https://mlflow.org) | tracking | every config leaf as a param, metrics, run folder + fitted model as artifacts |
| [Loguru](https://github.com/Delgan/loguru) | logging | console + `run.log` in every run folder; stdlib/MLflow logs routed in |
| [uv](https://docs.astral.sh/uv/) | environment | `pyproject.toml` + `uv.lock`, `uv run` with no venv activation |
| [Docker](https://docs.docker.com) | packaging | *optional*: `Dockerfile` built from `uv.lock`, `compose.yaml` with `train` + `mlflow` UI services |

## Create a project

Requires only [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Default: no Docker
uvx copier copy gh:sgjholt/data-science-project-templates my-project

# With Docker
uvx copier copy -d use_docker=true gh:sgjholt/data-science-project-templates my-project
```

Copier prompts for the remaining answers (`project_name`, `project_slug`, `description`,
`author_name`, `python_version`, `use_docker`). Add `--defaults` to skip prompts.

Python versioning policy: default to **N-1**, one minor behind the newest CPython release
(currently `3.14`; `3.12` and `3.13` selectable).

Then:

```bash
cd my-project
git init && git add -A && git commit -m "Initial commit"
uv sync
uv run src/main.py
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Pull template updates into an existing project

Copier records answers in `.copier-answers.yml`:

```bash
uvx copier update            # merge upstream template changes
uvx copier update -d use_docker=true   # add Docker to an existing project later
```

## What you get

```
my-project/
├── conf/
│   ├── config.yaml              # defaults, seed, paths, split, mlflow, hydra run dirs
│   ├── data/{wine,breast_cancer}.yaml
│   └── model/{random_forest,logistic_regression}.yaml
├── data/{raw,processed}/        # git-ignored
├── notebooks/01_exploration.ipynb   # composes the same config, queries MLflow runs
├── src/
│   ├── ingest/                  # source -> data/raw
│   ├── pre_processing/          # clean + split -> data/processed
│   ├── model/                   # hydra.utils.instantiate(cfg.model.estimator) + fit
│   ├── post_processing/         # metrics.json, predictions.csv, report
│   ├── logger.py  tracking.py  compat.py
│   └── main.py                  # @hydra.main entrypoint, one MLflow run per job
├── outputs/                     # one folder per run (git-ignored)
├── mlruns/                      # MLflow SQLite DB + artifacts (git-ignored)
├── tests/                       # stage tests + end-to-end CLI test
├── Dockerfile  .dockerignore  compose.yaml    # only with use_docker=true
├── pyproject.toml  .python-version
└── README.md                    # usage examples for every tool
```

The example pipeline trains scikit-learn classifiers on bundled datasets (no downloads), so a
fresh project runs end-to-end immediately. Usage examples in the generated README:

```bash
uv run src/main.py model=logistic_regression data=breast_cancer
uv run src/main.py model.estimator.n_estimators=500 split.test_size=0.3
uv run src/main.py -m model=random_forest,logistic_regression seed=0,1,2
docker compose run --rm train model=logistic_regression     # Docker variant
docker compose up mlflow                                    # Docker variant, UI on :5000
```

### Deviations from the reference diagram

- `logger.py` lives in `src/`, not the project root: `uv run src/main.py` puts `src/` on
  `sys.path`, so a root-level module would not be importable without path hacks.
- `src/tracking.py` holds the MLflow setup (tracking URI, experiment, config -> params).
- `src/compat.py`: hydra-core 1.3.8 (latest stable) crashes on Python 3.14 at startup
  (`ValueError: badly formed help string`; 3.14 argparse rejects Hydra's non-`str` help
  object). The shim converts that help object to `str` on 3.14 only. Remove it and the import
  in `main.py` once hydra-core 1.4 is released (1.4.0rc3 fixes it).
- MLflow uses a SQLite backend at `mlruns/mlflow.db`. MLflow 3.x raises on the plain
  file-store backend (`./mlruns`), so `mlflow ui` needs `--backend-store-uri sqlite:///mlruns/mlflow.db`.

## Developing the template

```bash
scripts/test-template.sh          # render both variants; lint, test, run, sweep each
scripts/test-template.sh false    # no-Docker variant only
scripts/test-template.sh true     # Docker variant only (builds and runs the image)
```

CI (`.github/workflows/test-template.yml`) runs the same script for both variants.
