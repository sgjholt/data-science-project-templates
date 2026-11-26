#!/usr/bin/env bash
# Render the template in both variants and check each generated project works.
#
#   scripts/test-template.sh            # both variants
#   scripts/test-template.sh false      # no-Docker variant only
#   scripts/test-template.sh true       # Docker variant only (needs a Docker daemon)
#   PYTHON_VERSION=3.15 scripts/test-template.sh   # any copier.yml choice (default: template default)
#
# Uses the working tree (uncommitted changes included).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
# Docker writes root-owned files into bind mounts; don't fail cleanup on them.
trap 'rm -rf "$WORK" 2>/dev/null || true' EXIT
VARIANTS="${*:-false true}"
PY_ARGS=()
[[ -n "${PYTHON_VERSION:-}" ]] && PY_ARGS=(-d python_version="$PYTHON_VERSION")
export MLFLOW_DISABLE_AGENT_HINT=1

for use_docker in $VARIANTS; do
  dst="$WORK/demo_docker_$use_docker"
  echo "==> Rendering use_docker=$use_docker python=${PYTHON_VERSION:-default} -> $dst"
  uvx copier copy --quiet --defaults --overwrite \
    -d project_name="Demo docker $use_docker" -d use_docker="$use_docker" "${PY_ARGS[@]}" \
    "$REPO" "$dst"

  cd "$dst"
  git init -q
  if [[ "$use_docker" == "true" ]]; then
    for f in Dockerfile .dockerignore compose.yaml; do test -f "$f" || { echo "missing $f"; exit 1; }; done
  else
    for f in Dockerfile .dockerignore compose.yaml; do test ! -e "$f" || { echo "unexpected $f"; exit 1; }; done
  fi

  echo "==> uv sync"
  uv sync --quiet
  uv run python -c "import sys; print('python', sys.version.split()[0])"
  echo "==> lint"
  uv run ruff check .
  uv run ruff format --check .
  echo "==> pytest"
  uv run pytest -q
  echo "==> single run"
  uv run src/main.py model.estimator.n_estimators=10
  echo "==> sweep"
  uv run src/main.py -m model=random_forest,logistic_regression mlflow.log_model=false
  test "$(find outputs/multirun -name metrics.json | wc -l)" -eq 2

  if [[ "$use_docker" == "true" ]]; then
    echo "==> docker build + run"
    docker compose build
    docker compose run --rm train model=logistic_regression mlflow.log_model=false
    docker compose down --rmi local --volumes
  fi
  cd "$REPO"
done

echo "==> OK: $VARIANTS"
