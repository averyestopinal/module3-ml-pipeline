#!/usr/bin/env sh
set -e

echo "START: container entrypoint"

# optional: run preprocessing in container (skip if artifacts already present)
if [ "${RUN_ARTIFACTS:-0}" = "1" ] && [ ! -f artifacts/preprocessor.joblib ]; then
  echo "Running preprocessing to create artifacts..."
  python scripts/preprocess.py
fi

if [ "${RUN_ARTIFACTS:-0}" = "1" ] && [ ! -f artifacts/model.joblib ]; then
  echo "Running training to create model..."
  python scripts/train.py
fi

# start server
exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}