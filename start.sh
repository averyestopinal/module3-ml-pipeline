#!/usr/bin/env sh
set -e

echo "START: container entrypoint"

# optional: run preprocessing in container (skip if artifacts already present)
if [ ! -f artifacts/preprocessor.joblib ]; then
  echo "Running preprocessing to create artifacts..."
  python scripts/preprocess.py
fi

# start server
exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}