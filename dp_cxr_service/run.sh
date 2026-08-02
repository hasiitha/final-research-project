#!/usr/bin/env bash
# One-command local start.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f dp_cxr_service/bundle/bundle_config.json ]; then
  echo "ERROR: no bundle found at dp_cxr_service/bundle/"
  echo "Unpack dp_cxr_deployment_bundle.zip (notebook section 8.2) there first."
  exit 1
fi

python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r dp_cxr_service/requirements.txt
echo "Starting on http://localhost:8000  (docs at /docs)"
uvicorn dp_cxr_service.app:app --host 0.0.0.0 --port 8000
