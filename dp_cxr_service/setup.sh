#!/usr/bin/env bash
# One-command setup and start for the DP-CXR prediction service.
#
#   bash dp_cxr_service/setup.sh
#
# Checks prerequisites first and stops with a plain-English message rather than
# failing halfway through a two-gigabyte install.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mSTOP\033[0m  %s\n' "$*"; }
info() { printf '        %s\n' "$*"; }

# ── 1 · Python ───────────────────────────────────────────────────────────────
say "1. Checking Python"
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    maj=${v%%.*}; min=${v##*.}
    if [ "$maj" = "3" ] && [ "$min" -ge 9 ] && [ "$min" -le 12 ]; then PY="$c"; break; fi
    LAST="$c ($v)"
  fi
done

if [ -z "$PY" ]; then
  bad "No suitable Python found."
  if [ -n "${LAST:-}" ]; then
    info "Found $LAST, but PyTorch needs Python 3.9-3.12."
    info "Python 3.13 is too new — the wheels do not exist yet."
  fi
  info ""
  info "Install Python 3.12:"
  info "  macOS    https://www.python.org/downloads/release/python-3128/"
  info "           download 'macOS 64-bit universal2 installer', run it,"
  info "           then open a NEW Terminal window and re-run this script."
  info "  Windows  https://www.python.org/downloads/  (tick 'Add Python to PATH')"
  exit 1
fi
ok "$PY $($PY -c 'import sys;print(sys.version.split()[0])')"

# ── 2 · bundle ───────────────────────────────────────────────────────────────
say "2. Checking the model bundle"
if [ ! -f dp_cxr_service/bundle/bundle_config.json ]; then
  bad "No model bundle at dp_cxr_service/bundle/"
  info ""
  info "In Google Drive open  MyDrive/dp_cxr_mv/  and download"
  info "  dp_cxr_deployment_bundle.zip"
  info "Unzip it so these files sit DIRECTLY in dp_cxr_service/bundle/ :"
  info "  bundle_config.json   head_state_dict.pth   thresholds.json"
  info "  text_rule.json       model_card.json       selection_ranking.csv"
  info ""
  info "A nested folder (bundle/deployment_bundle/...) is the usual mistake."
  exit 1
fi
"$PY" dp_cxr_service/preflight.py || { bad "Bundle check failed — see above."; exit 1; }

# ── 3 · virtual environment ──────────────────────────────────────────────────
say "3. Creating an isolated environment"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv || { bad "Could not create .venv"; exit 1; }
  ok "created .venv"
else
  ok ".venv already exists — reusing it"
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install --quiet --upgrade pip

# ── 4 · dependencies ─────────────────────────────────────────────────────────
say "4. Installing dependencies (~2 GB, mostly PyTorch — please wait)"
if ! pip install -r dp_cxr_service/requirements.txt; then
  info ""
  info "Flexible versions failed. Retrying with the locked set..."
  pip install -r dp_cxr_service/requirements-locked.txt || {
    bad "Install failed. Send the last 20 lines above for diagnosis."; exit 1; }
fi
ok "dependencies installed"

# ── 5 · import check ─────────────────────────────────────────────────────────
say "5. Verifying the install"
python - <<'PYCHECK' || { echo "  STOP  imports failed"; exit 1; }
import importlib, sys
for m in ("torch", "torchvision", "transformers", "torchxrayvision",
          "fastapi", "uvicorn", "PIL", "numpy", "matplotlib"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  STOP  {m:16s} {type(e).__name__}: {e}")
        sys.exit(1)
PYCHECK

# ── 6 · start ────────────────────────────────────────────────────────────────
say "6. Starting the service"
info "The first request downloads DenseNet-121 and Bio_ClinicalBERT (~500 MB)"
info "and may take a minute. This happens once."
info ""
info "  Open  http://localhost:8000/docs   to submit a radiograph and report"
info "  Stop  Ctrl-C"
info ""
exec uvicorn dp_cxr_service.app:app --host 127.0.0.1 --port 8000
