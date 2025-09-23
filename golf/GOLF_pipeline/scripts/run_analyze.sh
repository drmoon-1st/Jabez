#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-config/analyze.yaml}"
python3 -m pip install -r requirements.txt >/dev/null 2>&1 || true
python3 -m src.compute_metrics --config "$CFG"
