#!/usr/bin/env bash
# No Docker, GPU, model loading or dataset access. Safetensors headers only.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY:-.venv/bin/python}" scripts/docker_final.py check "$@"
