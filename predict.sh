#!/usr/bin/env bash
# Entry point of the LifeMate image (Zalo AI Challenge 2026). BTC runs, with no network:
#   docker run --gpus '"device=0"' -v <data>:/data:ro -v <result>:/result IMAGE /bin/bash /code/predict.sh
# Reads /data/<split>/ (private_test if present), writes /result/submission.jsonl ({id, tool_calls, state_hash, time})
# and prints model-load and prediction timings. Logs, vLLM log and the per-sample traces go to /result/predict_logs/.
# All hyper-parameters are fixed in configs/docker-final.yaml and predict.py; nothing needs to be passed.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "preprocessing... output will be saved in ${OUT:-/result}/submission.jsonl"
exec ./.venv-jupyter/bin/python -u predict.py
