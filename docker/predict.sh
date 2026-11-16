#!/usr/bin/env bash
# Kept for older scripts (docker_check.sh): the entry point is now /code/predict.sh (predict.py).
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../predict.sh" "$@"
