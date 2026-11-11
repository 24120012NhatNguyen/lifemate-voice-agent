#!/usr/bin/env bash
# Chạy LLM qua vLLM với API chuẩn OpenAI ở cổng 8000. Đổi MODEL để thử mô hình khác.
# GPU 16 GB: đủ cho mô hình 4B bf16 với ngữ cảnh 16K. Tắt server trước khi chạy ASR trên GPU.
set -euo pipefail
# DETERMINISTIC=1: kernel bất biến theo batch, để chạy lại ra cùng kết quả (BTC chạy lại Docker của top 2).
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
if [ "${DETERMINISTIC:-0}" = "1" ]; then export VLLM_BATCH_INVARIANT=1; fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
exec "$HERE/.venv-vllm/bin/vllm" serve "$MODEL" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --seed 0 ${VLLM_EXTRA:-} "$@"
