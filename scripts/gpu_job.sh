#!/usr/bin/env bash
# Việc GPU xếp hàng qua một khoá chung, để nhiều người và nhiều worktree dùng chung card 16 GB mà không giẫm lên nhau.
#   scripts/gpu_job.sh train sft/train.jsonl models/qwen3-4b-sft [đối số thêm cho train_sft.py]
#   SET="limit=100" scripts/gpu_job.sh eval models/qwen3-4b-sft configs/experiments/40_*.yaml ...
#   SET="limit=100" scripts/gpu_job.sh run configs/experiments/21_*.yaml   (config không cần vLLM, vd. chỉ ASR)
# eval bật vLLM với DETERMINISTIC=1 cho đúng mô hình đó, chạy hàng config rồi tắt server.
# GPU_LOCK, GPU_LOG: khoá và nhật ký dùng chung (mặc định logs/gpu.lock, logs/GPU.md). WHO: tên ghi vào nhật ký.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
mkdir -p logs
LOCK=${GPU_LOCK:-logs/gpu.lock}
LOG=${GPU_LOG:-logs/GPU.md}
WHO=${WHO:-$(basename "$HERE")}
note() { echo "- $(date '+%F %H:%M') $WHO: $*" >> "$LOG"; echo "[$(date '+%H:%M:%S')] $*"; }

if [ -z "${_GPU_LOCKED:-}" ]; then
  note "QUEUED on gpu.lock: $*"
  _GPU_LOCKED=1 exec flock "$LOCK" "$0" "$@"
fi

kind=${1:-}; shift || true
vpid=""
trap 'exit 143' TERM INT
trap '[ -n "$vpid" ] && kill -TERM -- -"$vpid" 2>/dev/null; note "RELEASE gpu.lock ($kind rc=${rc:-dừng})"' EXIT

case "$kind" in
  train)
    data=$1 out=$2; shift 2
    note "TAKE gpu.lock, train $out"
    .venv/bin/python scripts/train_sft.py --data "$data" --out "$out" "$@" > "logs/train_$(basename "$out").log" 2>&1
    rc=$?
    ;;
  eval)
    model=$1; shift
    if curl -sf localhost:8000/v1/models >/dev/null; then echo "cổng 8000 đang có server khác"; rc=3; exit $rc; fi
    note "TAKE gpu.lock, vLLM $model + eval"
    DETERMINISTIC=1 MODEL="$model" setsid scripts/serve_vllm.sh > "logs/vllm_$(basename "$model").log" 2>&1 &
    vpid=$!
    for _ in $(seq 120); do
      curl -s localhost:8000/v1/models | grep -qF "\"$model\"" && break
      kill -0 "$vpid" 2>/dev/null || break
      sleep 5
    done
    if curl -s localhost:8000/v1/models | grep -qF "\"$model\""; then scripts/run_queue.sh "$@"; rc=$?; else echo "vLLM không lên"; rc=4; fi
    kill -TERM -- -"$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null; vpid=""; sleep 5
    ;;
  run)
    note "TAKE gpu.lock, run $*"
    scripts/run_queue.sh "$@"
    rc=$?
    ;;
  *)
    echo "dùng: $0 train|eval|run ..."; rc=2
    ;;
esac
exit $rc
