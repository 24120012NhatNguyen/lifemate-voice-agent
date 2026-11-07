#!/usr/bin/env bash
# Kiểm tra nấc 6 trên cả public_test: các lượt chạy từ đầu (ASR, khởi động vLLM, agent) phải ra cùng file nộp từng
# byte; rồi đo ASR khi mỗi mẫu chạy một mình và tóm tắt độ trễ, bộ nhớ, tham số. validate chạy trong mỗi lượt.
#   flock <gpu.lock> scripts/docker_check.sh                 container lifemate:98, lệnh docker run tối thiểu
#   MODE=native flock <gpu.lock> scripts/docker_check.sh     docker/predict.sh ngay trên máy, mạng cắt bằng unshare -n
# Biến: IMAGE, DATA (gói dữ liệu, chứa public_test/), OUT (mặc định artifacts/docker_check/<giờ>_<mode>), REF (file
# nộp ngoài Docker để so), RUNS (số lượt, mặc định 2), LATENCY (1: đo ASR từng mẫu, mặc định 1), LIMIT (chạy thử).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
export PYTHONPATH=$HERE/src
MODE=${MODE:-docker}
IMAGE=${IMAGE:-lifemate:98}
DATA=$(readlink -f "${DATA:-$(.venv/bin/python -c 'from lifemate.data import data_root; print(data_root())')}")
OUT=${OUT:-artifacts/docker_check/$(date +%Y%m%d-%H%M%S)_$MODE}
RUNS=${RUNS:-2}
LATENCY=${LATENCY:-1}
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
log() { echo "[$(date '+%F %T')] $*"; }
log "$MODE, data $DATA, ra $OUT"

# Chỉ các cờ bắt buộc (GPU, tắt mạng, hai thư mục gắn): BTC có thể chạy không có --ipc=host hay --shm-size.
drun() { docker run --rm --gpus all --network none -v "$DATA:/data:ro" "$@"; }
nrun() { unshare -rn bash -c 'ip link set lo up && exec "$@"' _ "$@"; }

for i in $(seq "$RUNS"); do
  log "lượt $i/$RUNS"
  mkdir -p "$OUT/run$i"
  if [ "$MODE" = docker ]; then
    drun -v "$OUT/run$i:/result" ${LIMIT:+-e LIMIT=$LIMIT} "$IMAGE"
  else
    DATA=$DATA OUT=$OUT/run$i nrun docker/predict.sh
  fi
  log "lượt $i xong (mã $?)"
done

if [ "$LATENCY" = 1 ]; then
  log "ASR từng mẫu một mình"
  if [ "$MODE" = docker ]; then
    drun -v "$OUT/run1:/result" -e LIFEMATE_DATA=/data -e HF_HUB_OFFLINE=1 --entrypoint /app/.venv/bin/python \
      "$IMAGE" scripts/asr_latency.py --out /result/asr_latency.json
  else
    LIFEMATE_DATA=$DATA HF_HUB_OFFLINE=1 nrun .venv/bin/python scripts/asr_latency.py --out "$OUT/run1/asr_latency.json"
  fi
  .venv/bin/python scripts/docker_report.py "$OUT/run1" --models models/phowhisper-large-ft models/qwen3-4b-sft-full \
    > /dev/null
fi

outs=()
for i in $(seq "$RUNS"); do outs+=("$OUT/run$i"); done
.venv/bin/python scripts/docker_report.py "${outs[@]}" ${REF:+--ref "$REF"} | tee "$OUT/compare.txt"
