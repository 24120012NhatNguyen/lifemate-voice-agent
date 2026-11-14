#!/usr/bin/env bash
# Kiểm tra nấc 6 trên cả public_test: các lượt chạy từ đầu (ASR, khởi động vLLM, agent) phải ra cùng file nộp từng
# byte; rồi đo ASR khi mỗi mẫu chạy một mình và tóm tắt độ trễ, bộ nhớ, tham số. validate chạy trong mỗi lượt.
#   flock <gpu.lock> scripts/docker_check.sh                 container lifemate:98, lệnh docker run tối thiểu
#   MODE=native flock <gpu.lock> scripts/docker_check.sh     docker/predict.sh ngay trên máy, mạng cắt bằng unshare -n
# Biến: IMAGE, DATA (gói dữ liệu, chứa public_test/), OUT (mặc định artifacts/docker_check/<giờ>_<mode>), REF (file
# nộp ngoài Docker để so), RUNS (số lượt, mặc định 2), LATENCY (1: đo ASR từng mẫu, mặc định 1), LIMIT (chạy thử).
# Kiểm trên tập val của train thay cho public_test (LM-50; model 161 không train trên val):
#   IMAGE=lifemate:q35 DATA=<gói có train/> SPLIT=train SUBSET=val_app LIMIT=120 ASR_BEAMS=4 ASR_NBEST=4
#     LLM=models/qwen35-4b-sft-synth flock <gpu.lock> scripts/docker_check.sh
# SPLIT, SUBSET, CONFIG, EXTRA_SET (vd. "workers=8"), MAX_NUM_SEQS, LLM_DETERMINISTIC, LLM_FLAVOR, GPU_UTIL, VLLM_EXTRA
# được chuyển nguyên cho docker/predict.sh. ASR, LLM: thư mục model để đếm tham số trong báo cáo. ASR_BEAMS, ASR_NBEST:
# cách nghe của config đang kiểm, để phép đo ASR từng mẫu dùng đúng cách nghe đó (mặc định 1, 1 như bản 98).
# DOCKER_ARGS: cờ thêm cho docker run, vd. gắn train thật khi DATA chỉ chứa liên kết:
#   DOCKER_ARGS="-v /đường/dẫn/thật/train:/data/train:ro" (liên kết trỏ ra ngoài thư mục gắn thì trong container không tới được).
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
ASR=${ASR:-models/phowhisper-large-ft}
LLM=${LLM:-models/qwen3-4b-sft-full}
PASS=()   # biến chuyển cho predict.sh: -e trong container, export khi chạy native
for v in SPLIT SUBSET LIMIT CONFIG EXTRA_SET MAX_NUM_SEQS LLM_DETERMINISTIC LLM_FLAVOR GPU_UTIL VLLM_EXTRA; do
  if [ -n "${!v:-}" ]; then PASS+=(-e "$v=${!v}"); export "${v?}"; fi
done
LAT=(--split "${SPLIT:-public_test}" --subset "${SUBSET:-all}" ${LIMIT:+--limit "$LIMIT"}
     --num-beams "${ASR_BEAMS:-1}" --n-best "${ASR_NBEST:-1}")
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
log() { echo "[$(date '+%F %T')] $*"; }
log "$MODE, data $DATA, ra $OUT"

# Chỉ các cờ bắt buộc (GPU, tắt mạng, hai thư mục gắn): BTC có thể chạy không có --ipc=host hay --shm-size.
drun() { docker run --rm --gpus all --network none -v "$DATA:/data:ro" ${DOCKER_ARGS:-} "$@"; }
nrun() { unshare -rn bash -c 'ip link set lo up && exec "$@"' _ "$@"; }

for i in $(seq "$RUNS"); do
  log "lượt $i/$RUNS"
  mkdir -p "$OUT/run$i"
  if [ "$MODE" = docker ]; then
    drun -v "$OUT/run$i:/result" "${PASS[@]}" "$IMAGE"
  else
    DATA=$DATA OUT=$OUT/run$i nrun docker/predict.sh
  fi
  log "lượt $i xong (mã $?)"
done

if [ "$LATENCY" = 1 ]; then
  log "ASR từng mẫu một mình"
  if [ "$MODE" = docker ]; then
    drun -v "$OUT/run1:/result" -e LIFEMATE_DATA=/data -e HF_HUB_OFFLINE=1 --entrypoint /app/.venv/bin/python \
      "$IMAGE" scripts/asr_latency.py "${LAT[@]}" --model "$ASR" --out /result/asr_latency.json
  else
    LIFEMATE_DATA=$DATA HF_HUB_OFFLINE=1 nrun .venv/bin/python scripts/asr_latency.py "${LAT[@]}" --model "$ASR" \
      --out "$OUT/run1/asr_latency.json"
  fi
  .venv/bin/python scripts/docker_report.py "$OUT/run1" --models "$ASR" "$LLM" > /dev/null
fi

outs=()
for i in $(seq "$RUNS"); do outs+=("$OUT/run$i"); done
.venv/bin/python scripts/docker_report.py "${outs[@]}" ${REF:+--ref "$REF"} | tee "$OUT/compare.txt"
