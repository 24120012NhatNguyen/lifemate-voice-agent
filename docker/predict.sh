#!/usr/bin/env bash
# Điểm vào của image (nấc 6): từ audio tới submission.jsonl, không cần mạng.
#   docker run --rm --gpus all --network none -v <gói dữ liệu>:/data:ro -v <thư mục ra>:/result lifemate:98
# Ba pha, mỗi pha một tiến trình để pha trước trả hết GPU cho pha sau (card 16 GB không chứa cùng lúc cả hai):
#   1. ASR mọi lượt nói của split theo lô, như lúc làm bản nộp public; cache trong thư mục tạm nên luôn nghe lại;
#   2. vLLM phục vụ agent SFT với DETERMINISTIC=1 (kernel bất biến theo batch), chỉ nghe ở 127.0.0.1;
#   3. agent chạy tuần tự từng mẫu qua ToolEnv và luật, rồi chép file nộp và kiểm bằng zaic_tools của split.
# Ra trong OUT: submission.jsonl, runs/ (transcript, vết agent, metrics), predict.log, vllm.log, timing.json,
# gpu_mem.csv, report.json.
# Biến: DATA (thư mục chứa <split>/, hoặc chính thư mục split), OUT, SPLIT (auto: private_test nếu có, không thì
# split duy nhất khác train), CONFIG, LLM_MODEL, ASR_MODEL (đường dẫn tương đối với gốc repo hoặc tuyệt đối, vd.
# trọng số gắn bằng -v), GPU_UTIL, PORT, VLLM_WAIT (giây chờ vLLM lên), LIMIT (chạy thử trên vài mẫu).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
DATA=${DATA:-/data}
OUT=${OUT:-/result}
SPLIT=${SPLIT:-auto}
CONFIG=${CONFIG:-configs/docker.yaml}
LLM_MODEL=${LLM_MODEL:-models/qwen3-4b-sft-full}
ASR_MODEL=${ASR_MODEL:-models/phowhisper-large-ft}
PORT=${PORT:-8000}
VLLM_WAIT=${VLLM_WAIT:-900}
PY=${PY:-.venv/bin/python}
SUBSET=${SUBSET:-all}  # tập con của train khi kiểm (val_app, val_random); split thi luôn là all

# Qwen3.5 chỉ văn bản (config.json: model_type qwen3_5_text) phục vụ khác Qwen3: vLLM chưa có kernel bất biến theo
# batch cho lớp Gated DeltaNet nên không bật DETERMINISTIC; chạy lại ra cùng kết quả nhờ mỗi lúc chỉ một chuỗi
# (--max-num-seqs 1, agent tuần tự). Chat template là khung think rỗng mà model đã học; prefill GDN bằng Triton.
# Tự nhận theo model; ép bằng LLM_FLAVOR=qwen3|qwen35. Đổi chế độ bằng MAX_NUM_SEQS, LLM_DETERMINISTIC, VLLM_EXTRA.
if [ -z "${LLM_FLAVOR:-}" ]; then
  if grep -qs '"model_type": *"qwen3_5_text"' "$LLM_MODEL/config.json"; then LLM_FLAVOR=qwen35; else LLM_FLAVOR=qwen3; fi
fi
if [ "$LLM_FLAVOR" = qwen35 ]; then
  LLM_DETERMINISTIC=${LLM_DETERMINISTIC:-0}
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  export GPU_UTIL=${GPU_UTIL:-0.80}
  export VLLM_EXTRA=${VLLM_EXTRA:-"--gdn-prefill-backend triton --max-num-seqs $MAX_NUM_SEQS --max-num-batched-tokens 8192 --chat-template $HERE/prompts/qwen35_scaffold.jinja"}
else
  LLM_DETERMINISTIC=${LLM_DETERMINISTIC:-1}
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
# Bộ lấy mẫu PyTorch như lúc làm bản nộp public. Ở máy nhà FlashInfer đọc nvcc 12.4 của hệ thống, cho là card không
# hỗ trợ và tự lùi về PyTorch; trong image không có nvcc nên nó tưởng hỗ trợ, rồi biên dịch kernel lúc chạy và hỏng
# vì thiếu ninja (trên A30 cũng vậy).
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$HERE/src${PYTHONPATH:+:$PYTHONPATH}  # chạy ngoài image: dùng đúng mã của thư mục này

mkdir -p "$OUT"
exec > >(tee -a "$OUT/predict.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
now() { date +%s.%N; }

if [ -f "$DATA/$(basename "$DATA").jsonl" ]; then  # gắn thẳng thư mục split
  SPLIT=$(basename "$DATA")
  DATA=$(dirname "$DATA")
fi
if [ "$SPLIT" = auto ]; then
  found=()
  for d in "$DATA"/*/; do
    n=$(basename "$d")
    if [ "$n" != train ] && [ -f "$d$n.jsonl" ]; then found+=("$n"); fi
  done
  if [[ " ${found[*]} " == *" private_test "* ]]; then
    SPLIT=private_test
  elif [ ${#found[@]} -eq 1 ]; then
    SPLIT=${found[0]}
  else
    log "không chọn được split trong $DATA (thấy: ${found[*]:-không có}); đặt biến SPLIT"
    exit 2
  fi
fi
export LIFEMATE_DATA=$DATA
export LIFEMATE_CACHE=${LIFEMATE_CACHE:-$(mktemp -d)}
log "split $SPLIT ($(wc -l < "$DATA/$SPLIT/$SPLIT.jsonl") mẫu) ở $DATA, config $CONFIG"
log "LLM $LLM_MODEL ($LLM_FLAVOR, DETERMINISTIC=$LLM_DETERMINISTIC${VLLM_EXTRA:+, vLLM thêm: $VLLM_EXTRA}), ASR $ASR_MODEL"

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
log "GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo '?'), driver ${drv:-?}"
# torch của vLLM dựng với CUDA 13 (cần driver 580 trở lên). GPU trung tâm dữ liệu (A30) chạy được trên driver cũ hơn
# nhờ thư viện forward compatibility có sẵn trong ảnh nền nvidia/cuda; card gia dụng thì không.
if [ -n "$drv" ] && [ "${drv%%.*}" -lt 580 ] && [ -e /usr/local/cuda/compat/libcuda.so.1 ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  log "driver $drv < 580: dùng /usr/local/cuda/compat"
fi

SMI="" VPID=""
cleanup() {  # không gọi `wait` trơn: nó chờ cả tee ở trên, mà tee chỉ dừng khi script đã thoát
  if [ -n "$VPID" ]; then kill -TERM -- -"$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true; fi
  if [ -n "$SMI" ]; then kill "$SMI" 2>/dev/null || true; fi
  chown -R "$(stat -c %u:%g "$OUT")" "$OUT" 2>/dev/null || true  # container chạy root: trả file cho chủ thư mục ra
}
trap cleanup EXIT
if command -v nvidia-smi >/dev/null; then  # bộ nhớ GPU cả card, mỗi 0,5 giây, để báo cáo đỉnh từng pha
  (while :; do echo "$(now),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 0.5; done) \
    > "$OUT/gpu_mem.csv" 2>/dev/null &
  SMI=$!
fi

SETS=(--set "split=$SPLIT" --set "subset=$SUBSET" --set "asr.model=$ASR_MODEL" --set "agent.llm.model=$LLM_MODEL"
      --set "agent.llm.base_url=http://127.0.0.1:$PORT/v1")
if [ -n "${LIMIT:-}" ]; then SETS+=(--set "limit=$LIMIT"); fi  # chạy thử trên LIMIT mẫu
for kv in ${EXTRA_SET:-}; do SETS+=(--set "$kv"); done        # ghi đè thêm khi kiểm, vd. EXTRA_SET="workers=8"
t0=$(now)
log "1/3 ASR"
"$PY" -m lifemate.cli run "$CONFIG" "${SETS[@]}" --set name=asr --set agent.name=none --runs-dir "$OUT/runs"

t1=$(now)
log "2/3 vLLM"
DETERMINISTIC=$LLM_DETERMINISTIC MODEL=$LLM_MODEL PORT=$PORT setsid scripts/serve_vllm.sh --host 127.0.0.1 > "$OUT/vllm.log" 2>&1 &
VPID=$!
ready=""
for _ in $(seq "$VLLM_WAIT"); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -qF "\"$LLM_MODEL\""; then ready=1; break; fi
  kill -0 "$VPID" 2>/dev/null || break
  sleep 1
done
if [ -z "$ready" ]; then
  log "vLLM không lên, xem $OUT/vllm.log"
  tail -30 "$OUT/vllm.log"
  exit 4
fi

t2=$(now)
log "3/3 agent"
"$PY" -m lifemate.cli run "$CONFIG" "${SETS[@]}" --set name=submit --runs-dir "$OUT/runs"
t3=$(now)
kill -TERM -- -"$VPID" 2>/dev/null || true
wait "$VPID" 2>/dev/null || true
VPID=""

run=$(ls -d "$OUT"/runs/*_submit | tail -1)
cp "$run/predictions.jsonl" "$OUT/submission.jsonl"
printf '{"asr": [%s, %s], "vllm_start": [%s, %s], "agent": [%s, %s]}\n' "$t0" "$t1" "$t1" "$t2" "$t2" "$t3" > "$OUT/timing.json"
log "submission.jsonl: $(wc -l < "$OUT/submission.jsonl") dòng, sha256 $(sha256sum "$OUT/submission.jsonl" | cut -c1-16)"
# tham số: ASR gốc, LLM, và mọi adapter LoRA của ASR có trong image (models/*-lora) khi dùng một gốc + nhiều adapter
# shellcheck disable=SC2046
"$PY" scripts/docker_report.py "$OUT" --models "$ASR_MODEL" "$LLM_MODEL" $(ls -d models/*-lora 2>/dev/null) || log "không dựng được report.json"
log "kiểm bằng zaic_tools"
# submission.jsonl đã ghi xong: bước kiểm chỉ để đọc log, không được làm container thoát mã khác 0
"$PY" -m lifemate.cli validate "$run" || log "CẢNH BÁO: zaic_tools --validate trả mã khác 0"
log "xong"
