#!/usr/bin/env bash
# Dựng ứng viên cuối sau scripts/docker_swap.sh; mặc định CONFIG=configs/docker-final.yaml, TAG=lifemate:final.
# Đổi model (cùng kiến trúc Qwen3.5-4B đã gộp) = hai lệnh, không sửa tay file nào:
#   scripts/docker_swap.sh /đường/dẫn/qwen35-4b-soup-XXX && TAG=lifemate:XXX scripts/docker_build.sh
# Với bản cuối, CPU checker quyết định đúng bốn build context từ receipt; không copy cả models/.
# Legacy: CONFIG=configs/docker.yaml ASR=... LLM=... TAG=lifemate:98 scripts/docker_build.sh
# Ghi image id, cỡ và full sha256 trọng số vào artifacts/docker/<tag>.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
# Final candidate defaults are read from its CPU-checked config and exact context inventory.
# Set CONFIG to a legacy config to retain the old explicit ASR/LLM/ADAPTERS path.
CONFIG=${CONFIG:-configs/docker-final.yaml}
if [ "$(readlink -f "$CONFIG")" = "$HERE/configs/docker-final.yaml" ]; then
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
  final_env=$("${PY:-.venv/bin/python}" scripts/docker_final.py build-env)
  eval "$final_env"  # only fixed keys and shlex.quote values emitted by our CPU checker
fi
TAG=${TAG:-lifemate:final}
VARIANT=${VARIANT:-full}
ASR=${ASR:-models/phowhisper-large-ft}
LLM=${LLM:-models/qwen3-4b-sft-full}
args=(-f docker/Dockerfile --build-arg "VARIANT=$VARIANT" --build-arg "CONFIG=$CONFIG" -t "$TAG")
ADAPTERS=${ADAPTERS:-}   # hai thư mục adapter LoRA của ASR (VARIANT=full2), vd. "models/a-lora models/b-lora"
if [ -n "$ADAPTERS" ]; then
  read -r AD1 AD2 extra <<< "$ADAPTERS"
  if [ -z "${AD2:-}" ] || [ -n "${extra:-}" ]; then echo "ADAPTERS cần đúng hai thư mục"; exit 2; fi
  VARIANT=full2
  [ -f "${APP_RANK:-}" ] || { echo "VARIANT=full2 cần APP_RANK=<đường dẫn app_rank_train.json>"; exit 2; }
  ar=$(mktemp -d)
  cp "$APP_RANK" "$ar/app_rank_train.json"
  args=(-f docker/Dockerfile --build-arg "VARIANT=$VARIANT" --build-arg "CONFIG=$CONFIG" -t "$TAG"
        --build-context "apprank=$ar"
        --build-context "adapter1=$(readlink -f "$AD1")" --build-context "adapter2=$(readlink -f "$AD2")"
        --build-arg "ADAPTER1_DIR=$(basename "$AD1")" --build-arg "ADAPTER2_DIR=$(basename "$AD2")")
fi
if [ "$VARIANT" = full ] || [ "$VARIANT" = full2 ]; then
  args+=(--build-context "asr=$(readlink -f "$ASR")" --build-context "llm=$(readlink -f "$LLM")"
         --build-arg "ASR_DIR=$(basename "$ASR")" --build-arg "LLM_DIR=$(basename "$LLM")")
fi
# Dữ liệu SFT của LLM vào /code/training_code/data (BTC: training_code/ phải có dữ liệu train). Thư mục chỉ chứa đúng
# các file trong training_code/DATA_SHA256SUMS; kiểm hash trước khi dựng.
TRAINING_DATA=${TRAINING_DATA:-artifacts/btc_training_data}
(cd "$TRAINING_DATA" && sha256sum --quiet -c "$HERE/training_code/DATA_SHA256SUMS") || { echo "dữ liệu train sai hash: $TRAINING_DATA"; exit 2; }
extra=$(LC_ALL=C comm -13 <(awk '{print $2}' training_code/DATA_SHA256SUMS | LC_ALL=C sort) <(ls "$TRAINING_DATA" | LC_ALL=C sort))
[ -z "$extra" ] || { echo "file lạ trong $TRAINING_DATA: $extra"; exit 2; }
args+=(--build-context "trainingdata=$(readlink -f "$TRAINING_DATA")")
DOCKER_BUILDKIT=1 docker build "${args[@]}" .
mkdir -p artifacts/docker
info=artifacts/docker/${TAG//[:\/]/_}.txt
{
  echo "tag $TAG ($VARIANT, config $CONFIG), git $(git rev-parse --short HEAD)$(git diff --quiet || echo '+sửa chưa commit'), $(date '+%F %T')"
  docker image inspect "$TAG" --format 'id {{.Id}}  size {{.Size}}'
  if [ "$VARIANT" = full ] || [ "$VARIANT" = full2 ]; then
    for m in "$ASR" "$LLM" $ADAPTERS; do (cd "$(readlink -f "$m")" && sha256sum -- *.safetensors | sed "s|^|$(basename "$m")  |"); done
  fi
} | tee "$info"
