#!/usr/bin/env bash
# Dựng image nộp (docker/Dockerfile). Trọng số đi vào qua hai build context riêng, nên models/ có là liên kết
# (sang worktree khác) cũng được: đường dẫn được giải ra thư mục thật trước khi đưa cho Docker.
#   scripts/docker_build.sh                                 lifemate:98, trọng số trong image
#   VARIANT=slim TAG=lifemate:98-slim scripts/docker_build.sh   không trọng số, gắn bằng -v <models>:/app/models:ro
#   LLM=models/qwen3-4b-sft-synth TAG=lifemate:synth scripts/docker_build.sh
#   CONFIG=configs/docker-nbest.yaml TAG=lifemate:98n scripts/docker_build.sh   bản nộp 98 nghe N-best 4
# Ghi id image, cỡ và sha256 trọng số vào artifacts/docker/<tag>.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
TAG=${TAG:-lifemate:98}
VARIANT=${VARIANT:-full}
ASR=${ASR:-models/phowhisper-large-ft}
LLM=${LLM:-models/qwen3-4b-sft-full}
CONFIG=${CONFIG:-configs/docker.yaml}
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
