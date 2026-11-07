#!/usr/bin/env bash
# Gom gói nộp BTC (luật 5.2/6.2) vào artifacts/submission/<tag>/: image đã dựng, mã nguồn ở commit hiện tại,
# sha256 trọng số, mô tả phương pháp và README dựng lại. Không tải lên đâu cả; việc gửi do người nộp làm.
#   scripts/package_submission.sh                 image lifemate:98
#   TAG=lifemate:synth scripts/package_submission.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
TAG=${TAG:-lifemate:98}
ASR=${ASR:-models/phowhisper-large-ft}
LLM=${LLM:-models/qwen3-4b-sft-full}
git diff --quiet HEAD || { echo "còn thay đổi chưa commit: commit trước để mã nguồn khớp image"; exit 1; }
OUT=artifacts/submission/${TAG//[:\/]/_}
mkdir -p "$OUT"
rev=$(git rev-parse --short HEAD)
git archive --format=tar.gz --prefix=lifemate/ -o "$OUT/lifemate-src-$rev.tar.gz" HEAD
cp docs/METHOD.md docs/SUBMISSION.md "$OUT"/
{
  echo "image $TAG, mã nguồn $rev"
  for m in "$ASR" "$LLM"; do (cd "$(readlink -f "$m")" && sha256sum -- *.safetensors | sed "s|^|$(basename "$m")  |"); done
} > "$OUT/MANIFEST.txt"
if command -v docker >/dev/null && docker image inspect "$TAG" >/dev/null 2>&1; then
  docker image inspect "$TAG" --format 'image id {{.Id}}' >> "$OUT/MANIFEST.txt"
  docker save "$TAG" | gzip -1 > "$OUT/${TAG//[:\/]/_}.tar.gz"
else
  echo "không thấy image $TAG: dựng bằng scripts/docker_build.sh rồi chạy lại" | tee -a "$OUT/MANIFEST.txt"
fi
(cd "$OUT" && sha256sum -- *.tar.gz >> MANIFEST.txt)
ls -lh "$OUT"
