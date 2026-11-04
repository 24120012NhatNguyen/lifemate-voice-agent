#!/usr/bin/env bash
# Chạy lần lượt nhiều config; bỏ qua config đã có lượt chạy hoàn tất trong runs/ (FORCE=1 để chạy lại).
# Bị ngắt giữa chừng thì chạy lại đúng lệnh này: nó nối tiếp từ config chưa xong.
# SET="limit=100 workers=8" ghi đè cho mọi config trong hàng (được lưu vào config.yaml của từng lượt chạy).
#   scripts/run_queue.sh configs/experiments/10_*.yaml configs/experiments/11_*.yaml
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
SETS=()
for kv in ${SET:-}; do SETS+=(--set "$kv"); done
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  if [ -z "${FORCE:-}" ] && ls runs/*_"$name"/metrics.json >/dev/null 2>&1; then
    echo "[$name] đã có kết quả, bỏ qua"
    continue
  fi
  rm -rf runs/*_"$name"  # lượt dở dang không có metrics.json
  .venv/bin/lifemate run "$cfg" "${SETS[@]}" 2>&1 | grep -E --line-buffered "\] (điểm|xong)|Traceback|Error|lần 1: [0-9]+/[0-9]+" || echo "[$name] LỖI"
done
echo "HÀNG ĐỢI XONG"
