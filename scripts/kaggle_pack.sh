#!/usr/bin/env bash
# Dựng sft/kaggle/ để tải lên làm dataset Kaggle RIÊNG TƯ (dữ liệu BTC chỉ dành cho đội, không chia sẻ công khai).
# Cần đủ các file jsonl trong sft/ trước (scripts/build_sft.py, xem docs/BASELINE.md mục 7).
#   scripts/kaggle_pack.sh                                        bộ mặc định vào sft/kaggle
#   scripts/kaggle_pack.sh sft/kaggle_calc train_calc train_full_calc   gói nhỏ chỉ có các file kể ra
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
OUT=${1:-sft/kaggle}
shift || true
FILES=("$@")
[ ${#FILES[@]} -gt 0 ] || FILES=(train train_drop train_drop_rename train_drop_rename_all train_full train_full_rename_all)
rm -rf "$OUT" && mkdir -p "$OUT"
cp scripts/train_sft.py kaggle/run_sft.py "$OUT"/
for f in "${FILES[@]}"; do cp "sft/$f.jsonl" "$OUT"/; done
du -sh "$OUT"/*
