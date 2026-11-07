#!/usr/bin/env bash
# Dựng hai thư mục để tải lên làm dataset Kaggle RIÊNG TƯ cho kaggle/run_asr.py (dữ liệu BTC chỉ dành cho đội):
#   artifacts/kaggle_asr/data  dữ liệu train (audio, train.jsonl, catalog), khoảng 800 MB, tải lên một lần
#   artifacts/kaggle_asr/code  gói lifemate, finetune_whisper.py, run_asr.py, tải lại mỗi khi mã đổi
#   scripts/kaggle_pack_asr.sh <chủ tài khoản Kaggle>
# rồi: kaggle datasets create -p artifacts/kaggle_asr/data --dir-mode zip   (lần sau: datasets version -m "...")
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
OWNER=${1:?chủ tài khoản Kaggle}
OUT=artifacts/kaggle_asr
DATA=$(.venv/bin/python -c "from lifemate.data import data_root; print(data_root())")
rm -rf "$OUT" && mkdir -p "$OUT/data/lifemate/train" "$OUT/code/src" "$OUT/code/scripts"
cp "$DATA/VERSION" "$OUT/data/lifemate/"
cp "$DATA/train/train.jsonl" "$DATA/train/train_tools_catalog.json" "$OUT/data/lifemate/train/"
cp -r "$DATA/train/audio" "$OUT/data/lifemate/train/"
rsync -a --exclude __pycache__ src/lifemate "$OUT/code/src/"
cp scripts/finetune_whisper.py kaggle/run_asr.py "$OUT/code/scripts/"
meta() {  # <thư mục> <slug> <tiêu đề>
  cat > "$1/dataset-metadata.json" <<EOF
{"title": "$3", "id": "$OWNER/$2", "licenses": [{"name": "other"}], "isPrivate": true}
EOF
}
meta "$OUT/data" lifemate-asr-data "lifemate-asr-data"
meta "$OUT/code" lifemate-asr-code "lifemate-asr-code"
du -sh "$OUT"/*
