#!/usr/bin/env bash
# LifeMate · train một seed Qwen3.5-4B trên máy GPU thuê (vast.ai / RunPod, H100 hoặc A100, image PyTorch có sẵn torch CUDA).
# Cùng runner + trainer + dữ liệu + siêu tham số với notebook Colab (LifeMate_Q35_seed_Colab.ipynb); chỉ khác chỗ chạy.
#
# Dùng:  [GPU=chỉ_số] bash cloud_train.sh JOB [file_checkpoint.zip]
#   Máy nhiều GPU: chạy song song mỗi JOB một GPU, vd. GPU=0 ... ren2s2 ckpt.zip; GPU=1 ... gr0; GPU=2 ... gr1; GPU=3 ... gr2
#   (mỗi JOB thư mục tạm + WORK riêng; cài thư viện dưới khoá chung nên chạy cùng lúc an toàn).
#   JOB: gr0 | gr1 | gr2      (F′ + RenamerV2, gói lifemate_q35_pkg_full_guard_ren2.zip)
#        ren2s1 | ren2s2       (F + RenamerV2,  gói lifemate_q35_pkg_full_ren2.zip)
#   file_checkpoint.zip (tuỳ chọn): qwen35-4b-sft-<NAME>-ckpt.zip lấy từ Drive (WORK của Colab) để chạy tiếp, không học lại.
# Đặt file zip gói cùng thư mục với script này. Ổ: ~20 GB cho 1 JOB, ~30 GB cho 4 JOB (model nền dùng chung; đỉnh lúc xuất ~18 GB).
# Xong: dòng "XONG" và file ./work/<NAME>/qwen35-4b-sft-<NAME>-lora.zip (~120 MB) → tải về PC, đặt vào max-inbox/colab/.
# Bị ngắt: chạy lại đúng lệnh, tự chạy tiếp từ checkpoint trong ./work/<NAME> (lưu mỗi 20 bước).
set -euo pipefail
JOB=${1:?"thiếu JOB (gr0|gr1|gr2|ren2s1|ren2s2)"}; CK=${2:-}
case "$JOB" in
  gr0|gr1|gr2) PKG=lifemate_q35_pkg_full_guard_ren2.zip; DATA=train_full_para_v1_asrfold_val_guard_ren2.jsonl
               SEED=${JOB#gr}; NAME=full-guard-ren2-s$SEED ;;
  ren2s1|ren2s2) PKG=lifemate_q35_pkg_full_ren2.zip; DATA=train_full_para_v1_asrfold_val_ren2.jsonl
               SEED=${JOB#ren2s}; NAME=full-ren2-s$SEED ;;
  *) echo "JOB lạ: $JOB"; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/$PKG" ] || { echo "thiếu $HERE/$PKG"; exit 2; }
WORK=$HERE/work/$NAME; PKGDIR=$HERE/pkg_$JOB; CKIN=$HERE/ckin_$JOB
mkdir -p "$WORK" "$PKGDIR" "$CKIN"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HERE/$PKG" "$PKGDIR"
if [ -n "$CK" ]; then
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$CK" "$CKIN"; echo "chạy tiếp từ checkpoint $CK"
fi
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU   # 1 GPU/JOB: không để runner tự chạy DDP trên mọi GPU của máy
# Cài đúng phiên bản như Colab/Kaggle (run_sft_q35.py PINS), giữ nguyên torch của image; khoá chung cho nhiều JOB cùng máy.
TORCH_V=$(python3 -c "import torch;print(torch.__version__)")
( flock 9; python3 -m pip install -q "transformers==5.20.2" "peft==0.21.2" accelerate "fla-core==0.6.0" "flash-linear-attention==0.6.0" "torch==$TORCH_V" ) 9> /tmp/q35_pip.lock
nvidia-smi -L; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'GPU thấy',torch.cuda.device_count(),torch.cuda.get_device_name(0))"
echo "JOB=$JOB NAME=$NAME SEED=$SEED DATA=$DATA"
cd "$PKGDIR"
# Model nền DÙNG CHUNG cho mọi JOB trên máy: tải + xuất đúng một lần (khoá), xong xoá bản tải → ~8,4 GB thay vì ~18 GB/JOB.
export Q35_SCRATCH=${Q35_SCRATCH:-$HERE/scratch_shared}; mkdir -p "$Q35_SCRATCH"
( flock 8
  if [ ! -f "$Q35_SCRATCH/qwen35-4b-text/export_receipt.json" ]; then
    echo "tải + xuất model nền một lần vào $Q35_SCRATCH/qwen35-4b-text (~10–15 phút)"
    python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import run_sft as r; from pathlib import Path; r.export_base(Path(sys.argv[2]))" "$PKGDIR" "$Q35_SCRATCH/qwen35-4b-text"
  fi ) 8> "$HERE/base.lock"
[ -f "$Q35_SCRATCH/qwen35-4b-text/export_receipt.json" ] || { echo "xuất model nền lỗi"; exit 1; }
WORK=$WORK KAGGLE_INPUT=$CKIN python3 -u run_sft.py "$NAME=$DATA" --deadline 660 --ckpt-every 20 -- --seed "$SEED" 2>&1 | tee -a "$WORK/cloud_stdout.log"
if [ -f "$WORK/qwen35-4b-sft-$NAME-lora.zip" ]; then
  echo "XONG: $WORK/qwen35-4b-sft-$NAME-lora.zip"
else
  echo "CHƯA XONG: chạy lại 'bash cloud_train.sh $JOB' để chạy tiếp từ checkpoint"; exit 1
fi
