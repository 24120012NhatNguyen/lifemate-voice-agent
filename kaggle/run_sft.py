"""Chạy SFT LoRA trên Kaggle: mỗi GPU một thí nghiệm, song song, rồi nén adapter để tải về đánh giá ở máy nhà.

Dataset Kaggle (để riêng tư: dữ liệu BTC chỉ dành cho đội) là thư mục sft/kaggle/ do scripts/kaggle_pack.sh dựng,
gồm file này, train_sft.py và các file jsonl. Notebook bật Accelerator "GPU T4 x2" và Internet, rồi:

    !pip -q uninstall -y torchao
    !pip -q install "transformers==4.57.6" "peft==0.21.2" accelerate
    !python $(find /kaggle/input -name run_sft.py | head -1) drop drop-rename

Gỡ torchao vì ảnh Kaggle có bản 0.10, còn peft 0.21 từ chối mọi bản dưới 0.16 ngay khi tạo LoRA. Đường dẫn dataset
tìm bằng find vì Kaggle có thể gắn nó ở thư mục con. Chạy bằng "Save Version → Save & Run All" để chạy nền tối đa
12 giờ. Lệnh `!` không làm ô lỗi theo mã thoát, nên Kaggle vẫn báo "Successfully ran" khi thí nghiệm lỗi: xem dòng
LỖI hoặc XONG ở cuối Logs.

Mỗi thí nghiệm ghi /kaggle/working/<tên>.log và nén adapter (khoảng 120 MB) thành
/kaggle/working/qwen3-4b-sft-<tên>-lora.zip. Nhiều thí nghiệm hơn số GPU thì chạy thành từng đợt.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK", "/kaggle/working"))
BASE = "Qwen/Qwen3-4B-Instruct-2507"
# tên → (dữ liệu, đối số thêm cho train_sft.py). Đánh giá ở máy nhà bằng configs 44-53, 91-92.
EXPS = {
    "drop": ("train_drop.jsonl", []),                          # gỡ app giữ lại khỏi schema: val_app đo đúng app chưa thấy
    "drop-rename": ("train_drop_rename.jsonl", []),            # như drop, đổi tên 50% mẫu
    "drop-rename-all": ("train_drop_rename_all.jsonl", []),    # như drop, đổi tên mọi mẫu
    "drop-half": ("train_drop.jsonl", ["--epochs", "0.5"]),    # như drop, nửa epoch: ít khớp app đã thấy hơn
    "t4-check": ("train.jsonl", []),                           # lặp lại 40/41 bằng fp16 trên T4 để hiệu chuẩn
    # bản nộp: cả 1.545 mẫu, không đo được trên val nữa (configs 91, 92 chạy public_test)
    "full": ("train_full.jsonl", []),                          # công thức của 40 trên toàn bộ train
    "full-rename-all": ("train_full_rename_all.jsonl", []),    # như full, đổi tên mọi mẫu: dùng nếu 48/49 thắng 44/45
    # máy tính ẩn sys.calc (src/lifemate/numbers.py): dữ liệu dựng bằng config 70, đánh giá bằng configs 70-73, 93
    "calc": ("train_calc.jsonl", []),                          # như 40, chèn bước sys.calc ở 64 mẫu train_rest
    "full-calc": ("train_full_calc.jsonl", []),                # như full, có sys.calc: bản nộp nếu 70-73 thắng
    # app tổng hợp kiểu CRUD (src/lifemate/synth/, build_sft.py --synth 600): đánh giá bằng configs 99_*
    "synth": ("train_synth.jsonl", []),                        # như 40, thêm 600 mẫu app tổng hợp
    "synth-seed1": ("train_synth.jsonl", ["--seed", "1"]),     # seed thứ hai khi hiệu dưới 3 điểm
    "synth-seed2": ("train_synth.jsonl", ["--seed", "2"]),     # seed 2, 3: so được với seed1-3 không synth (80-85)
    "synth-seed3": ("train_synth.jsonl", ["--seed", "3"]),
}


def last_step(log: Path) -> str:
    lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()] if log.exists() else []
    return next((x for x in reversed(lines) if x.startswith("bước") or "Error" in x or "→" in x), lines[-1] if lines else "")


def main(names: list[str]) -> None:
    import torch
    from huggingface_hub import snapshot_download

    bad = [n for n in names if n not in EXPS]
    if not names or bad:
        sys.exit(f"chọn trong: {', '.join(EXPS)}" + (f" (không có {bad})" if bad else ""))
    gpus = torch.cuda.device_count()
    print(f"{gpus} GPU: {', '.join(torch.cuda.get_device_name(i) for i in range(gpus))}", flush=True)
    snapshot_download(BASE)  # tải một lần trước, để các tiến trình con không tải chồng lên nhau
    failed = []
    for wave in range(0, len(names), max(1, gpus)):
        procs = {}
        for gpu, name in enumerate(names[wave:wave + max(1, gpus)]):
            data, extra = EXPS[name]
            log = WORK / f"{name}.log"
            cmd = [sys.executable, str(HERE / "train_sft.py"), "--data", str(HERE / data),
                   "--out", str(WORK / f"qwen3-4b-sft-{name}"), "--base", BASE, "--no-merge", *extra]
            procs[name] = (subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT,
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}), log)
            print(f"GPU {gpu}: {name} ← {data} {' '.join(extra)}", flush=True)
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(60)
            print(time.strftime("%H:%M"), " | ".join(f"{n}: {last_step(log)}" for n, (_, log) in procs.items()), flush=True)
        for name, (p, log) in procs.items():
            adapter = WORK / f"qwen3-4b-sft-{name}-lora"
            if p.returncode == 0 and (adapter / "adapter_model.safetensors").exists():
                shutil.make_archive(str(adapter), "zip", adapter.parent, adapter.name)
                print(f"XONG {name}: {adapter}.zip ({Path(f'{adapter}.zip').stat().st_size / 2**20:.0f} MB)", flush=True)
            else:
                failed.append(name)
                print(f"LỖI {name} (mã {p.returncode}), 30 dòng cuối của {log}:", flush=True)
                print("\n".join(log.read_text(errors="replace").splitlines()[-30:]), flush=True)
    if failed:
        sys.exit(f"lỗi: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
