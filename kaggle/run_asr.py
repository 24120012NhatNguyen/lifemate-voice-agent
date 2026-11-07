"""Fine-tune PhoWhisper-large trên Kaggle "GPU T4 x2": mỗi GPU một lượt, song song, rồi nén adapter để tải về gộp ở
máy nhà (`scripts/finetune_whisper.py --merge`). Đánh giá đầu-cuối luôn chạy ở máy nhà.

Hai dataset Kaggle RIÊNG TƯ (dữ liệu BTC chỉ dành cho đội), dựng bằng scripts/kaggle_pack_asr.sh:
- dữ liệu: lifemate/ gồm VERSION, train/train.jsonl, train/train_tools_catalog.json, train/audio/
- mã: src/lifemate, scripts/finetune_whisper.py và file này

Kernel là file này (kiểu script), bật GPU T4 x2 và Internet (tải vinai/PhoWhisper-large):
    kaggle kernels push -p <thư mục có kernel-metadata.json> --accelerator NvidiaTeslaT4
Tham số dòng lệnh không truyền được qua kernel, nên chọn thí nghiệm bằng biến EXPS_RUN dưới đây; SMOKE=True chạy thử
vài chục câu, 1 epoch (khoảng 15 phút, chủ yếu là tải mô hình) trước khi chạy thật.

T4 không có bf16 nên học fp16 (GradScaler), chậm khoảng gấp đôi RTX 5060 Ti. Mỗi lượt ghi /kaggle/working/<tên>.log
và /kaggle/working/phowhisper-large-<tên>-adapters.zip (adapter tốt nhất theo dev, hoặc mọi epoch với --data all).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/kaggle/working"))
BASE = "vinai/PhoWhisper-large"
RECIPE = ["--epochs", "6", "--speed", "0.9,1.0,1.1", "--hint", "profile_history_apps", "--hint-prob", "0.5"]
# tên → đối số thêm cho finetune_whisper.py. Đánh giá ở máy nhà bằng configs 29_asr_phowhisper-large-ft2*, 54g-55h.
EXPS = {
    "ft2": RECIPE,                                              # train_rest, chọn epoch theo dev
    "ft2-all": RECIPE + ["--data", "all", "--save-every-epoch"],  # cả 1.545 mẫu cho bài nộp; chọn epoch theo ft2
}
EXPS_RUN = ["ft2", "ft2-all"]
SMOKE = False


def find(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        sys.exit(f"không thấy {name} trong /kaggle/input")
    return hits[0]


def last_line(log: Path) -> str:
    lines = [x for x in log.read_text(errors="replace").splitlines() if x.strip()] if log.exists() else []
    keep = ("epoch", "Error", "tốt nhất", "không epoch")
    return next((x.strip() for x in reversed(lines) if x.lstrip().startswith(keep) or "Error" in x), lines[-1] if lines else "")


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==4.57.6", "peft==0.21.2", "jiwer",
                    "accelerate"], check=True)
    import torch
    from huggingface_hub import snapshot_download

    script = find("finetune_whisper.py")
    src = find("lifemate/__init__.py").parent.parent  # thư mục dữ liệu cũng tên lifemate, nên tìm theo __init__.py
    data = find("VERSION").parent
    gpus = torch.cuda.device_count()
    print(f"{gpus} GPU: {', '.join(torch.cuda.get_device_name(i) for i in range(gpus))}; dữ liệu {data}; mã {src}",
          flush=True)
    snapshot_download(BASE)  # tải một lần trước, để các tiến trình con không tải chồng lên nhau
    env = {**os.environ, "LIFEMATE_DATA": str(data), "PYTHONPATH": str(src), "PYTHONUNBUFFERED": "1"}
    failed = []
    for wave in range(0, len(EXPS_RUN), max(1, gpus)):
        procs = {}
        for gpu, name in enumerate(EXPS_RUN[wave:wave + max(1, gpus)]):
            extra = EXPS[name] + (["--limit", "48", "--epochs", "1"] if SMOKE else [])
            log = WORK / f"{name}.log"
            cmd = [sys.executable, str(script), "--model", BASE, "--out", str(WORK / f"phowhisper-large-{name}"),
                   "--dtype", "float16", "--no-merge", "--workers", "2", *extra]
            procs[name] = (subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, cwd=WORK,
                                            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}), log)
            print(f"GPU {gpu}: {name} ← {' '.join(extra)}", flush=True)
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(30 if SMOKE else 300)
            print(time.strftime("%H:%M"), " | ".join(f"{n}: {last_line(log)}" for n, (_, log) in procs.items()), flush=True)
        for name, (p, log) in procs.items():
            out = WORK / f"phowhisper-large-{name}"
            adapters = sorted(out.glob("adapter*/adapter_model.safetensors"))
            if p.returncode == 0 and adapters:
                shutil.make_archive(str(out) + "-adapters", "zip", out.parent, out.name)
                size = Path(f"{out}-adapters.zip").stat().st_size / 2**20
                print(f"XONG {name}: {out}-adapters.zip ({size:.0f} MB, {len(adapters)} adapter)", flush=True)
            else:
                failed.append(name)
                print(f"LỖI {name} (mã {p.returncode}), 40 dòng cuối của {log}:", flush=True)
                print("\n".join(log.read_text(errors="replace").splitlines()[-40:]), flush=True)
    if failed:
        sys.exit(f"lỗi: {', '.join(failed)}")


if __name__ == "__main__":
    main()
