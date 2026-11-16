"""Chạy SFT LoRA cho Qwen3.5-4B CHỈ VĂN BẢN trên Kaggle (T4 ×2): cài đúng phiên bản thư viện, dựng model nền chỉ văn
bản ngay trong kernel, train một thí nghiệm trên cả hai GPU (torchrun), lưu checkpoint để chạy tiếp ở phiên sau, rồi
nén adapter để tải về gộp và đánh giá ở máy nhà (LM-50).

Gói Kaggle là dataset RIÊNG TƯ (dữ liệu BTC chỉ dành cho đội) do scripts/kaggle_q35.sh dựng, một thư mục phẳng:

    run_sft.py              file này, đổi tên để tools-pro/kpush.sh gọi được như gói Qwen3
    train_sft.py            scripts/train_sft_q35r.py
    export_text_only.py     scripts/export_text_only.py
    qwen35_scaffold.jinja   prompts/qwen35_scaffold.jinja (trainer so mã hoá của nó với template phục vụ)
    <dữ liệu>.jsonl
    ckpt__<tên>__<file>     tuỳ chọn: checkpoint của phiên trước, dạng file phẳng

    python run_sft.py --ddp synth=train_synth.jsonl                   # một thí nghiệm trên cả 2 GPU
    python run_sft.py --ddp --smoke synth=train_synth.jsonl           # 4 bước: dừng sau bước 2, chạy tiếp, nén adapter
    python run_sft.py --ddp a=train_a.jsonl -- --epochs 2 --seed 1    # sau "--" là đối số thêm cho train_sft.py

Mỗi thí nghiệm viết TÊN=FILE (chỉ TÊN thì dữ liệu là train_<TÊN>.jsonl, gạch nối đổi thành gạch dưới).

Các bước:
1. In môi trường (Python, torch, GPU, đĩa, RAM), cài transformers 5.20.2, peft 0.21.2, fla-core và
   flash-linear-attention 0.6.0 (kernel Triton của lớp Gated DeltaNet; thiếu thì Transformers tự lùi về bản PyTorch
   tham chiếu, đúng nhưng chậm gấp đôi). Giữ torch của ảnh Kaggle.
2. Model nền: nếu input có sẵn thư mục qwen35-4b-text (dataset riêng tư) thì dùng; không thì tải Qwen/Qwen3.5-4B đúng
   revision đã ghim (9,3 GB, công khai) và xuất bản chỉ văn bản bằng export_text_only.py (4.205.751.296 tham số, bỏ
   vision và MTP) vào đĩa tạm, không vào /kaggle/working.
3. Train: T4 không có bf16 thật nên train_sft.py tự chọn fp16 + GradScaler; --ddp chạy torchrun 2 rank, gradient của
   một bước tối ưu bằng đúng một GPU chạy cả cửa sổ. Checkpoint mỗi --ckpt-every phút và khi sắp quá --deadline.
4. Xong: /kaggle/working/qwen35-4b-sft-<tên>-lora.zip (adapter, ~120 MB). Chưa xong: dòng "CHƯA XONG <tên>" và
   /kaggle/working/qwen35-4b-sft-<tên>-ckpt.zip chứa các file phẳng ckpt__<tên>__*; phiên sau đưa các file đó vào
   dataset (scripts/kaggle_q35.sh resume) và chạy lại đúng lệnh. Số GPU hai phiên có thể khác nhau.

Mỗi thí nghiệm ghi /kaggle/working/<tên>.log; tóm tắt cả phiên (phiên bản, thời gian từng bước, tốc độ, bộ nhớ đỉnh
từng GPU, loss) ở /kaggle/working/q35_summary.log (JSON). Kaggle báo "Successfully ran" cả khi thí nghiệm lỗi: xem dòng
XONG, CHƯA XONG hoặc LỖI ở cuối log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # trùng scripts/export_text_only.py và model 161 ở máy nhà
SOURCE_FILES = ["*.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "vocab.json", "merges.txt", "LICENSE"]
# Phiên bản đã chạy ở máy nhà (RTX 5060 Ti): model 161 và 169. torch giữ của ảnh Kaggle.
PINS = ["transformers==5.20.2", "peft==0.21.2", "accelerate", "fla-core==0.6.0", "flash-linear-attention==0.6.0"]
PREFIX = "qwen35-4b-sft-"
SECTION = "##### LƯỢT "  # đầu mỗi lượt train_sft.py trong <tên>.log (torchrun tự in các hàng "=====" khi một rank thoát mã 3)
T0 = time.time()
SUMMARY: dict = {"stages": {}, "experiments": {}}
RUNNING: list = []  # tiến trình train đang chạy (mỗi cái một nhóm tiến trình riêng), để giết khi cả phiên hết giờ


def stage(name: str, t: float) -> None:
    SUMMARY["stages"][name] = round(time.time() - t, 1)
    print(f"[{(time.time() - T0) / 60:5.1f} phút] {name}: {time.time() - t:.0f} giây", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **kw)


def env_report() -> None:
    """Thứ cần biết khi kernel lỗi mà không vào được máy: GPU, driver, đĩa, RAM, các gói có sẵn."""
    print(f"python {sys.version.split()[0]}, {os.cpu_count()} CPU", flush=True)
    for cmd in (["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                ["free", "-g"]):
        try:
            print(sh(cmd).stdout.strip(), flush=True)
        except OSError as e:
            print(f"{cmd[0]}: {e}", flush=True)
    for p in sorted({str(WORK), "/kaggle/temp", "/tmp", str(Path.home())}):
        if Path(p).exists():
            u = shutil.disk_usage(p)
            print(f"đĩa {p}: trống {u.free / 2**30:.0f} / {u.total / 2**30:.0f} GB", flush=True)


def versions() -> dict:
    """Phiên bản thư viện và việc Transformers có thấy kernel FLA không, hỏi trong tiến trình con (tiến trình này
    không nạp torch trước khi cài xong)."""
    code = r'''
import importlib.metadata as m, importlib.util as u, json
out = {}
for p in ["torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "triton",
          "fla-core", "flash-linear-attention", "numpy"]:
    try:
        out[p] = m.version(p)
    except Exception:
        out[p] = None
import torch
out["cuda_build"] = torch.version.cuda
out["gpus"] = [f"{torch.cuda.get_device_name(i)} sm{''.join(map(str, torch.cuda.get_device_capability(i)))}"
               for i in range(torch.cuda.device_count())]
out["fla_importable"] = bool(u.find_spec("fla"))
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    out["fla_gated_delta_rule"] = True
except Exception as e:
    out["fla_gated_delta_rule"] = f"{type(e).__name__}: {e}"
try:  # kernel SDPA nào dùng được cho attention của Qwen3.5 (16 đầu, 4 đầu K/V, head_dim 256) trên GPU này
    if torch.cuda.is_available():
        import torch.backends.cuda as bc
        dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        q = torch.zeros(1, 16, 8, 256, device="cuda", dtype=dt)
        k = torch.zeros(1, 4, 8, 256, device="cuda", dtype=dt)
        def can(kv, gqa):
            p = bc.SDPAParams(q, kv, kv, None, 0.0, True, gqa)
            return {"flash": bool(bc.can_use_flash_attention(p)), "efficient": bool(bc.can_use_efficient_attention(p))}
        out["sdpa_enable_gqa"] = can(k, True)
        out["sdpa_repeat_kv"] = can(k.repeat_interleave(4, 1), False)
except Exception as e:
    out["sdpa_probe"] = f"{type(e).__name__}: {e}"
print("VERSIONS " + json.dumps(out))
'''
    r = sh([sys.executable, "-c", code])
    line = next((x for x in r.stdout.splitlines() if x.startswith("VERSIONS ")), None)
    if line is None:
        print(r.stdout[-3000:], flush=True)
        raise RuntimeError("không đọc được phiên bản thư viện")
    v = json.loads(line[len("VERSIONS "):])
    print("phiên bản: " + ", ".join(f"{k} {v[k]}" for k in v), flush=True)
    return v


def pip_install(pins: list[str], constraints: Path) -> bool:
    r = sh([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints), *pins])
    print(r.stdout[-4000:], flush=True)
    return r.returncode == 0


def install(a) -> None:
    """Cài PINS mà không cho pip đổi torch của ảnh Kaggle (bánh xe torch khác có thể không hợp driver của T4). Nếu bộ
    đủ không cài được thì cài bộ không có FLA: Transformers lùi về kernel tham chiếu, chậm hơn nhưng vẫn train được."""
    t = time.time()
    if a.no_install:
        print("--no-install: dùng thư viện có sẵn", flush=True)
    else:
        # ảnh Kaggle có torchao cũ: peft 0.21 từ chối mọi bản dưới 0.16 ngay khi tạo LoRA
        subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"], check=False)
        r = sh([sys.executable, "-c", "import importlib.metadata as m; print('TORCH', m.version('torch'))"])
        cur = next((x.split()[1] for x in r.stdout.splitlines() if x.startswith("TORCH ")), None)
        if cur is None:
            raise RuntimeError("ảnh không có torch: " + r.stdout[-500:])
        constraints = Path(os.environ.get("TMPDIR", "/tmp")) / "q35_constraints.txt"
        constraints.write_text(f"torch=={cur.split('+')[0]}\n", encoding="utf-8")
        lean = [p for p in PINS if not p.startswith(("fla-core", "flash-linear-attention"))]
        pins = lean if a.no_fla else PINS
        print(f"giữ torch {cur}; cài {' '.join(pins)}", flush=True)
        if a.no_fla:
            SUMMARY["fla_dropped"] = "--no-fla"
        if not pip_install(pins, constraints):
            if a.no_fla:
                raise RuntimeError("pip install lỗi")
            print(f"CẢNH BÁO: không cài được bộ đủ với torch {cur}; thử không có FLA (kernel tham chiếu, chậm): {' '.join(lean)}", flush=True)
            SUMMARY["fla_dropped"] = "pip không cài được FLA"
            if not pip_install(lean, constraints):
                raise RuntimeError("pip install lỗi cả khi bỏ FLA")
    SUMMARY["versions"] = versions()
    stage("cài thư viện", t)


def scratch_dir() -> Path:
    """Đĩa tạm còn trống nhiều nhất: model nền 8,4 GB và bản tải 9,3 GB không được nằm trong /kaggle/working (20 GB,
    lưu thành output)."""
    if os.environ.get("Q35_SCRATCH"):
        return Path(os.environ["Q35_SCRATCH"])
    cands = [p for p in (Path("/kaggle/temp"), Path("/tmp"), Path.home() / ".cache") if p.exists() or p.parent.exists()]
    for p in cands:
        p.mkdir(parents=True, exist_ok=True)
    return max(cands, key=lambda p: shutil.disk_usage(p).free) / "lifemate-q35"


def prepare_base(a) -> Path:
    """Thư mục model nền chỉ văn bản: --base, hoặc thư mục qwen35-4b-text gắn sẵn trong input, hoặc tải rồi xuất."""
    t = time.time()
    if a.base:
        base = Path(a.base)
        how = "--base"
    else:
        hits = [p.parent for p in INPUT.rglob("export_receipt.json") if (p.parent / "config.json").exists()] if INPUT.exists() else []
        if hits:
            base, how = hits[0], "dataset gắn sẵn"
        else:
            base, how = scratch_dir() / "qwen35-4b-text", "tải + xuất trong kernel"
            if not (base / "export_receipt.json").exists():
                export_base(base)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "qwen3_5_text":
        raise RuntimeError(f"{base} không phải bản Qwen3.5 chỉ văn bản (model_type {cfg.get('model_type')})")
    receipt = base / "export_receipt.json"
    if receipt.exists():
        r = json.loads(receipt.read_text(encoding="utf-8"))
        SUMMARY["base"] = {"how": how, "revision": r.get("source_revision"), "parameters": r.get("exported_parameters"),
                           "sha256": {k: v["sha256"] for k, v in r.get("output_files", {}).items() if k.endswith(".safetensors")}}
        print(f"model nền ({how}): {base}, revision {r.get('source_revision')}, {r.get('exported_parameters'):,} tham số", flush=True)
        for k, v in SUMMARY["base"]["sha256"].items():
            print(f"  sha256 {k} {v}", flush=True)
    else:
        SUMMARY["base"] = {"how": how}
        print(f"model nền ({how}): {base}", flush=True)
    stage("model nền", t)
    return base


def export_base(base: Path) -> None:
    from huggingface_hub import snapshot_download

    src = base.parent / "src" / REVISION  # export_text_only đòi thư mục mang đúng tên revision
    t = time.time()
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(src), allow_patterns=SOURCE_FILES)
    size = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
    stage(f"tải {MODEL_ID} ({size / 2**30:.1f} GB)", t)
    t = time.time()
    sys.path.insert(0, str(HERE))
    import export_text_only

    if export_text_only.REVISION != REVISION:
        raise RuntimeError("revision trong run_sft.py và export_text_only.py lệch nhau")
    receipt = export_text_only.export(src, base)
    print(f"đã xuất {receipt['exported_parameters']:,} tham số, bỏ {receipt['removed_parameters']:,} (vision, MTP)", flush=True)
    for f in src.glob("*.safetensors"):  # bản tải không còn cần: trả 9,3 GB đĩa tạm
        f.unlink()
    stage("xuất bản chỉ văn bản", t)


def parse_exp(spec: str) -> tuple[str, Path]:
    name, _, data = spec.partition("=")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise SystemExit(f"tên thí nghiệm không hợp lệ: {name!r}")
    path = HERE / (data or f"train_{name.replace('-', '_')}.jsonl")
    if not path.is_file():
        raise SystemExit(f"{name}: không thấy dữ liệu {path.name} trong gói ({', '.join(sorted(p.name for p in HERE.glob('*.jsonl'))) or 'không có jsonl'})")
    return name, path


def restore_flat_checkpoint(name: str) -> None:
    """ckpt__<tên>__<file> trong input (checkpoint phiên trước, tải lên dạng file phẳng) → WORK/<adapter>/checkpoint/."""
    flat = [p for p in INPUT.rglob(f"ckpt__{name}__*") if p.is_file()] if INPUT.exists() else []
    if not flat:
        return
    d = WORK / f"{PREFIX}{name}-lora" / "checkpoint"
    if (d / "state.pt").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    for p in flat:
        shutil.copy(p, d / p.name.split("__", 2)[2])
        print(f"checkpoint phiên trước: {p.name} ({p.stat().st_size / 2**20:.0f} MB)", flush=True)


def find_checkpoint(name: str) -> Path | None:
    """Checkpoint của thí nghiệm ở input (output phiên trước gắn vào) hoặc trong WORK; lấy bản nhiều bước nhất."""
    restore_flat_checkpoint(name)
    hits = []
    for root in (INPUT, WORK):
        if root.exists():
            hits += [p.parent for p in root.rglob("state.pt")
                     if p.parent.name == "checkpoint" and p.parent.parent.name == f"{PREFIX}{name}-lora"]

    def step(c: Path) -> int:
        try:
            return json.loads((c / "meta.json").read_text(encoding="utf-8"))["step"]
        except Exception:
            return -1
    return max(hits, key=step) if hits else None


def remaining_minutes(a) -> float:
    return float("inf") if a.deadline == 0 else max(1.0, a.deadline - (time.time() - T0) / 60)


def command(name: str, data: Path, base: Path, a, nproc: int, extra: list[str]) -> list[str]:
    cmd = [str(HERE / "train_sft.py"), "--data", str(data), "--out", str(WORK / f"{PREFIX}{name}"), "--base", str(base),
           "--no-merge", "--deadline", "0" if a.deadline == 0 else f"{remaining_minutes(a):.1f}",
           "--ckpt-every", str(a.ckpt_every), *a.train_args, *extra]
    ckpt = find_checkpoint(name)
    cmd += ["--resume", str(ckpt) if ckpt else "auto"]
    if a.ddp and nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *cmd]
    return [sys.executable, *cmd]


def last_step(log: Path) -> str:
    lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()] if log.exists() else []
    return next((x for x in reversed(lines) if x.startswith("bước") or "Error" in x or "→" in x), lines[-1] if lines else "")[:200]


def gpu_mem() -> list[int]:
    try:
        r = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except OSError:
        return []


def run_training(name: str, cmd: list[str], log: Path, a, cap_min: float) -> tuple[int, list[int]]:
    """Chạy một lượt train_sft.py, ghi nối vào log; mỗi phút in bước mới nhất và theo dõi bộ nhớ đỉnh từng GPU.
    Quá cap_min phút mà chưa thoát (treo ở all-reduce, kernel biên dịch mãi...) thì giết cả nhóm tiến trình: một kernel
    treo sẽ đốt hạn mức GPU tới hết 12 giờ."""
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if a.ddp:
        env["OMP_NUM_THREADS"] = "2"
    print(f"{name} ← {' '.join(cmd[cmd.index('--data') - 1:])}", flush=True)
    peak: list[int] = []
    t = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n{SECTION}{time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        RUNNING.append(p)
        tick = 0
        while p.poll() is None:
            time.sleep(5)
            tick += 1
            if tick % 3 == 0:
                m = gpu_mem()
                peak = [max(x, y) for x, y in zip(m, peak + [0] * len(m))] if m else peak
            if tick % 12 == 0:
                print(time.strftime("%H:%M"), f"{name}: {last_step(log)}", flush=True)
            if time.time() - t > cap_min * 60:
                print(f"{name}: quá {cap_min:.0f} phút chưa thoát, giết nhóm tiến trình", flush=True)
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    p.kill()
                p.wait()
                RUNNING.remove(p)
                f.write(f"\nGIẾT: quá {cap_min:.0f} phút\n")
                return 124, peak
    RUNNING.remove(p)
    return p.returncode, peak


def measure(text: str) -> dict:
    """Số đo rút từ log của train_sft.py."""
    steps = re.findall(r"^bước (\d+)/(\d+) loss ([0-9.naif]+) lr \S+ (\d+) tok/s.*?(?:bộ nhớ đỉnh ([0-9.]+) GB)?(?:, bỏ (\d+) mẫu.*)?$", text, re.M)
    out: dict = {"fallback_gdn": len(re.findall(r"chunk_gated_delta_rule. is falling back", text)),
                 "fallback_conv": len(re.findall(r"causal_conv1d_fn. is falling back", text))}
    if steps:
        out.update(step=int(steps[-1][0]), total=int(steps[-1][1]), loss_first=steps[0][2], loss_last=steps[-1][2],
                   tok_s=int(steps[-1][3]), peak_alloc_gb_rank0=max(float(s[4]) for s in steps if s[4]) if any(s[4] for s in steps) else None,
                   dropped_nonfinite=int(steps[-1][5] or 0), losses=[s[2] for s in steps][:200])
    m = re.search(r"^(.*GradScaler.*|.*: (?:bf16|fp16|fp32).*)$", text, re.M)
    if m:
        out["device_line"] = m.group(1).strip()[:200]
    return out


def finish(name: str, rc: int, log: Path, peak: list[int]) -> str:
    """Nén adapter (xong) hoặc checkpoint dạng file phẳng (chưa xong); trả về 'done' | 'unfinished' | 'failed'."""
    adapter = WORK / f"{PREFIX}{name}-lora"
    text = log.read_text(encoding="utf-8", errors="replace")
    info = {**measure(text), "exit_code": rc, "gpu_mem_peak_mib": peak}
    SUMMARY["experiments"][name] = info
    if rc == 0 and (adapter / "adapter_model.safetensors").exists():
        shutil.make_archive(str(adapter), "zip", adapter.parent, adapter.name)
        info["status"] = "done"
        print(f"XONG {name}: {adapter}.zip ({Path(f'{adapter}.zip').stat().st_size / 2**20:.0f} MB), {info.get('tok_s')} tok/s, "
              f"loss {info.get('loss_first')} → {info.get('loss_last')}, bộ nhớ GPU đỉnh {peak} MiB, bỏ {info.get('dropped_nonfinite')} mẫu", flush=True)
        return "done"
    ckpt = adapter / "checkpoint"
    if (ckpt / "state.pt").exists():
        import zipfile

        z = WORK / f"{PREFIX}{name}-ckpt.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # file phẳng ckpt__<tên>__<file>, đưa thẳng vào dataset phiên sau
            for f in sorted(ckpt.iterdir()):
                if f.is_file():
                    zf.write(f, f"ckpt__{name}__{f.name}")
        info["checkpoint_zip_mb"] = round(z.stat().st_size / 2**20)
    if "CHƯA XONG" in text.rsplit(SECTION, 1)[-1] and (ckpt / "state.pt").exists():
        info["status"] = "unfinished"
        line = next(x for x in reversed(text.splitlines()) if "CHƯA XONG" in x)
        print(f"CHƯA XONG {name}: {line.split('CHƯA XONG: ', 1)[-1]}\n  → checkpoint {info['checkpoint_zip_mb']} MB ở "
              f"{PREFIX}{name}-ckpt.zip; chạy tiếp bằng scripts/kaggle_q35.sh resume", flush=True)
        return "unfinished"
    info["status"] = "failed"
    print(f"LỖI {name} (mã {rc}), 40 dòng cuối của {log}:", flush=True)
    print("\n".join(text.splitlines()[-40:]), flush=True)
    if (ckpt / "state.pt").exists():  # chết sau một checkpoint định kỳ: vẫn chạy tiếp được
        print(f"  còn checkpoint ({info['checkpoint_zip_mb']} MB): chạy tiếp bằng scripts/kaggle_q35.sh resume", flush=True)
    return "failed"


def smoke(name: str, data: Path, base: Path, a, nproc: int, log: Path) -> tuple[int, list[int]]:
    """4 bước: chặng 1 dừng sau bước 2 (phải để lại checkpoint bước 2), chặng 2 chạy tiếp tới hết. Trả mã thoát của
    chặng cuối (khác 0 nếu một chặng hỏng) và bộ nhớ GPU đỉnh."""
    rc, peak = run_training(name, command(name, data, base, a, nproc, ["--max-steps", "4", "--stop-after", "2"]), log, a, a.smoke_cap)
    ck = WORK / f"{PREFIX}{name}-lora" / "checkpoint" / "meta.json"
    step = json.loads(ck.read_text(encoding="utf-8")).get("step") if ck.exists() else None
    print(f"SMOKE {name} chặng 1 (dừng sau bước 2): mã thoát {rc}, checkpoint bước {step}", flush=True)
    if step != 2:
        return rc or 1, peak
    rc, peak2 = run_training(name, command(name, data, base, a, nproc, ["--max-steps", "4"]), log, a, a.smoke_cap)
    peak = [max(x, y) for x, y in zip(peak, peak2)] if peak and peak2 else peak or peak2
    resumed = "chạy tiếp từ" in log.read_text(encoding="utf-8", errors="replace").rsplit(SECTION, 1)[-1]
    print(f"SMOKE {name} chặng 2 (chạy tiếp): mã thoát {rc}, {'có' if resumed else 'KHÔNG'} dòng 'chạy tiếp từ'", flush=True)
    return (rc if resumed else rc or 1), peak


def drop_fla(why: str) -> None:
    """Gỡ FLA để Transformers dùng kernel tham chiếu (PyTorch thuần): đúng nhưng chậm khoảng gấp đôi."""
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "flash-linear-attention", "fla-core"], check=False)
    SUMMARY["fla_dropped"] = why
    SUMMARY["versions"] = versions()


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", metavar="TEN=FILE", help="thí nghiệm và file dữ liệu trong gói (chỉ TEN: train_<TEN>.jsonl)")
    ap.add_argument("--ddp", action="store_true", help="mỗi thí nghiệm dùng cả các GPU (torchrun), chạy lần lượt")
    ap.add_argument("--nproc", type=int, default=None, help="số rank của --ddp (mặc định: số GPU)")
    ap.add_argument("--deadline", type=float, default=675.0, metavar="MIN",
                    help="phút kể từ lúc chạy file này (Kaggle: 12 giờ, chừa 45 phút); 0 = không giới hạn")
    ap.add_argument("--ckpt-every", type=float, default=30.0, metavar="MIN")
    ap.add_argument("--smoke", action="store_true",
                    help="thử nhanh: 4 bước, dừng sau bước 2 (lưu checkpoint), chạy tiếp tới hết, nén adapter; kernel FLA "
                         "hỏng thì gỡ FLA và thử lại bằng kernel tham chiếu")
    ap.add_argument("--smoke-cap", type=float, default=40.0, metavar="MIN", help="--smoke: giết một chặng chạy quá số phút này")
    ap.add_argument("--no-fla", action="store_true", help="không cài flash-linear-attention: dùng kernel tham chiếu của Transformers")
    ap.add_argument("--require-checkpoint", action="store_true",
                    help="phiên chạy tiếp: không thấy checkpoint của thí nghiệm thì dừng ngay, không học lại từ đầu "
                         "(11/11: phiên 2 của synth-k không thấy file phẳng và đốt 3,6 giờ hạn mức học lại 102 bước)")
    ap.add_argument("--require-step", type=int, default=None, metavar="N",
                    help="phiên chạy tiếp: checkpoint tìm thấy phải ở bước N trở lên (N là bước của checkpoint vừa đẩy lên); "
                         "thấp hơn nghĩa là kernel gắn bản dataset cũ, dừng ngay thay vì học lại phần đã học")
    ap.add_argument("--base", default=None, help="thư mục model chỉ văn bản có sẵn (bỏ qua tải và xuất)")
    ap.add_argument("--no-install", action="store_true", help="không pip install (thư viện đã đủ)")
    if "--" in argv:
        k = argv.index("--")
        argv, train_args = argv[:k], argv[k + 1:]
    else:
        train_args = []
    a = ap.parse_args(argv)
    a.train_args = train_args
    exps = [parse_exp(x) for x in a.exps]
    WORK.mkdir(parents=True, exist_ok=True)
    if a.require_checkpoint or a.require_step is not None:
        # Kiểm TRƯỚC khi cài và dựng model nền: kernel có thể đã gắn bản dataset cũ (kpush đẩy kernel khi bản mới chưa
        # lên). Bản cũ hoặc không có checkpoint (phiên 2), hoặc có checkpoint của phiên trước nữa (phiên 3 trở đi).
        for name, _ in exps:
            ck = find_checkpoint(name)
            try:
                step = json.loads((ck / "meta.json").read_text(encoding="utf-8"))["step"] if ck else None
            except Exception:
                step = None
            if ck is None or (a.require_step is not None and (step is None or step < a.require_step)):
                seen = sorted(p.name for p in INPUT.rglob("ckpt__*"))[:8] if INPUT.exists() else []
                SUMMARY["experiments"][name] = {"status": "failed", "reason": "require-checkpoint", "checkpoint_step": step,
                                                "required_step": a.require_step}
                raise SystemExit(f"LỖI {name}: phiên chạy tiếp cần checkpoint" + (f" từ bước {a.require_step}" if a.require_step is not None else "")
                                 + f" nhưng thấy {'bước ' + str(step) if ck else 'không có checkpoint'} (file ckpt__ trong input: {seen or 'không có'}). "
                                 "Dataset chưa lên bản mới khi kernel bắt đầu: đẩy lại phiên chạy tiếp.")
            print(f"{name}: có checkpoint bước {step}" + (f" (cần từ bước {a.require_step})" if a.require_step is not None else ""), flush=True)
    # lưới cuối: cả phiên treo (tải model, pip...) thì thoát, không đốt hạn mức GPU tới hết 12 giờ
    total_cap = (2 * a.smoke_cap + 45) * len(exps) if a.smoke else (24 * 60 if a.deadline == 0 else a.deadline + 45)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(f"cả phiên quá {total_cap:.0f} phút")))
        signal.alarm(int(total_cap * 60))

    env_report()
    install(a)
    base = prepare_base(a)
    names = SUMMARY["versions"]["gpus"]  # hỏi trong tiến trình con: tiến trình này không giữ ngữ cảnh CUDA nào
    gpus = len(names)
    nproc = a.nproc or max(1, gpus)
    print(f"{gpus} GPU: {', '.join(names)}" + (f" — mỗi thí nghiệm dùng {nproc} rank" if a.ddp else ""), flush=True)
    SUMMARY.update(gpus=gpus, nproc=nproc if a.ddp else 1, args=vars(a))

    failed, unfinished = [], []
    for name, data in exps:
        log = WORK / f"{name}.log"
        if remaining_minutes(a) < 15:  # không đáng nạp model 8 GB để chạy một bước rồi dừng
            unfinished.append(name)
            print(f"CHƯA XONG {name}: hết hạn phiên, chưa bắt đầu; chạy ở phiên sau", flush=True)
            continue
        t = time.time()
        if a.smoke:
            rc, peak = smoke(name, data, base, a, nproc, log)
            if rc and SUMMARY["versions"].get("flash-linear-attention"):
                print(f"SMOKE {name} hỏng khi có FLA (mã {rc}); 30 dòng cuối:", flush=True)
                print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]), flush=True)
                SUMMARY["experiments"][name + " (FLA)"] = {**measure(log.read_text(encoding="utf-8", errors="replace")), "exit_code": rc}
                drop_fla("smoke hỏng với FLA")
                stale = WORK / f"{PREFIX}{name}-lora"
                if stale.is_dir():  # checkpoint dở của lượt hỏng: lượt thử lại phải bắt đầu từ đầu
                    shutil.rmtree(stale)
                log.rename(log.with_name(f"{name}.fla-failed.log"))
                print(f"SMOKE {name}: thử lại bằng kernel tham chiếu", flush=True)
                rc, peak = smoke(name, data, base, a, nproc, log)
        else:
            # train_sft.py tự dừng ở --deadline; chừa 30 phút cho bước đang chạy, lưu checkpoint và nén
            cap = 24 * 60 if a.deadline == 0 else remaining_minutes(a) + 30
            rc, peak = run_training(name, command(name, data, base, a, nproc, []), log, a, cap)
        status = finish(name, rc, log, peak)
        stage(f"train {name}", t)
        if status == "failed":
            failed.append(name)
        elif status == "unfinished":
            unfinished.append(name)
        elif a.smoke:
            kernel = "kernel tham chiếu (đã gỡ FLA)" if SUMMARY.get("fla_dropped") else f"FLA {SUMMARY['versions'].get('flash-linear-attention')}"
            print(f"SMOKE ĐẠT {name}: {kernel}", flush=True)
    if failed or unfinished:
        raise SystemExit((f"lỗi: {', '.join(failed)}" if failed else "") + (f" chưa xong: {', '.join(unfinished)}" if unfinished else ""))


if __name__ == "__main__":
    code = 0
    try:
        main(sys.argv[1:])
    except SystemExit as e:
        code = 0 if e.code is None else e.code if isinstance(e.code, int) else 1
        if code and not isinstance(e.code, int):
            print(e.code, flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
        for p in list(RUNNING):  # hết giờ cả phiên hoặc lỗi bất ngờ: không để tiến trình train mồ côi giữ GPU
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
    SUMMARY["minutes"] = round((time.time() - T0) / 60, 1)
    SUMMARY["exit_code"] = code
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "q35_summary.log").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    sys.exit(code)
