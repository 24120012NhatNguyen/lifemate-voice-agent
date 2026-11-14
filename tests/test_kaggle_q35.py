"""Gói Kaggle của Qwen3.5 (kaggle/run_sft_q35.py + scripts/train_sft_q35r.py, dựng bằng scripts/kaggle_q35.sh pack)
chạy trên CPU với model Qwen3.5 tí hon: --smoke (dừng sau bước 2 rồi chạy tiếp), checkpoint dạng file phẳng sang
"phiên" khác ra đúng adapter của lượt chạy liền, và 2 rank ra cùng adapter như 1 rank.

Cần Transformers có Qwen3_5 (5.x), peft, và tokenizer thật của Qwen3.5 (biến QWEN35_TOKENIZER, mặc định
models/qwen35-4b-text): trainer so mã hoá của nó với chat template gốc. Không có thì bỏ qua. Trên PC:

    CUDA_VISIBLE_DEVICES= QWEN35_TOKENIZER=<pro-threshold>/models/qwen35-4b-text PYTHONPATH=<.test-deps> \
        <.venv-vllm>/bin/python -m pytest -q tests/test_kaggle_q35.py

Không đặt flash-linear-attention lên PYTHONPATH: kernel Triton của nó chỉ chạy trên GPU. Dữ liệu là chữ ngẫu nhiên,
không dính dữ liệu BTC; mỗi lượt là tiến trình con như trên Kaggle, vài chục giây, dưới 2 GB RAM.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
if not hasattr(transformers, "Qwen3_5ForCausalLM"):
    pytest.skip("Transformers chưa có Qwen3.5", allow_module_level=True)
if sys.platform == "win32":
    pytest.skip("gói Kaggle chạy bằng bash + torchrun trên Linux", allow_module_level=True)

from safetensors.torch import load_file  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = Path(os.environ.get("QWEN35_TOKENIZER", REPO / "models" / "qwen35-4b-text"))
if not (TOKENIZER / "tokenizer.json").exists():
    pytest.skip(f"không có tokenizer Qwen3.5 ở {TOKENIZER}", allow_module_level=True)

WORDS = ["xin", "chào", "bạn", "nhé", "ơi", "đặt", "lịch", "họp", "ngày", "mai", "gọi", "cho", "anh", "chị", "tiền", "nghìn",
         "bật", "đèn", "phòng", "khách", "mở", "nhạc", "tìm", "quán", "gần", "đây", "nhắc", "tôi", "lúc", "giờ", "sáng", "tối"]
TRAIN = ["--device", "cpu", "--dtype", "fp32", "--backend", "gloo", "--grad-accum", "4", "--lr", "1e-3", "--rank", "4",
         "--alpha", "8", "--max-len", "512"]
PREFIX = "qwen35-4b-sft-"


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    """Qwen3.5 2 lớp (một Gated DeltaNet, một attention đủ), ẩn 32, với tokenizer thật."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    path = tmp_path_factory.mktemp("tiny-qwen35")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(1234)
    cfg = Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]})
    Qwen3_5ForCausalLM(cfg).save_pretrained(path)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if (TOKENIZER / f).exists():
            shutil.copy(TOKENIZER / f, path / f)
    assert json.loads((path / "config.json").read_text())["model_type"] == "qwen3_5_text"
    return path


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> Path:
    """Gói phẳng như dataset Kaggle, dựng bằng chính scripts/kaggle_q35.sh pack, với 24 hội thoại giả."""
    root = tmp_path_factory.mktemp("pack")
    rng = random.Random(7)
    data = root / "train_fake.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for i in range(24):
            words = lambda n: " ".join(rng.choice(WORDS) for _ in range(n))  # noqa: E731
            msgs = [{"role": "system", "content": words(rng.randint(20, 40))}, {"role": "user", "content": words(rng.randint(5, 15))},
                    {"role": "assistant", "content": json.dumps({"thought": "", "action": "call", "name": "demo." + words(1)})}]
            if i % 3 == 0:
                msgs += [{"role": "user", "content": "KẾT QUẢ " + words(8)}, {"role": "assistant", "content": '{"thought": "", "action": "stop"}'}]
            f.write(json.dumps({"id": f"fake_{i:03d}", "messages": msgs}, ensure_ascii=False) + "\n")
    out = root / "pack"
    subprocess.run(["bash", str(REPO / "scripts" / "kaggle_q35.sh"), "pack", "t", str(data), str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    assert {p.name for p in out.iterdir()} >= {"run_sft.py", "train_sft.py", "export_text_only.py", "qwen35_scaffold.jinja",
                                               "train_fake.jsonl"}
    return out


def session(pack: Path, base: Path, work: Path, inputs: Path, *args: str, train: tuple[str, ...] = (), check: bool = True):
    """Một "phiên Kaggle": run_sft.py trong gói, với WORK và KAGGLE_INPUT riêng, không GPU, không cài gì."""
    inputs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "WORK": str(work), "KAGGLE_INPUT": str(inputs), "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1",
           "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, str(pack / "run_sft.py"), "--no-install", "--base", str(base), "--deadline", "0", *args,
           "t=train_fake.jsonl", "--", *TRAIN, *train]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                       errors="replace", timeout=600)
    if check:
        assert r.returncode == 0, r.stdout[-4000:] + "\n" + (work / "t.log").read_text(encoding="utf-8", errors="replace")[-4000:]
    return r


def adapter_of(work: Path) -> dict:
    z = work / f"{PREFIX}t-lora.zip"
    assert z.exists(), sorted(p.name for p in work.iterdir())
    with zipfile.ZipFile(z) as zf:
        zf.extract(f"{PREFIX}t-lora/adapter_model.safetensors", work / "unzipped")
    return load_file(work / "unzipped" / f"{PREFIX}t-lora" / "adapter_model.safetensors")


def worst(a: dict, b: dict) -> float:
    assert a.keys() == b.keys()
    return max((a[k] - b[k]).abs().max().item() for k in a)


def test_smoke_stops_resumes_and_zips(pack, base, tmp_path):
    work = tmp_path / "work"
    r = session(pack, base, work, tmp_path / "input", "--ddp", "--nproc", "2", "--smoke")
    assert "SMOKE t chặng 1 (dừng sau bước 2): mã thoát" in r.stdout and "checkpoint bước 2" in r.stdout
    assert "SMOKE ĐẠT t" in r.stdout, r.stdout[-3000:]
    s = json.loads((work / "q35_summary.log").read_text(encoding="utf-8"))
    e = s["experiments"]["t"]
    assert (e["status"], e["step"], e["total"], s["exit_code"]) == ("done", 4, 4, 0)
    assert e["fallback_gdn"] >= 1  # CPU: kernel tham chiếu của Transformers, và log ghi nhận điều đó
    log = (work / "t.log").read_text(encoding="utf-8")
    assert "CHƯA XONG: dừng sau bước 2/4" in log and "chạy tiếp từ" in log
    assert not (work / f"{PREFIX}t-lora" / "checkpoint").exists()  # xong thì checkpoint bị xoá
    names = set(adapter_of(work))
    assert any("linear_attn.in_proj_qkv" in k for k in names) and any("self_attn.q_proj" in k for k in names)


def test_flat_checkpoint_resumes_in_another_session(pack, base, tmp_path):
    """Phiên 1 dừng sau bước 3 và để lại <adapter>-ckpt.zip; các file phẳng trong đó làm input của phiên 2 (WORK mới);
    adapter cuối bằng đúng lượt chạy liền 6 bước."""
    w1, w2, w3 = tmp_path / "w1", tmp_path / "w2", tmp_path / "w3"
    r1 = session(pack, base, w1, tmp_path / "in1", "--ddp", "--nproc", "2", train=("--stop-after", "3"), check=False)
    assert r1.returncode != 0 and "CHƯA XONG t" in r1.stdout, r1.stdout[-3000:]
    assert json.loads((w1 / "q35_summary.log").read_text(encoding="utf-8"))["experiments"]["t"]["status"] == "unfinished"
    flat = tmp_path / "in2" / "lifemate-q35-t-data"
    flat.mkdir(parents=True)
    with zipfile.ZipFile(w1 / f"{PREFIX}t-ckpt.zip") as zf:
        assert all(n.startswith("ckpt__t__") and "/" not in n for n in zf.namelist()), zf.namelist()
        assert "ckpt__t__state.pt" in zf.namelist() and "ckpt__t__adapter_model.safetensors" in zf.namelist()
        zf.extractall(flat)
    r2 = session(pack, base, w2, tmp_path / "in2", "--ddp", "--nproc", "2")
    assert "checkpoint phiên trước: ckpt__t__state.pt" in r2.stdout
    assert "chạy tiếp từ" in (w2 / "t.log").read_text(encoding="utf-8") and "bước 3/6" in (w2 / "t.log").read_text(encoding="utf-8")
    session(pack, base, w3, tmp_path / "in3", "--ddp", "--nproc", "2")
    assert worst(adapter_of(w2), adapter_of(w3)) <= 1e-6


def test_resume_session_refuses_missing_or_stale_checkpoint(pack, base, tmp_path):
    """Phiên chạy tiếp gặp bản dataset cũ: không có checkpoint, hoặc checkpoint ở bước thấp hơn bước vừa đẩy. Cả hai phải
    dừng ngay (trước khi train), không học lại."""
    r = session(pack, base, tmp_path / "w0", tmp_path / "in0", "--ddp", "--nproc", "2", "--require-checkpoint", check=False)
    assert r.returncode != 0 and "không có checkpoint" in r.stdout and not (tmp_path / "w0" / "t.log").exists(), r.stdout[-1500:]
    w1 = tmp_path / "w1"
    session(pack, base, w1, tmp_path / "in1", "--ddp", "--nproc", "2", train=("--stop-after", "2"), check=False)
    flat = tmp_path / "in2" / "data"
    flat.mkdir(parents=True)
    with zipfile.ZipFile(w1 / f"{PREFIX}t-ckpt.zip") as zf:
        zf.extractall(flat)
    stale = session(pack, base, tmp_path / "w2", tmp_path / "in2", "--ddp", "--nproc", "2", "--require-checkpoint",
                    "--require-step", "4", check=False)
    assert stale.returncode != 0 and "thấy bước 2" in stale.stdout and not (tmp_path / "w2" / "t.log").exists(), stale.stdout[-1500:]
    ok = session(pack, base, tmp_path / "w3", tmp_path / "in2", "--ddp", "--nproc", "2", "--require-checkpoint", "--require-step", "2")
    assert "có checkpoint bước 2 (cần từ bước 2)" in ok.stdout and "bước 3/6" in (tmp_path / "w3" / "t.log").read_text(encoding="utf-8")


def test_two_ranks_match_one_rank(pack, base, tmp_path):
    w1, w2 = tmp_path / "one", tmp_path / "two"
    session(pack, base, w1, tmp_path / "in1")
    session(pack, base, w2, tmp_path / "in2", "--ddp", "--nproc", "2")
    assert worst(adapter_of(w1), adapter_of(w2)) <= 1e-5
