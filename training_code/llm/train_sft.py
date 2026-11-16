"""LoRA SFT cho agent react_json trên một GPU 16 GB, rồi gộp adapter vào trọng số để vLLM phục vụ trực tiếp.

bf16, gradient checkpointing, batch 1 không padding cộng dồn gradient. Loss chỉ tính trên lượt assistant (nội dung
và <|im_end|>), và lm_head chỉ chạy ở các vị trí đó: logits đủ 152K từ cho cả chuỗi 5K token sẽ tốn vài GB.
Adapter giữ fp32 nhưng nhân trong autocast bf16, không ép đầu vào sang fp32. Đo trên RTX 5060 Ti: khoảng 600 token/s
(60-65% mức matmul bf16 của card), đỉnh 11,2 GB; 1 epoch 4,3 triệu token mất 2 giờ kể cả nạp và gộp, loss từ 1,09
xuống 0,045, nên mặc định 1 epoch.

GPU chưa có bf16 (T4, P100 của Kaggle) tự chuyển sang fp16 có GradScaler; adapter vẫn fp32. Đường dẫn tuyệt đối
cho --data và --out được giữ nguyên, nên chạy được ngoài repo (kaggle/run_sft.py).

    .venv/bin/python scripts/train_sft.py --data sft/train.jsonl --out models/qwen3-4b-sft
    DETERMINISTIC=1 MODEL=models/qwen3-4b-sft scripts/serve_vllm.sh

Nhiều GPU cho MỘT lượt (torchrun, DDP tự viết, không bọc model):

    torchrun --standalone --nproc_per_node=2 scripts/train_sft.py --data sft/train.jsonl --out models/qwen3-4b-sft

Mọi rank cùng seed nên cùng thứ tự mẫu; trong mỗi cửa sổ cộng dồn, mẫu thứ k về rank k mod N; cuối cửa sổ gộp gradient
adapter của mọi rank bằng all-reduce TỔNG. Mỗi rank đã chia loss cho --grad-accum, nên gradient sau gộp
= Σ ∇loss_i / grad_accum, đúng bằng một GPU chạy cả cửa sổ. Vì vậy --grad-accum là cỡ lô HIỆU DỤNG (tổng mẫu mỗi bước
tối ưu, mặc định 16), mỗi rank tính grad_accum / N mẫu; số bước tối ưu = ceil(số mẫu × epochs / grad_accum) và lịch lr
(warmup 5%, cosine) không phụ thuộc N. N = 1 chạy đúng mã cũ, không gọi all-reduce. Chỉ rank 0 in log, lưu và gộp.
Khi loss một mẫu không hữu hạn (fp16 tràn) mẫu đó vẫn bị bỏ như cũ; số mẫu bỏ và loss trung bình được cộng qua các rank.

Lưu và chạy tiếp qua nhiều phiên: cứ --ckpt-every phút (30), và khi sắp quá --deadline phút kể từ lúc tiến trình khởi
động (675 = 11 giờ 15, cho phiên Kaggle 12 giờ), rank 0 lưu <out>-lora/checkpoint/ (adapter, optimizer, scheduler,
GradScaler, bước, thứ tự mẫu, RNG) rồi mọi rank thoát mã 3 và in dòng "CHƯA XONG". Phiên sau chạy lại cùng lệnh với
--resume <thư mục checkpoint> (Kaggle: output phiên trước gắn làm input) hoặc --resume auto (tìm <out>-lora/checkpoint)
để chạy tiếp tới hết; số rank có thể khác phiên trước. Xong thì checkpoint bị xoá.

--dry-run chỉ đọc dữ liệu: in số mẫu, số bước, tổng token qua model, token có loss, và thời gian ước lượng theo --speeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Pro (11/11): Qwen3.5 chỉ văn bản (model_type qwen3_5_text, xuất bởi LM-43). Thêm 5 phép chiếu Linear của 24 khối DeltaNet.
QWEN35_LORA_TARGETS = LORA_TARGETS + ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
SCAFFOLD = "<think>\n\n</think>\n\n"
# Trong repo: prompts/qwen35_scaffold.jinja. Trong gói Kaggle (thư mục phẳng): file nằm cạnh script này.
SCAFFOLD_TEMPLATE = next((p for p in (REPO / "prompts" / "qwen35_scaffold.jinja",
                                      Path(__file__).resolve().with_name("qwen35_scaffold.jinja")) if p.exists()),
                         REPO / "prompts" / "qwen35_scaffold.jinja")


def lora_modules(model) -> list[str]:
    """Tên đầy đủ các Linear gắn LoRA của Qwen3.5 (không đụng Conv1d / RMSNorm)."""
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in QWEN35_LORA_TARGETS]
    missing = set(QWEN35_LORA_TARGETS) - {n.rsplit(".", 1)[-1] for n in names}
    if missing:
        raise ValueError(f"thiếu module LoRA: {sorted(missing)}")
    return names


def keep_efficient_sdpa(dev: torch.device, say=print) -> None:
    """GPU trước Ampere (T4, sm75) không có kernel flash của scaled_dot_product_attention. Với Qwen3.5 (16 đầu
    attention, 4 đầu K/V, head_dim 256, không có mask) Transformers gọi SDPA với enable_gqa=True; không có flash thì
    PyTorch rơi về đường "math" và dựng cả ma trận attention 16 × L × L (1,27 GiB ở 6.500 token, vài bản trong
    backward): smoke Kaggle 11/11 hết 14,56 GiB ngay bước đầu ở cả hai kernel GDN. Ép Transformers lặp K/V theo nhóm
    (repeat_kv, đường mặc định khi có mask) để SDPA dùng được kernel tiết kiệm bộ nhớ. GPU từ Ampere trở lên không đổi."""
    if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8:
        return
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        say("GPU trước Ampere: SDPA không dùng enable_gqa (lặp K/V theo nhóm) để giữ kernel tiết kiệm bộ nhớ", flush=True)


def trained(m: dict) -> bool:
    """Tin nhắn có loss: lượt assistant, trừ khi mang "train": false (LM-92: bước sai còn trong ngữ cảnh nhưng không học,
    vd. call lọc trả rỗng trước ghi chú empty_retry). Hàng không có khoá "train" cho nhãn y như trước."""
    return m["role"] == "assistant" and m.get("train", True) is not False


def encode_scaffold(tok, messages: list[dict]) -> tuple[list[int], list[int], str]:
    """Một lượt cho cả hội thoại, khung think rỗng trước MỌI lượt assistant; phục vụ bằng prompts/qwen35_scaffold.jinja
    (vllm --chat-template) nên prefix của từng bước trùng từng token."""
    ids, labels, text = [], [], ""
    for m in messages:
        if set(m) - {"role", "content", "train"} or m["role"] not in {"system", "user", "assistant"}:
            raise ValueError("chỉ nhận role/content (và train) của react_json")
        c = m["content"].strip()
        if "<think>" in c or "<|im_start|>" in c or "<|im_end|>" in c:
            raise ValueError("token dành riêng trong dữ liệu")
        h = f"<|im_start|>{m['role']}\n" + (SCAFFOLD if m["role"] == "assistant" else "")
        b = f"{c}<|im_end|>"
        head = tok.encode(h, add_special_tokens=False)
        body = tok.encode(b, add_special_tokens=False)
        tail = tok.encode("\n", add_special_tokens=False)
        ids += head + body + tail
        labels += [-100] * len(head) + (body if trained(m) else [-100] * len(body)) + [-100] * len(tail)
        text += h + b + "\n"
    if ids != tok.encode(text, add_special_tokens=False):
        raise ValueError("mã hoá từng đoạn lệch với mã hoá cả chuỗi")
    return ids, labels, text
T_START = time.time()  # hạn phiên tính từ lúc tiến trình khởi động, gồm cả nạp model
EXIT_RESUME = 3  # thoát vì sắp hết hạn phiên: checkpoint đã lưu, chạy tiếp bằng --resume
# tốc độ đo được (BASELINE.md): T4 fp16 330 tok/s, RTX 5060 Ti bf16 600 tok/s; T4 ×2 lấy 2 × 330 × 0,9 (gộp gradient)
SPEEDS = "T4x1:330,T4x2:594,RTX5060Ti:600"
# tham số phải trùng giữa checkpoint và lần chạy tiếp (đổi là lịch lr hoặc thứ tự mẫu khác đi)
RESUME_KEYS = ["base", "epochs", "lr", "rank", "alpha", "dropout", "grad_accum", "warmup", "max_len", "max_steps", "seed"]


def encode(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Định dạng chat của Qwen viết tay theo từng tin nhắn, để biết chính xác token nào thuộc lượt assistant."""
    ids, labels = [], []
    for m in messages:
        head = tok.encode(f"<|im_start|>{m['role']}\n", add_special_tokens=False)
        body = tok.encode(f"{m['content']}<|im_end|>", add_special_tokens=False)
        tail = tok.encode("\n", add_special_tokens=False)
        ids += head + body + tail
        labels += [-100] * len(head) + (body if trained(m) else [-100] * len(body)) + [-100] * len(tail)
    return ids, labels


def load(tok, path: Path, max_len: int, say=print, q35: bool = False) -> list[tuple[list[int], list[int]]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    out, dropped = [], 0
    tmpl = SCAFFOLD_TEMPLATE.read_text(encoding="utf-8") if q35 else None
    for i, r in enumerate(rows):
        if q35:
            ids, labels, text = encode_scaffold(tok, r["messages"])
            if i < 50:  # trùng template phục vụ; bước đầu trùng template gốc của Qwen3.5 (enable_thinking=False)
                assert tok.apply_chat_template(r["messages"], chat_template=tmpl, tokenize=False) == text, f"{r['id']}: lệch template"
                k = next(j for j, m in enumerate(r["messages"]) if m["role"] == "assistant")
                assert (tok.apply_chat_template(r["messages"][:k], chat_template=tmpl, tokenize=False, add_generation_prompt=True)
                        == tok.apply_chat_template(r["messages"][:k], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)), f"{r['id']}: prefix bước đầu lệch template gốc"
        else:
            ids, labels = encode(tok, r["messages"])
        if not q35 and i < 20:  # phải trùng với chat template mà vLLM dùng lúc chạy
            ref = tok.apply_chat_template(r["messages"], tokenize=True)
            ref = ref["input_ids"] if isinstance(ref, dict) else ref
            assert ids == list(ref), f"{r['id']}: định dạng tự viết lệch với chat template"
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append((ids, labels))
    lens = sorted(len(x[0]) for x in out)
    n_lab = sum(sum(1 for t in lab if t != -100) for _, lab in out)
    say(f"{path.name}: {len(out)} mẫu (bỏ {dropped} dài hơn {max_len}), token p50 {lens[len(lens) // 2]}, "
        f"p95 {lens[int(.95 * (len(lens) - 1))]}, max {lens[-1]}, tổng {sum(lens)}, token có loss {n_lab}", flush=True)
    return out


def loss_fn(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE trên các vị trí có nhãn; chỉ chiếu lm_head ở đó."""
    causal = model.get_base_model()
    hidden = causal.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    target = labels[0, 1:]
    pos = (target != -100).nonzero(as_tuple=True)[0]
    logits = causal.lm_head(hidden[pos]).float()
    return F.cross_entropy(logits, target[pos])


def make_order(n: int, total: int, grad_accum: int, rng=random) -> list[int]:
    """Thứ tự mẫu: trộn từng epoch, nối tới khi đủ total × grad_accum mẫu (như mã một GPU)."""
    order: list[int] = []
    while len(order) < total * grad_accum:
        ep = list(range(n))
        rng.shuffle(ep)
        order += ep
    return order


def dist_setup(a) -> tuple[int, int, torch.device]:
    """Đọc RANK / WORLD_SIZE / LOCAL_RANK do torchrun đặt; không có thì một tiến trình như cũ."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    if a.dry_run:  # chỉ cần tokenizer: không chạm GPU (máy đang train vẫn chạy được), không cần process group
        return 0, 1, torch.device("cpu")
    if a.device == "auto":  # như mã cũ: không có CUDA thì dừng ngay, không âm thầm học trên CPU
        a.device = "cuda"
    if a.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("không có CUDA; --device cpu chỉ dành cho test với model tí hon")
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
    if world > 1:
        backend = a.backend or ("nccl" if dev.type == "cuda" else "gloo")
        # nạp 8 GB trọng số trên mỗi rank có thể mất vài phút; không để rank nhanh hơn hết giờ chờ
        dist.init_process_group(backend, rank=rank, world_size=world, timeout=timedelta(minutes=60))
    return rank, world, dev


def allreduce_grads(params: list[torch.Tensor]) -> None:
    """Gộp gradient adapter của mọi rank bằng TỔNG trong một bộ đệm phẳng (33M tham số fp32 ≈ 132 MB).
    Rank bỏ cả cửa sổ (mọi loss tràn) chưa có .grad thì góp 0."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    off = 0
    for p, g in zip(params, grads):
        p.grad = flat[off:off + g.numel()].view_as(p)
        off += g.numel()


def save_checkpoint(model, opt, sched, scaler, state: dict, ckpt: Path) -> None:
    """Ghi vào .tmp rồi đổi tên, để lỡ chết giữa lúc ghi vẫn còn checkpoint cũ (checkpoint.old)."""
    tmp, old = ckpt.with_name(ckpt.name + ".tmp"), ckpt.with_name(ckpt.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)  # adapter_model.safetensors + adapter_config.json: tự nó cũng là một adapter dùng được
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng, **state},
               tmp / "state.pt")
    (tmp / "meta.json").write_text(json.dumps({k: v for k, v in state.items() if k != "order"}, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    if ckpt.exists():
        ckpt.rename(old)
    tmp.rename(ckpt)
    shutil.rmtree(old, ignore_errors=True)


def data_id(a) -> str:
    """Tên và cỡ file dữ liệu: đủ để phân biệt hai file cùng số mẫu (train_drop so với train_drop_rename)."""
    f = REPO / a.data
    return f"{f.name}:{f.stat().st_size}"


def find_checkpoint(path: Path) -> Path | None:
    for c in (path, path / "checkpoint", path.with_name(path.name + ".old")):
        if (c / "state.pt").exists():
            return c
    return None


def load_checkpoint(ckpt: Path, model, opt, sched, scaler, a, n_data: int, say) -> dict:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    st = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
    bad = {k: (st["args"].get(k), getattr(a, k)) for k in RESUME_KEYS if st["args"].get(k) != getattr(a, k)}
    if st["n_data"] != n_data:
        bad["số mẫu"] = (st["n_data"], n_data)
    if "data" in st and st["data"] != data_id(a):  # cùng số mẫu nhưng file khác (drop / drop-rename): thứ tự mẫu trỏ sai dữ liệu
        bad["dữ liệu"] = (st["data"], data_id(a))  # checkpoint của bản trước 072abf7 không có khoá này: chỉ so số mẫu
    if bad:
        sys.exit(f"checkpoint {ckpt} không cùng cấu hình: " + ", ".join(f"{k} {v[0]!r} → {v[1]!r}" for k, v in bad.items()))
    if st["args"].get("dtype") != a.dtype:
        say(f"cảnh báo: checkpoint học bằng {st['args'].get('dtype')}, phiên này {a.dtype}; kết quả sẽ lệch trong sai số dtype")
    res = set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
    if getattr(res, "unexpected_keys", None):
        sys.exit(f"adapter trong checkpoint không khớp model: {res.unexpected_keys[:3]}")
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    if scaler.is_enabled():  # checkpoint bf16/fp32 không có trạng thái scaler ({}), nạp vào scaler đang bật sẽ lỗi
        if st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        else:
            say(f"checkpoint không có GradScaler (học bằng {st['args'].get('dtype')}); fp16 bắt đầu lại từ init_scale 2^12")
    random.setstate(st["rng"]["python"])
    torch.set_rng_state(st["rng"]["torch"])
    if torch.cuda.is_available() and st["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(st["rng"]["cuda"][: torch.cuda.device_count()])
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="sft/train.jsonl")
    ap.add_argument("--out", default="models/qwen3-4b-sft")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=16, help="số mẫu mỗi bước tối ưu, tính chung mọi rank")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--max-steps", type=int, default=None, help="dừng sớm sau N bước tối ưu (đo tốc độ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="auto: bf16 nếu GPU hỗ trợ thật, không thì fp16; fp32 cho CPU (test)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="cpu chỉ để test với model tí hon")
    ap.add_argument("--backend", default=None, help="nccl (GPU) hay gloo (CPU); mặc định theo --device")
    ap.add_argument("--resume", default=None, metavar="DIR", help="thư mục checkpoint, hoặc auto = <out>-lora/checkpoint nếu có")
    ap.add_argument("--ckpt-every", type=float, default=30.0, metavar="MIN", help="phút giữa hai lần lưu checkpoint; 0 = mỗi bước")
    ap.add_argument("--deadline", type=float, default=675.0, metavar="MIN",
                    help="phút kể từ lúc khởi động; sắp quá thì lưu checkpoint và thoát mã 3 (0 = không giới hạn)")
    ap.add_argument("--stop-after", type=int, default=None, metavar="STEP", help="lưu checkpoint và thoát mã 3 sau bước này (giả lập hết hạn, để test)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đọc dữ liệu và ước lượng thời gian, không nạp model")
    ap.add_argument("--speeds", default=SPEEDS, help="tốc độ tok/s cho --dry-run, dạng tên:số,tên:số")
    a = ap.parse_args(argv)

    rank, world, dev = dist_setup(a)
    main_rank = rank == 0
    say = print if main_rank else (lambda *x, **k: None)

    from peft import LoraConfig, get_peft_model
    from peft.helpers import disable_input_dtype_casting
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = REPO / a.out
    adapter = out.with_name(out.name + "-lora")
    ckpt_dir = adapter / "checkpoint"
    tok = AutoTokenizer.from_pretrained(a.base)
    q35 = AutoConfig.from_pretrained(a.base).model_type == "qwen3_5_text"
    data = load(tok, REPO / a.data, a.max_len, say, q35)
    total = math.ceil(len(data) * a.epochs / a.grad_accum)
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(a.warmup * total))

    if a.dry_run:
        order = make_order(len(data), total, a.grad_accum, random.Random(a.seed))
        used = order[: total * a.grad_accum]
        n_tok = sum(len(data[i][0]) for i in used)
        n_lab = sum(sum(1 for t in data[i][1] if t != -100) for i in used)
        say(f"ước lượng: {len(data)} mẫu × {a.epochs} epoch = {total} bước × {a.grad_accum} mẫu; {n_tok:,} token qua model, "
            f"{n_lab:,} token có loss ({100 * n_lab / max(1, n_tok):.1f}%); warmup {warm} bước", flush=True)
        for item in a.speeds.split(","):
            name, speed = item.split(":")
            hours = n_tok / float(speed) / 3600
            sessions = f", {math.ceil(hours * 60 / a.deadline)} phiên Kaggle {a.deadline / 60:.2f} giờ" if a.deadline else ""
            say(f"  {name:<12} {float(speed):>6.0f} tok/s → {hours:5.1f} giờ{sessions}", flush=True)
        return

    if a.dtype == "auto":  # is_bf16_supported() trả True cả khi chỉ giả lập (T4), nên xét thẳng compute capability
        if dev.type == "cuda":
            a.dtype = "bf16" if torch.cuda.get_device_capability(dev)[0] >= 8 else "fp16"
        else:
            a.dtype = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    # init_scale nhỏ: mức mặc định 2^16 làm tràn và bỏ mất vài bước đầu, đáng kể khi cả lượt chỉ khoảng 60 bước
    scaler = torch.amp.GradScaler(dev.type, init_scale=2.0**12, enabled=a.dtype == "fp16")
    where = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    say(f"{where}: {a.dtype}" + (", GradScaler bật" if scaler.is_enabled() else "") + (f", {world} rank" if world > 1 else ""),
        flush=True)
    if q35:
        keep_efficient_sdpa(dev, say)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=lora_modules(model) if q35 else LORA_TARGETS, task_type="CAUSAL_LM"))
    if main_rank:
        model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    order = make_order(len(data), total, a.grad_accum)
    step, toks, bad, elapsed = 0, 0, 0, 0.0
    resume = None
    if a.resume:
        resume = find_checkpoint(ckpt_dir if a.resume == "auto" else Path(a.resume))
        if resume is None and a.resume != "auto":
            sys.exit(f"không thấy state.pt trong {a.resume}")
    if resume:
        st = load_checkpoint(resume, model, opt, sched, scaler, a, len(data), say)
        step, toks, bad, elapsed = st["step"], st["toks"], st["bad"], st["elapsed"]
        if st["order"] != order:
            say("cảnh báo: thứ tự mẫu sinh lại khác checkpoint, dùng thứ tự trong checkpoint")
        order = st["order"]
        say(f"chạy tiếp từ {resume}: bước {step}/{total}, {toks:,} token, {elapsed / 60:.0f} phút đã học, dữ liệu {data_id(a)}",
            flush=True)
    elif a.resume == "auto":
        say("--resume auto: chưa có checkpoint, học từ đầu", flush=True)

    if world > a.grad_accum:
        sys.exit(f"{world} rank nhưng --grad-accum {a.grad_accum}: có rank không bao giờ nhận mẫu; dùng ít GPU hơn hoặc tăng --grad-accum")
    if scaler.is_enabled():  # tạo _scale ngay: rank mà cả cửa sổ đầu đều tràn vẫn phải unscale_ được (run_n là số toàn cục)
        scaler.scale(torch.zeros((), device=dev))
    model.train()
    t0 = time.time() - elapsed
    t_loop, step0, last_ckpt = time.time(), step, time.time()
    run_loss, run_n, win_bad, win_toks = 0.0, 0, 0, 0
    for k in range(step * a.grad_accum, total * a.grad_accum):
        if k % world == rank:  # mẫu thứ k của cửa sổ về rank k mod N; N = 1 thì rank này lấy hết
            ids, labels = data[order[k]]
            ids_t = torch.tensor([ids], device=dev)
            lab_t = torch.tensor([labels], device=dev)
            with torch.autocast(dev.type, dtype=dtype, enabled=a.dtype != "fp32"), disable_input_dtype_casting(model):
                loss = loss_fn(model, ids_t, lab_t)
            win_toks += len(ids)
            if torch.isfinite(loss):
                scaler.scale(loss / a.grad_accum).backward()
                run_loss += loss.item()
                run_n += 1
            else:  # fp16 tràn ở forward: bỏ mẫu này thay vì làm hỏng cả bước
                win_bad += 1
        if (k + 1) % a.grad_accum:
            continue
        # cuối cửa sổ. Rank 0 quyết định lưu / dừng; quyết định đi cùng thống kê để mọi rank làm giống nhau
        save_now = stop_now = False
        if main_rank and step + 1 < total:
            now = time.time()
            per_step = (now - t_loop) / (step + 1 - step0)
            stop_now = bool(a.stop_after and step + 1 >= a.stop_after) or \
                bool(a.deadline and now - T_START + 1.5 * per_step > a.deadline * 60)
            save_now = stop_now or now - last_ckpt >= a.ckpt_every * 60
        if world > 1:
            stats = torch.tensor([run_loss, run_n, win_bad, win_toks, save_now, stop_now], dtype=torch.float64, device=dev)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            run_loss, run_n, win_bad, win_toks = stats[0].item(), int(stats[1].item()), int(stats[2].item()), int(stats[3].item())
            save_now, stop_now = bool(stats[4].item()), bool(stats[5].item())
            allreduce_grads(params)
        toks += win_toks
        bad += win_bad
        if run_n:  # cả cửa sổ đều tràn thì không có gradient, GradScaler sẽ báo lỗi nếu vẫn step
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        el = time.time() - t0
        say(f"bước {step}/{total} loss {run_loss / max(1, run_n):.4f} lr {sched.get_last_lr()[0]:.2e} "
            f"{toks / el:.0f} tok/s, còn {el / step * (total - step) / 60:.0f} phút"
            + (f", bộ nhớ đỉnh {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB" if dev.type == "cuda" else "")
            + (f", bỏ {bad} mẫu loss không hữu hạn" if bad else ""), flush=True)
        run_loss, run_n, win_bad, win_toks = 0.0, 0, 0, 0
        if bad > len(data) // 10:
            sys.exit(f"quá nhiều mẫu loss không hữu hạn ({bad}): thử --dtype bf16 hoặc lr nhỏ hơn")
        if save_now:
            if main_rank:
                save_checkpoint(model, opt, sched, scaler,
                                {"step": step, "toks": toks, "bad": bad, "elapsed": el, "order": order, "n_data": len(data),
                                 "data": data_id(a), "total": total, "world": world, "args": vars(a)}, ckpt_dir)
                last_ckpt = time.time()
                say(f"checkpoint bước {step} → {ckpt_dir}", flush=True)
            if world > 1:
                dist.barrier()  # không rank nào thoát trước khi rank 0 ghi xong (torchrun sẽ giết các rank còn lại)
        if stop_now:
            say(f"CHƯA XONG: dừng sau bước {step}/{total} ({(time.time() - T_START) / 60:.0f} phút kể từ khởi động); "
                f"chạy tiếp bằng --resume {ckpt_dir}", flush=True)
            if world > 1:
                dist.destroy_process_group()
            sys.exit(EXIT_RESUME)

    if world > 1:
        dist.destroy_process_group()
    if not main_rank:
        return
    model.save_pretrained(adapter)
    (adapter / "train_args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # adapter cuối thay cho checkpoint; để output Kaggle nhẹ
    print(f"adapter → {adapter}", flush=True)
    if a.no_merge:
        return
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    print(f"đã gộp → {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
