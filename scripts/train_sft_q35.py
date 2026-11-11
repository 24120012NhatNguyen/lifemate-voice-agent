"""LoRA SFT cho agent react_json trên một GPU 16 GB, rồi gộp adapter vào trọng số để vLLM phục vụ trực tiếp.

Qwen3.5 text-only: --base models/qwen35-4b-text; xem docs/QWEN35_TEXT_ONLY.md.
Các số đo tốc độ/VRAM dưới đây chỉ là Qwen3 cũ, chưa đo Qwen3.5.

bf16, gradient checkpointing, batch 1 không padding cộng dồn gradient. Loss chỉ tính trên lượt assistant (nội dung
và <|im_end|>), và lm_head chỉ chạy ở các vị trí đó: logits đủ 152K từ cho cả chuỗi 5K token sẽ tốn vài GB.
Adapter giữ fp32 nhưng nhân trong autocast bf16, không ép đầu vào sang fp32. Đo trên RTX 5060 Ti: khoảng 600 token/s
(60-65% mức matmul bf16 của card), đỉnh 11,2 GB; 1 epoch 4,3 triệu token mất 2 giờ kể cả nạp và gộp, loss từ 1,09
xuống 0,045, nên mặc định 1 epoch.

GPU chưa có bf16 (T4, P100 của Kaggle) tự chuyển sang fp16 có GradScaler; adapter vẫn fp32. Đường dẫn tuyệt đối
cho --data và --out được giữ nguyên, nên chạy được ngoài repo (kaggle/run_sft.py).

    .venv/bin/python scripts/train_sft.py --data sft/train.jsonl --out models/qwen3-4b-sft
    DETERMINISTIC=1 MODEL=models/qwen3-4b-sft scripts/serve_vllm.sh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections.abc import Mapping

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QWEN35_LORA_TARGETS = LORA_TARGETS + ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def lora_modules(model):
    """Resolve exact Linear names, including the 24 DeltaNet blocks. Never target Conv1d/RMSNorm."""
    targets = QWEN35_LORA_TARGETS if model.config.model_type == "qwen3_5_text" else LORA_TARGETS
    modules = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    missing = set(targets) - {name.rsplit(".", 1)[-1] for name in modules}
    if missing:
        raise ValueError(f"LoRA targets absent from model: {sorted(missing)}")
    return modules


def encode_no_thinking(tok, messages):
    """One assistant step with the exact native inference prefix; only completion + EOS has loss.

    Qwen3.5 removes historical think scaffolds on later user turns. Expanding trajectories
    into assistant-step examples avoids teaching earlier completions under a different prefix.
    """
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected a conversation ending in an assistant target")
    for m in messages:
        if set(m) - {"role", "content"} or m["role"] not in {"system", "user", "assistant"}:
            raise ValueError("Native SFT accepts only react_json role/content messages")
        if not isinstance(m["content"], str) or "<think>" in m["content"] or "</think>" in m["content"]:
            raise ValueError("Thinking/multimodal data is not supported in no-thinking SFT")
        if "<|im_start|>" in m["content"] or "<|im_end|>" in m["content"]:
            raise ValueError("Reserved chat tokens in data")
    kwargs = dict(tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    complete = tok.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if not complete.startswith(prompt):
        raise ValueError("Native template target differs from inference prefix")
    prefix_ids = tok.encode(prompt, add_special_tokens=False)
    ids = tok.encode(complete, add_special_tokens=False)
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Token boundary differs from inference prefix")
    # The final newline is formatting; EOS remains supervised.
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if eos not in ids[len(prefix_ids):]:
        raise ValueError("No assistant EOS")
    stop = len(ids) - 1 - ids[::-1].index(eos) + 1
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):stop] + [-100] * (len(ids) - stop)
    return ids, labels


SCAFFOLD = "<think>\n\n</think>\n\n"
SCAFFOLD_TEMPLATE = REPO / "prompts" / "qwen35_scaffold.jinja"


def encode_scaffold(tok, messages):
    """Pro, 11/11: một lượt cho cả hội thoại, khung think rỗng trước MỌI lượt assistant.

    Lúc phục vụ dùng prompts/qwen35_scaffold.jinja (vllm --chat-template) nên prefix của từng bước trùng từng token,
    không cần mở rộng mỗi bước assistant thành một ví dụ (đỡ khoảng 3 lần token).
    """
    ids, labels, text = [], [], ""
    for m in messages:
        if set(m) - {"role", "content"} or m["role"] not in {"system", "user", "assistant"}:
            raise ValueError("chỉ nhận role/content của react_json")
        c = m["content"].strip()
        if "<think>" in c or "<|im_start|>" in c or "<|im_end|>" in c:
            raise ValueError("token dành riêng trong dữ liệu")
        h = f"<|im_start|>{m['role']}\n" + (SCAFFOLD if m["role"] == "assistant" else "")
        b = f"{c}<|im_end|>"
        head = tok.encode(h, add_special_tokens=False)
        body = tok.encode(b, add_special_tokens=False)
        tail = tok.encode("\n", add_special_tokens=False)
        ids += head + body + tail
        labels += [-100] * len(head) + (body if m["role"] == "assistant" else [-100] * len(body)) + [-100] * len(tail)
        text += h + b + "\n"
    if ids != tok.encode(text, add_special_tokens=False):
        raise ValueError("mã hoá từng đoạn lệch với mã hoá cả chuỗi")
    return ids, labels, text


def encode(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Định dạng chat của Qwen viết tay theo từng tin nhắn, để biết chính xác token nào thuộc lượt assistant."""
    ids, labels = [], []
    for m in messages:
        head = tok.encode(f"<|im_start|>{m['role']}\n", add_special_tokens=False)
        body = tok.encode(f"{m['content']}<|im_end|>", add_special_tokens=False)
        tail = tok.encode("\n", add_special_tokens=False)
        ids += head + body + tail
        labels += [-100] * len(head) + (body if m["role"] == "assistant" else [-100] * len(body)) + [-100] * len(tail)
    return ids, labels


def load(tok, path: Path, max_len: int, native_no_thinking=False) -> list[tuple[list[int], list[int]]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tmpl = SCAFFOLD_TEMPLATE.read_text(encoding="utf-8") if native_no_thinking else None
    out, dropped = [], 0
    for i, r in enumerate(rows):
        if native_no_thinking:
            ids, labels, text = encode_scaffold(tok, r["messages"])
            if i < 50:  # phải trùng template phục vụ, và bước đầu phải trùng template gốc của Qwen3.5
                ref = tok.apply_chat_template(r["messages"], chat_template=tmpl, tokenize=False)
                assert ref == text, f"{r['id']}: lệch với prompts/qwen35_scaffold.jinja"
                first = next(k for k, m in enumerate(r["messages"]) if m["role"] == "assistant")
                a_ = tok.apply_chat_template(r["messages"][:first], chat_template=tmpl, tokenize=False, add_generation_prompt=True)
                b_ = tok.apply_chat_template(r["messages"][:first], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                assert a_ == b_, f"{r['id']}: prefix bước đầu lệch với template gốc"
        else:
            ids, labels = encode(tok, r["messages"])
            if i < 20:  # phải trùng với chat template mà vLLM dùng lúc chạy
                ref = tok.apply_chat_template(r["messages"], tokenize=True)
                ref = ref["input_ids"] if isinstance(ref, Mapping) else ref
                assert ids == list(ref), f"{r['id']}: định dạng tự viết lệch với chat template"
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append((ids, labels))
    if not out:
        raise ValueError(f"No training examples remain in {path} (max_len={max_len})")
    lens = sorted(len(x[0]) for x in out)
    n_lab = sum(sum(1 for t in lab if t != -100) for _, lab in out)
    print(f"{path.name}: {len(out)} mẫu (bỏ {dropped} dài hơn {max_len}), token p50 {lens[len(lens) // 2]}, "
          f"max {lens[-1]}, tổng {sum(lens)}, có nhãn {n_lab}", flush=True)
    return out


def loss_fn(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE trên các vị trí có nhãn; chỉ chiếu lm_head ở đó."""
    causal = model.get_base_model()
    hidden = causal.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    target = labels[0, 1:]
    pos = (target != -100).nonzero(as_tuple=True)[0]
    logits = causal.lm_head(hidden[pos]).float()
    return F.cross_entropy(logits, target[pos])


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
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--max-steps", type=int, default=None, help="dừng sớm sau N bước tối ưu (đo tốc độ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto", help="auto: bf16 nếu GPU hỗ trợ thật, không thì fp16")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--local-files-only", action="store_true", help="never download a model/tokenizer")
    ap.add_argument("--inspect-only", action="store_true", help="CPU/meta: print/save resolved LoRA names, no data/GPU")
    a = ap.parse_args(argv)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = REPO / a.out
    config = AutoConfig.from_pretrained(a.base, local_files_only=a.local_files_only)
    if config.model_type == "qwen3_5":
        raise ValueError("Export text-only first: --base models/qwen35-4b-text")
    if os.environ.get('Q35_ENCODE_CHECK'):
        tok = AutoTokenizer.from_pretrained(a.base, local_files_only=a.local_files_only)
        load(tok, REPO / a.data, a.max_len, native_no_thinking=config.model_type == 'qwen3_5_text')
        return
    if a.inspect_only:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        names = lora_modules(model)
        out.mkdir(parents=True, exist_ok=True)
        (out / "lora_modules.json").write_text(json.dumps(names, indent=2) + "\n")
        print("\n".join(names))
        print(f"{len(names)} Linear modules; {sum(p.numel() for p in model.parameters())} base parameters")
        return
    from peft import LoraConfig, get_peft_model
    from peft.helpers import disable_input_dtype_casting

    if not torch.cuda.is_available():
        raise RuntimeError("Training needs an explicitly authorized CUDA GPU; use --inspect-only for CPU")
    if a.epochs <= 0 or a.grad_accum <= 0 or (a.max_steps is not None and a.max_steps <= 0):
        raise ValueError("epochs, grad-accum and max-steps must be positive")
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=a.local_files_only)
    data = load(tok, REPO / a.data, a.max_len, native_no_thinking=config.model_type == "qwen3_5_text")

    if a.dtype == "auto":  # is_bf16_supported() trả True cả khi chỉ giả lập (T4), nên xét thẳng compute capability
        a.dtype = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    # init_scale nhỏ: mức mặc định 2^16 làm tràn và bỏ mất vài bước đầu, đáng kể khi cả lượt chỉ khoảng 60 bước
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, enabled=a.dtype == "fp16")
    print(f"{torch.cuda.get_device_name()}: {a.dtype}" + (", GradScaler bật" if scaler.is_enabled() else ""), flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, attn_implementation="sdpa",
                                                  local_files_only=a.local_files_only).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    names = lora_modules(model)
    print("LoRA modules:\n" + "\n".join(names), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=names, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    a.chat_template = "prompts/qwen35_scaffold.jinja" if config.model_type == "qwen3_5_text" else None
    a.resolved_lora_modules = names

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total = math.ceil(len(data) * a.epochs / a.grad_accum)
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    order: list[int] = []
    while len(order) < total * a.grad_accum:
        ep = list(range(len(data)))
        random.shuffle(ep)
        order += ep
    model.train()
    t0, toks, step, bad = time.time(), 0, 0, 0
    run_loss, run_n = 0.0, 0
    for k, i in enumerate(order[: total * a.grad_accum]):
        ids, labels = data[i]
        ids_t = torch.tensor([ids], device="cuda")
        lab_t = torch.tensor([labels], device="cuda")
        with torch.autocast("cuda", dtype=dtype), disable_input_dtype_casting(model):
            loss = loss_fn(model, ids_t, lab_t)
        toks += len(ids)
        if torch.isfinite(loss):
            scaler.scale(loss / a.grad_accum).backward()
            run_loss += loss.item()
            run_n += 1
        else:  # fp16 tràn ở forward: bỏ mẫu này thay vì làm hỏng cả bước
            bad += 1
        if (k + 1) % a.grad_accum:
            continue
        if run_n:  # cả cửa sổ đều tràn thì không có gradient, GradScaler sẽ báo lỗi nếu vẫn step
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        el = time.time() - t0
        print(f"bước {step}/{total} loss {run_loss / max(1, run_n):.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"{toks / el:.0f} tok/s, còn {el / step * (total - step) / 60:.0f} phút, "
              f"bộ nhớ đỉnh {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
              + (f", bỏ {bad} mẫu loss không hữu hạn" if bad else ""), flush=True)
        run_loss, run_n = 0.0, 0
        if bad > len(data) // 10:
            sys.exit(f"quá nhiều mẫu loss không hữu hạn ({bad}): thử --dtype bf16 hoặc lr nhỏ hơn")

    adapter = out.with_name(out.name + "-lora")
    model.save_pretrained(adapter)
    (adapter / "train_args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    print(f"adapter → {adapter}", flush=True)
    tok.save_pretrained(adapter)
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
