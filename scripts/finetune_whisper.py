"""Fine-tune Whisper (mặc định vinai/PhoWhisper-large) bằng LoRA trên audio `train_rest` của LifeMate.

    .venv/bin/python scripts/finetune_whisper.py --model vinai/PhoWhisper-large --out models/phowhisper-large-ft
    .venv/bin/python scripts/finetune_whisper.py --out models/phowhisper-large-ft2 --epochs 6 \
        --speed 0.9,1.0,1.1 --hint profile_history_apps --hint-prob 0.5
    .venv/bin/python scripts/finetune_whisper.py --data all --out models/phowhisper-large-ft2-all --epochs <epoch tốt nhất> ...

LoRA trên mọi lớp chiếu attention và FFN của encoder lẫn decoder; mô hình gốc để bf16 và đóng băng nên vừa 16 GB.
Mỗi epoch đo WER dev và giữ adapter tốt nhất. Cuối cùng gộp adapter vào mô hình gốc (fp32 trên CPU, lưu fp16) thành
checkpoint HF đầy đủ, nên adapter `hf_whisper` nạp thẳng bằng `model: models/...`, không cần peft lúc suy luận.

Nhãn là `audio_text` nguyên dạng (hoa thường, dấu câu, chữ số) để lời ra giống transcript gold mà agent đã quen;
PhoWhisper gốc viết thường và đọc số thành chữ.

--speed: đổi tốc độ ngẫu nhiên từng câu (lấy mẫu lại như speed perturbation của Kaldi, đổi cả cao độ).
--hint: với xác suất --hint-prob, đặt chuỗi mồi `bias_hint` của mẫu (đúng chuỗi adapter dùng khi chạy với `asr.hint`)
làm phần <|startofprev|> trước nhãn; token mồi không tính loss. Mô hình học dùng mồi mà vẫn quen chạy không mồi.
--data all: học cả 1.545 mẫu (thêm val_random, val_app) cho bài nộp; không còn dev, lưu epoch cuối, nên đặt --epochs
bằng epoch tốt nhất của lượt train_rest cùng công thức (hoặc --save-every-epoch rồi chọn). Bản này không đo được trên val.

Kaggle T4 (kaggle/run_asr.py): --dtype float16 (T4 không có bf16, dùng GradScaler), --no-merge để chỉ giữ adapter
(khoảng 230 MB thay vì 3 GB), rồi gộp ở máy nhà trên CPU:
    .venv/bin/python scripts/finetune_whisper.py --merge <thư mục adapter> --out models/phowhisper-large-ft2
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import jiwer
import torch
import torchaudio.functional as AF
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, WhisperForConditionalGeneration

from lifemate import data
from lifemate.asr.base import bias_hint, load_audio
from lifemate.asr.metrics import norm
from lifemate.asr.train_data import utterances

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def prompt_ids(proc, sample_id: str, mode: str, max_tokens: int) -> list[int]:
    """Như adapter hf_whisper khi chạy với `asr.hint`: <|startofprev|> + mồi, cắt ở max_tokens."""
    hint = bias_hint(data.by_id("train")[sample_id], mode, "train")
    return list(proc.get_prompt_ids(hint))[:max_tokens] if hint else []


class Collator:
    """Đầu vào decoder: [mồi] <|startoftranscript|><|vi|><|transcribe|><|notimestamps|> lời <|endoftext|>."""

    def __init__(self, proc, speeds=(1.0,), hint="none", hint_prob=0.0, max_prompt_tokens=200):
        self.proc, self.tok = proc, proc.tokenizer
        self.prefix, self.eot = self.tok.prefix_tokens, self.tok.eos_token_id
        self.speeds, self.hint, self.hint_prob, self.max_prompt_tokens = speeds, hint, hint_prob, max_prompt_tokens

    def wav(self, u):
        wav = load_audio(Path(u["path"]))
        f = random.choice(self.speeds)  # DataLoader gieo lại `random` cho từng worker
        return wav if f == 1.0 else AF.resample(torch.from_numpy(wav), int(16000 * f), 16000).numpy()

    def __call__(self, batch):
        feats = self.proc.feature_extractor([self.wav(u) for u in batch], sampling_rate=16000,
                                            return_tensors="pt").input_features
        inputs, labels = [], []
        for u in batch:
            prompt = []
            if self.hint != "none" and random.random() < self.hint_prob:
                prompt = prompt_ids(self.proc, u["key"].split("/")[0], self.hint, self.max_prompt_tokens)
            full = prompt + self.prefix + self.tok(u["text"], add_special_tokens=False).input_ids + [self.eot]
            inputs.append(torch.tensor(full[:-1]))
            labels.append(torch.tensor([-100] * len(prompt) + full[len(prompt) + 1:]))  # mồi và SOT không tính loss
        pad = torch.nn.utils.rnn.pad_sequence
        return feats, pad(inputs, batch_first=True, padding_value=self.eot), pad(labels, batch_first=True,
                                                                                padding_value=-100)


def dev_wer(model, proc, dev: list[dict], hint="none", max_prompt_tokens=200, dtype=torch.bfloat16,
            batch_size=16) -> float:
    model.eval()
    hyps = []
    step = 1 if hint != "none" else batch_size  # mồi khác nhau từng câu: chạy lẻ
    for i in range(0, len(dev), step):
        part = dev[i:i + step]
        wavs = [load_audio(Path(u["path"])) for u in part]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        longest = max(len(w) for w in wavs) / 16000
        kw = {}
        ids = prompt_ids(proc, part[0]["key"].split("/")[0], hint, max_prompt_tokens) if hint != "none" else []
        if ids:
            kw["prompt_ids"] = torch.tensor(ids, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = model.generate(input_features=feats.cuda().to(dtype), language="vi", task="transcribe",
                                 num_beams=1, max_new_tokens=min(220, int(longest * 10) + 16), use_cache=True, **kw)
        texts = [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True)]
        if ids:  # như adapter: bỏ phần mồi nếu transformers giữ nó ở đầu chuỗi
            p = proc.decode(ids[1:], skip_special_tokens=True).strip()
            texts = [t[len(p):].strip() if t.startswith(p) else t for t in texts]
        hyps += texts
    model.train()
    return 100 * jiwer.wer([norm(u["text"], True) or "_" for u in dev], [norm(h, True) or "_" for h in hyps])


def merge(base: str, adapter: Path, out: Path, info: dict | None = None) -> None:
    """Gộp adapter vào mô hình gốc fp32 trên CPU, lưu fp16 kèm processor: adapter hf_whisper nạp thẳng được."""
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.to(torch.float16).save_pretrained(out)
    AutoProcessor.from_pretrained(base).save_pretrained(out)
    if info is None and (adapter / "train_info.json").exists():
        info = json.loads((adapter / "train_info.json").read_text())
    if info is not None:
        (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out", default="models/phowhisper-large-ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.1, help="tỉ lệ số bước khởi động")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data", choices=["train_rest", "all"], default="train_rest")
    ap.add_argument("--speed", default="1.0", help="các hệ số tốc độ, vd. 0.9,1.0,1.1")
    ap.add_argument("--hint", default="none", help="chế độ bias_hint khi học: none | profile_history | "
                    "profile_history_apps")
    ap.add_argument("--hint-prob", type=float, default=0.5)
    ap.add_argument("--max-prompt-tokens", type=int, default=200)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--no-merge", action="store_true", help="chỉ lưu adapter (out/adapter), không gộp")
    ap.add_argument("--save-every-epoch", action="store_true", help="lưu thêm out/adapter-ep<N> sau mỗi epoch")
    ap.add_argument("--merge", type=Path, help="chỉ gộp adapter này vào --model rồi ghi ra --out")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình nạp audio")
    ap.add_argument("--limit", type=int, default=0, help="chạy thử: chỉ lấy N câu học, N/4 câu dev")
    args = ap.parse_args()
    if args.merge:
        merge(args.model, args.merge, Path(args.out))
        print(f"đã gộp {args.merge} → {args.out}")
        return
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe", predict_timestamps=False)
    if args.data == "all":
        train, dev = utterances("all", ("train_rest", "val_random", "val_app")), []
    else:
        train, dev = utterances("train"), utterances("dev")
    if args.limit:
        train, dev = random.Random(0).sample(train, args.limit), dev[:max(1, args.limit // 4)] if dev else []
    collate = Collator(proc, tuple(float(x) for x in args.speed.split(",")), args.hint, args.hint_prob,
                       args.max_prompt_tokens)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.workers,
                        drop_last=True)

    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).cuda()
    model.config.use_cache = False
    model.config.apply_spec_augment, model.config.mask_time_prob = True, 0.05
    model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 10
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, bias="none"))
    model.print_trainable_parameters()

    info = {"base": args.model, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "train_utts": len(train), "dev_utts": len(dev),
            "dev_wer": {}, "dev_wer_hint": {}, "best_epoch": 0}

    def evaluate(ep: int) -> float:
        """WER dev không mồi (để chọn epoch), và có mồi nếu học với --hint. Không có dev (--data all): vô cực."""
        if not dev:
            return float("inf")
        wer = info["dev_wer"][ep] = round(dev_wer(model, proc, dev, dtype=dtype), 2)
        if args.hint != "none":
            info["dev_wer_hint"][ep] = round(dev_wer(model, proc, dev, args.hint, args.max_prompt_tokens, dtype), 2)
        print(f"epoch {ep}: WER dev {wer}, có mồi {info['dev_wer_hint'].get(ep)}", flush=True)
        return wer

    best = evaluate(0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(loader) // args.grad_accum)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)  # fp16 cần nhân loss để grad không tràn dưới
    model.train()
    for ep in range(1, args.epochs + 1):
        tot, t0 = 0.0, time.time()
        for i, (feats, dec_in, labels) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=dtype):
                loss = model(input_features=feats.cuda().to(dtype), decoder_input_ids=dec_in.cuda(),
                             labels=labels.cuda()).loss
            scaler.scale(loss / args.grad_accum).backward()
            tot += loss.item()
            if i % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 100 == 0:
                print(f"  epoch {ep} bước {i}/{len(loader)}: loss {tot / i:.3f}, {time.time() - t0:.0f} giây", flush=True)
        print(f"epoch {ep}: loss {tot / len(loader):.3f}, {time.time() - t0:.0f} giây", flush=True)
        wer = evaluate(ep)
        if args.save_every_epoch:
            model.save_pretrained(out / f"adapter-ep{ep}")
        if wer < best or (not dev and ep == args.epochs):
            best, info["best_epoch"] = wer, ep
            model.save_pretrained(out / "adapter")
        (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    if info["best_epoch"] == 0:
        print("không epoch nào tốt hơn mô hình gốc trên dev; không lưu checkpoint")
        return
    (out / "adapter" / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    if not args.no_merge:
        model.cpu()
        torch.cuda.empty_cache()
        merge(args.model, out / "adapter", out, info)
        shutil.rmtree(out / "adapter")
    print(f"tốt nhất: epoch {info['best_epoch']}, WER dev {info['dev_wer'].get(info['best_epoch'])} -> {out}")


if __name__ == "__main__":
    main()
