"""Fine-tune Whisper (mặc định vinai/PhoWhisper-large) bằng LoRA trên audio `train_rest` của LifeMate.

    .venv/bin/python scripts/finetune_whisper.py --model vinai/PhoWhisper-large --out models/phowhisper-large-ft

LoRA trên mọi lớp chiếu attention và FFN của encoder lẫn decoder; mô hình gốc để bf16 và đóng băng nên vừa 16 GB.
Mỗi epoch đo WER dev và giữ adapter tốt nhất. Cuối cùng gộp adapter vào mô hình gốc (fp32 trên CPU, lưu fp16) thành
checkpoint HF đầy đủ, nên adapter `hf_whisper` nạp thẳng bằng `model: models/...`, không cần peft lúc suy luận.

Nhãn là `audio_text` nguyên dạng (hoa thường, dấu câu, chữ số) để lời ra giống transcript gold mà agent đã quen;
PhoWhisper gốc viết thường và đọc số thành chữ.
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
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, WhisperForConditionalGeneration

from lifemate.asr.base import load_audio
from lifemate.asr.metrics import norm
from lifemate.asr.train_data import utterances

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


class Collator:
    def __init__(self, proc):
        self.proc = proc
        self.sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, batch):
        feats = self.proc.feature_extractor([load_audio(Path(u["path"])) for u in batch], sampling_rate=16000,
                                            return_tensors="pt").input_features
        lab = self.proc.tokenizer([u["text"] for u in batch], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
        if (labels[:, 0] == self.sot).all():  # mô hình tự chèn <|startoftranscript|> khi dịch phải nhãn
            labels = labels[:, 1:]
        return feats, labels


def dev_wer(model, proc, dev: list[dict], batch_size=16) -> float:
    model.eval()
    hyps = []
    for i in range(0, len(dev), batch_size):
        part = dev[i:i + batch_size]
        wavs = [load_audio(Path(u["path"])) for u in part]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        longest = max(len(w) for w in wavs) / 16000
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(input_features=feats.cuda().to(torch.bfloat16), language="vi", task="transcribe",
                                 num_beams=1, max_new_tokens=min(220, int(longest * 10) + 16), use_cache=True)
        hyps += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    model.train()
    return 100 * jiwer.wer([norm(u["text"], True) or "_" for u in dev], [norm(h, True) or "_" for h in hyps])


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
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe", predict_timestamps=False)
    train, dev = utterances("train"), utterances("dev")
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(proc), num_workers=4,
                        drop_last=True)

    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.config.apply_spec_augment, model.config.mask_time_prob = True, 0.05
    model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 10
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, bias="none"))
    model.print_trainable_parameters()

    best = dev_wer(model, proc, dev)
    print(f"epoch 0: WER dev {best:.2f}", flush=True)
    info = {"base": args.model, "args": vars(args), "train_utts": len(train), "dev_utts": len(dev),
            "dev_wer": {0: round(best, 2)}, "best_epoch": 0}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(loader) // args.grad_accum)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    model.train()
    for ep in range(1, args.epochs + 1):
        tot, t0 = 0.0, time.time()
        for i, (feats, labels) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=feats.cuda().to(torch.bfloat16), labels=labels.cuda()).loss
            (loss / args.grad_accum).backward()
            tot += loss.item()
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 100 == 0:
                print(f"  epoch {ep} bước {i}/{len(loader)}: loss {tot / i:.3f}, {time.time() - t0:.0f} giây", flush=True)
        wer = dev_wer(model, proc, dev)
        info["dev_wer"][ep] = round(wer, 2)
        print(f"epoch {ep}: loss {tot / len(loader):.3f}, WER dev {wer:.2f}, {time.time() - t0:.0f} giây", flush=True)
        if wer < best:
            best, info["best_epoch"] = wer, ep
            model.save_pretrained(out / "adapter")

    if info["best_epoch"] == 0:
        print("không epoch nào tốt hơn mô hình gốc trên dev; không lưu checkpoint")
        return
    del model, opt
    torch.cuda.empty_cache()
    base = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged.to(torch.float16).save_pretrained(out)
    AutoProcessor.from_pretrained(args.model).save_pretrained(out)
    shutil.rmtree(out / "adapter")
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"tốt nhất: epoch {info['best_epoch']}, WER dev {best:.2f} -> {out}")


if __name__ == "__main__":
    main()
