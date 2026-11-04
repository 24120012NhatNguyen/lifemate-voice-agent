"""Fine-tune ChunkFormer CTC (khanhld/chunkformer-ctc-large-vie) trên audio `train_rest` của LifeMate.

    .venv/bin/python scripts/finetune_chunkformer.py --out models/chunkformer-ctc-large-vie-ft

Giữ checkpoint có WER dev thấp nhất trong --out, cùng dạng thư mục HF gốc (config.yaml, global_cmvn, vocab.txt,
pytorch_model.bin), để adapter `chunkformer` nạp thẳng bằng `model: models/...`.
Mô hình gốc theo giấy phép CC BY-NC 4.0: checkpoint fine-tune cũng phi thương mại, phải hỏi BTC trước khi nộp.

Nhãn: chữ thường, số đọc thành chữ, bỏ dấu câu (bộ từ vựng 6.992 mảnh của mô hình không có chữ số và chữ hoa).
Bộ tách mảnh: khớp tiền tố dài nhất trong vocab.txt (repo không kèm file sentencepiece); hầu hết âm tiết là một mảnh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from chunkformer import ChunkFormerModel
from huggingface_hub import snapshot_download

from lifemate.asr.metrics import norm
from lifemate.asr.train_data import ctc_text, utterances

BASE = "khanhld/chunkformer-ctc-large-vie"
DECODE = dict(chunk_size=64, left_context_size=128, right_context_size=128)  # như adapter `chunkformer`


class Tokenizer:
    def __init__(self, vocab_path: Path):
        self.ids = {}
        for line in vocab_path.open(encoding="utf-8"):
            tok, idx = line.rstrip("\n").rsplit(" ", 1)
            self.ids[tok] = int(idx)
        self.maxlen = max(len(t) for t in self.ids)
        self.unk = self.ids["<unk>"]

    def encode(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            rest = "▁" + word
            while rest:
                for n in range(min(len(rest), self.maxlen), 0, -1):
                    if rest[:n] in self.ids:
                        out.append(self.ids[rest[:n]])
                        rest = rest[n:]
                        break
                else:
                    out.append(self.unk)
                    rest = rest[1:]
        return out


def fbank(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 16000, path
    x = torch.from_numpy(wav * 32768).unsqueeze(0)  # thang int16 như pydub ở bước suy luận
    return kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, energy_floor=0.0,
                       sample_frequency=16000)


def spec_aug(x: torch.Tensor, num_t=2, num_f=2, max_t=50, max_f=10) -> torch.Tensor:
    y = x.clone()
    for _ in range(num_t):
        s = random.randint(0, y.size(0) - 1)
        y[s:s + random.randint(1, max_t)] = 0
    for _ in range(num_f):
        s = random.randint(0, y.size(1) - 1)
        y[:, s:s + random.randint(1, max_f)] = 0
    return y


def batches(items: list[dict], max_frames: int, max_utts: int) -> list[list[dict]]:
    items = sorted(items, key=lambda u: u["feat"].size(0))
    out, cur, longest = [], [], 0
    for u in items:
        longest = max(longest, u["feat"].size(0))
        if cur and (longest * (len(cur) + 1) > max_frames or len(cur) == max_utts):
            out.append(cur)
            cur, longest = [], u["feat"].size(0)
        cur.append(u)
    return out + [cur] if cur else out


def collate(batch: list[dict], augment: bool) -> dict:
    feats = [spec_aug(u["feat"]) if augment else u["feat"] for u in batch]
    return {
        "feats": torch.nn.utils.rnn.pad_sequence(feats, batch_first=True),
        "feats_lengths": torch.tensor([f.size(0) for f in feats], dtype=torch.int32),
        "target": torch.nn.utils.rnn.pad_sequence([torch.tensor(u["ids"]) for u in batch], batch_first=True,
                                                  padding_value=-1),
        "target_lengths": torch.tensor([len(u["ids"]) for u in batch], dtype=torch.int32),
    }


def dev_wer(model, dev: list[dict]) -> float:
    model.eval()
    torch.cuda.empty_cache()
    hyps = []
    for i in range(0, len(dev), 32):  # từng nhóm như run_asr, không dồn cả tập dev vào một lượt
        hyps += model.batch_decode(audio_paths=[u["path"] for u in dev[i:i + 32]], **DECODE)
    return 100 * jiwer.wer([norm(u["text"], True) or "_" for u in dev], [norm(h, True) or "_" for h in hyps])


def save(model, src: Path, out: Path, info: dict):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "global_cmvn", "vocab.txt"):
        shutil.copy(src / name, out / name)
    torch.save(model.model.state_dict(), out / "pytorch_model.bin")
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/chunkformer-ctc-large-vie-ft")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.1, help="tỉ lệ số bước khởi động")
    ap.add_argument("--max-frames", type=int, default=9000, help="số khung fbank (10 ms) tối đa mỗi batch, đã đệm")
    ap.add_argument("--max-utts", type=int, default=24, help="số câu tối đa mỗi batch: bộ nhớ tăng theo số câu "
                    "(mỗi câu ngắn vẫn chiếm trọn một khối chunk 256), 24 câu khoảng 6 GB")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # đọc khi CUDA cấp phát lần đầu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    src = Path(snapshot_download(BASE))
    tok = Tokenizer(src / "vocab.txt")
    train, dev = utterances("train"), utterances("dev")
    t0 = time.time()
    for u in train:
        u["feat"], u["ids"] = fbank(u["path"]), tok.encode(ctc_text(u["text"]))
    n_unk = sum(i == tok.unk for u in train for i in u["ids"])
    print(f"{len(train)} câu train, {len(dev)} câu dev; fbank {time.time() - t0:.0f} giây; <unk> {n_unk}", flush=True)

    model = ChunkFormerModel.from_pretrained(BASE).cuda()
    best = dev_wer(model, dev)
    print(f"epoch 0: WER dev {best:.2f}", flush=True)
    info = {"base": BASE, "args": vars(args), "train_utts": len(train), "dev_utts": len(dev), "dev_wer": {0: round(best, 2)}}

    opt = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(batches(train, args.max_frames, args.max_utts))
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    best_epoch, out = 0, Path(args.out)
    for ep in range(1, args.epochs + 1):
        model.train()
        bs = batches(train, args.max_frames, args.max_utts)
        random.shuffle(bs)
        tot, t0 = 0.0, time.time()
        for b in bs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.model(collate(b, augment=True), torch.device("cuda"))["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        wer = dev_wer(model, dev)
        info["dev_wer"][ep] = round(wer, 2)
        print(f"epoch {ep}: loss {tot / len(bs):.2f}, WER dev {wer:.2f}, {time.time() - t0:.0f} giây", flush=True)
        if wer < best:
            best, best_epoch = wer, ep
            save(model, src, out, {**info, "best_epoch": ep})
    if best_epoch == 0:
        print("không epoch nào tốt hơn mô hình gốc trên dev; không lưu checkpoint")
    else:
        save_info = json.loads((out / "train_info.json").read_text())
        save_info["dev_wer"] = info["dev_wer"]
        (out / "train_info.json").write_text(json.dumps(save_info, ensure_ascii=False, indent=2))
        print(f"tốt nhất: epoch {best_epoch}, WER dev {best:.2f} -> {out}")


if __name__ == "__main__":
    main()
