"""Thời gian ASR khi mỗi mẫu chạy một mình (mọi lượt nói của mẫu gộp một lô), để cộng với sec_agent thành thời gian
tuần tự thật của từng mẫu. Lượt nộp chạy ASR theo lô 16 trên cả split, nên sec_asr trong samples.jsonl chỉ là phần
chia đều. Chỉ đo, không ghi cache, không đụng file nộp. Mẫu đầu chạy hai lần, lần đầu để khởi động kernel.

    .venv/bin/python scripts/asr_latency.py --split public_test --out asr_latency.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lifemate import data, splits
from lifemate.asr.base import Request
from lifemate.registry import build


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--model", default="models/phowhisper-large-ft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", default="all", help="tập con của train (val_app, val_random) khi kiểm ngoài public")
    ap.add_argument("--num-beams", type=int, default=1, help="như asr.num_beams của config đang kiểm")
    ap.add_argument("--n-best", type=int, default=1, help="như asr.n_best của config đang kiểm (N-best 4: beam 4, 4 bản)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    engine = build("asr", {"name": "hf_whisper", "model": a.model, "num_beams": a.num_beams, "n_best": a.n_best,
                           "batch_size": 16})
    pool = splits.select(a.split, a.subset, a.limit)
    sec: dict[str, float] = {}
    for s in [pool[0], *pool]:
        reqs = [Request(f"{s['id']}/{t}", p, s, t) for t, p in data.audio_paths(a.split, s).items()]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.transcribe(reqs)
        torch.cuda.synchronize()
        sec[s["id"]] = round(time.perf_counter() - t0, 3)
    a.out.write_text(json.dumps(sec, indent=0), encoding="utf-8")
    xs = sorted(sec.values())
    print(f"{len(xs)} mẫu: p50 {xs[len(xs) // 2]:.2f}s, p99 {xs[int(0.99 * (len(xs) - 1))]:.2f}s, max {xs[-1]:.2f}s")


if __name__ == "__main__":
    main()
