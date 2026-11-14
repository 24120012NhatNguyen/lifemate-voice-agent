"""So bản chép của một Whisper đã gộp với bản gốc + adapter tách lại (scripts/whisper_lora_extract.py) trên vài mẫu val.
Cần GPU vài phút: chạy qua khoá GPU chung. Chỉ in số mẫu khác và id, không in bản chép.

    flock <gpu.lock> .venv/bin/python scripts/check_asr_adapter.py --merged models/phowhisper-large-ft \
        --base models/phowhisper-large-base --adapter models/phowhisper-large-ft-lora --num-beams 4 --n-best 4 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import time

from lifemate import data, splits
from lifemate.asr.base import Request
from lifemate.registry import build


def transcribe(cfg: dict, reqs: list) -> tuple[list, float]:
    engine = build("asr", cfg)
    t0 = time.perf_counter()
    out = []
    for i in range(0, len(reqs), 32):
        out += engine.transcribe(reqs[i:i + 32])
    sec = time.perf_counter() - t0
    engine.close()
    return [(r.text, tuple(r.alts or ())) for r in out], sec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--subset", default="val_app")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--n-best", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    pool = splits.select("train", a.subset, a.limit)
    reqs = [Request(f"{s['id']}/{t}", p, s, t, "") for s in pool for t, p in data.audio_paths("train", s).items()]
    common = {"name": "hf_whisper", "num_beams": a.num_beams, "n_best": a.n_best, "batch_size": a.batch_size}
    ref, t_ref = transcribe({**common, "model": a.merged}, reqs)
    new, t_new = transcribe({**common, "model": a.base, "adapter": a.adapter}, reqs)
    diff_top = [r.key for r, x, y in zip(reqs, ref, new) if x[0] != y[0]]
    diff_any = [r.key for r, x, y in zip(reqs, ref, new) if x != y]
    print(json.dumps({"merged": a.merged, "adapter": a.adapter, "samples": len(pool), "utterances": len(reqs),
                      "num_beams": a.num_beams, "n_best": a.n_best,
                      "utterances_top_text_differs": len(diff_top), "utterances_any_alt_differs": len(diff_any),
                      "samples_affected": len({k.split("/")[0] for k in diff_any}), "ids": diff_any[:12],
                      "sec_merged": round(t_ref, 1), "sec_adapter": round(t_new, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
