"""Giao diện ASR và tầng chạy ASR có bộ nhớ đệm.

Hợp đồng đầu ra (proposal mục 9, `transcripts.jsonl`), mỗi mẫu một dòng:
    {"id", "final": lời lượt cuối, "turns": {"u1": ..., "u2": ...}, "alts": {"final": [...]}, "sec": giây chạy ASR}

Transcript thô được lưu theo khoá của config ASR trong cache/asr/<split>/; đổi hậu xử lý không phải chạy lại ASR.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import data
from ..config import REPO, stage_key
from ..registry import build


@dataclass
class Request:
    key: str  # "<id>/final" hoặc "<id>/u1"
    path: Path
    sample: dict
    turn: str
    hint: str = ""


@dataclass
class Result:
    text: str
    alts: list[str] = field(default_factory=list)
    conf: float | None = None


class ASR:
    """Mọi phương án ASR kế thừa lớp này và cài `transcribe`."""

    def transcribe(self, reqs: list[Request]) -> list[Result]:
        raise NotImplementedError

    def close(self):
        """Giải phóng GPU sau khi chạy xong tầng ASR."""


def load_audio(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio.functional as F
        wav = F.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


def bias_hint(sample: dict, mode: str) -> str:
    """Chuỗi mồi cho Whisper: tên trong hồ sơ, rồi tên riêng xuất hiện trong lịch sử của assistant."""
    if mode == "none":
        return ""
    p = sample["context"].get("user_profile", {})
    words = [p.get("name", ""), p.get("city", "")]
    if mode == "profile_history":
        blob = " ".join(t.get("text", "") + json.dumps(t.get("tool_results") or [], ensure_ascii=False)
                        for t in sample.get("history", []) if t.get("role") == "assistant")
        words += re.findall(r"(?:[A-ZĐÀ-Ỹ][\wÀ-ỹ]+)(?: [A-ZĐÀ-Ỹ][\wÀ-ỹ]+)+", blob)
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return ", ".join(out[:30])


def run_asr(split: str, samples: list[dict], cfg: dict, log=print) -> dict[str, dict]:
    """Chạy (hoặc lấy từ cache) transcript thô cho các mẫu, rồi áp các bước hậu xử lý trong cfg["asr_post"]."""
    asr_cfg = dict(cfg.get("asr") or {"name": "gold"})
    with_history = asr_cfg.pop("history", True)
    hint_mode = asr_cfg.pop("hint", "none")
    key = stage_key(split, asr_cfg, with_history, hint_mode)
    cache = REPO / "cache" / "asr" / split / f"{asr_cfg['name']}_{key}.jsonl"
    done: dict[str, dict] = {}
    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [s for s in samples if s["id"] not in done]
    if todo:
        log(f"ASR {asr_cfg['name']}: {len(todo)} mẫu chưa có trong cache {cache.name}")
        engine = build("asr", asr_cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as fout:
            for i in range(0, len(todo), 32):
                chunk = todo[i:i + 32]
                reqs = []
                for s in chunk:
                    hint = bias_hint(s, hint_mode)
                    for turn, path in data.audio_paths(split, s).items():
                        if turn == "final" or with_history:
                            reqs.append(Request(f"{s['id']}/{turn}", path, s, turn, hint))
                t0 = time.perf_counter()
                res = engine.transcribe(reqs)
                per = (time.perf_counter() - t0) / max(1, len(chunk))
                by: dict[str, dict] = {s["id"]: {"id": s["id"], "final": "", "turns": {}, "alts": {}, "sec": per} for s in chunk}
                for rq, rs in zip(reqs, res):
                    row = by[rq.sample["id"]]
                    if rq.turn == "final":
                        row["final"] = rs.text
                    else:
                        row["turns"][rq.turn] = rs.text
                    if rs.alts:
                        row["alts"][rq.turn] = rs.alts
                for row in by.values():
                    done[row["id"]] = row
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n = min(i + 32, len(todo))
                if n == len(todo) or (i // 32) % 8 == 7:
                    log(f"  ASR {n}/{len(todo)}")
        engine.close()
    out = {s["id"]: dict(done[s["id"]]) for s in samples}
    for post in cfg.get("asr_post") or []:
        step = build("asr_post", post)
        for s in samples:
            out[s["id"]] = step.apply(s, out[s["id"]])
    return out
