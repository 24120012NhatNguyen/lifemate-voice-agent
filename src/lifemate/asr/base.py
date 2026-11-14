"""Giao diện ASR và tầng chạy ASR có bộ nhớ đệm.

Hợp đồng đầu ra (proposal mục 9, `transcripts.jsonl`), mỗi mẫu một dòng:
    {"id", "final": lời lượt cuối, "turns": {"u1": ..., "u2": ...}, "alts": {"final": [...]}, "sec": giây chạy ASR}

Transcript thô được lưu theo khoá của config ASR trong cache/asr/<split>/; đổi hậu xử lý không phải chạy lại ASR.
Biến LIFEMATE_CACHE đổi thư mục cache (Docker trỏ nó vào thư mục tạm để mỗi lượt nghe lại từ đầu).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
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


SILENCE_SAMPLES = 8000  # nửa giây im lặng thay cho audio không đọc được


def load_audio(path: Path) -> np.ndarray:
    """Audio mono 16 kHz. File hỏng, thiếu hoặc rỗng trả nửa giây im lặng (ASR chép ra câu rỗng) thay vì ném lỗi:
    một file xấu trong tập thi không được làm sập cả pha ASR, tức mất cả bài nộp."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000 and len(wav):
            import torch
            import torchaudio.functional as F
            wav = F.resample(torch.from_numpy(wav), sr, 16000).numpy()
    except Exception as e:  # noqa: BLE001 - mọi lỗi đọc / giải mã đều xử lý như nhau
        print(f"CẢNH BÁO: không đọc được audio {path}: {type(e).__name__}: {e}", flush=True)
        return np.zeros(SILENCE_SAMPLES, dtype="float32")
    if len(wav) < 400:  # dưới 25 ms: không đủ một khung đặc trưng
        print(f"CẢNH BÁO: audio rỗng {path}", flush=True)
        return np.zeros(SILENCE_SAMPLES, dtype="float32")
    return wav


@lru_cache(maxsize=None)
def app_phrases(split: str) -> dict[str, list[str]]:
    """Từ vựng tiếng Việt của mỗi app trong catalog: tên app, các cụm trong mô tả app và mô tả tool."""
    def phrases(text):  # bỏ định danh kiểu playlist_id, app.tool và "bắt buộc: x"
        parts = [p.strip(" .") for p in re.split(r"[,;/()]| và | hoặc ", text or "")]
        return [p for p in parts if 2 <= len(p) <= 40 and not re.search(r"[_:]|\w\.\w", p)]
    out = {}
    for a in data.catalog(split)["apps"]:
        words = [a.get("name_vi", "")] + phrases(a.get("description"))
        out[a["id"]] = words + [w for t in a["tools"] for w in phrases(t.get("description"))]
    return out


@lru_cache(maxsize=None)
def _app_rank() -> dict[str, int]:
    """App ít được cài đứng trước: từ vựng của app phổ biến (thời tiết, lịch) mô hình fine-tune đã quen.

    Luôn đếm trên train, đúng thứ tự lúc train ASR (finetune_whisper.py gọi bias_hint với split train), nên mỗi mẫu test
    được mồi độc lập, không dùng thống kê chéo các mẫu test; app chưa có ở train đứng đầu (hạng -1). Docker không mang
    train: đọc $LIFEMATE_APP_RANK, file JSON do scripts/export_app_rank.py ghi cạnh model ASR.
    """
    f = os.environ.get("LIFEMATE_APP_RANK")
    if f:
        try:
            return json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # file thiếu hay hỏng không được làm sập pha ASR
            print(f"CẢNH BÁO: không đọc được $LIFEMATE_APP_RANK ({e}); mồi app giữ thứ tự installed_apps", flush=True)
            return {}
    try:
        n = Counter(app for s in data.samples("train") for app in s["context"]["installed_apps"])
    except FileNotFoundError:
        print("CẢNH BÁO: không có train lẫn $LIFEMATE_APP_RANK; mồi app giữ thứ tự installed_apps", flush=True)
        return {}
    return {app: i for i, (app, _) in enumerate(sorted(n.items(), key=lambda kv: (kv[1], kv[0])))}


def proper_names(text: str) -> list[str]:
    """Cụm từ hai chữ trở lên viết hoa đầu mỗi chữ ("Sếp Dũng", "Phạm Hùng"). Không dùng lớp [À-Ỹ]: lớp đó chứa
    cả chữ thường có dấu, nên bắt nhầm mảnh như "ền điện", "ủa Sơn Tùng"."""
    out = []
    for run in re.findall(r"[^\W\d_]+(?: [^\W\d_]+)+", text):
        cur = []
        for w in run.split() + [""]:
            if w[:1].isupper():
                cur.append(w)
                continue
            if len(cur) >= 2:
                out.append(" ".join(cur))
            cur = []
    return out


def bias_hint(sample: dict, mode: str, split: str = "train") -> str:
    """Chuỗi mồi cho Whisper: tên trong hồ sơ, rồi tên riêng xuất hiện trong lịch sử của assistant.

    `profile_history_apps` nối thêm từ vựng của các app đã cài (app_phrases, app hiếm trước). Chuỗi này dài hơn
    giới hạn mồi của Whisper; adapter cắt theo số token (`max_prompt_tokens`), nên tên riêng luôn còn.
    """
    if mode == "none":
        return ""
    p = sample["context"].get("user_profile") or {}  # khoá có thể mang giá trị null trong dữ liệu thi
    words = [str(p.get("name") or ""), str(p.get("city") or "")]
    if mode.startswith("profile_history"):
        blob = " ".join((t.get("text") or "") + json.dumps(t.get("tool_results") or [], ensure_ascii=False)
                        for t in sample.get("history") or [] if t.get("role") == "assistant")
        words += proper_names(blob)
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    out = out[:30]
    if mode.endswith("_apps"):
        rank, vocab = _app_rank(), app_phrases(split)
        seen = {w.lower() for w in out}
        for app in sorted(sample["context"]["installed_apps"], key=lambda a: rank.get(a, -1)):
            for w in vocab.get(app, []):
                if w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
    return ", ".join(out)


def _plain(text: str) -> str:
    """Bản chép bỏ dấu câu và hoa thường, để hai ASR chỉ khác dấu phẩy không thành hai phương án."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def run_asr(split: str, samples: list[dict], cfg: dict, log=print) -> dict[str, dict]:
    """Chạy (hoặc lấy từ cache) transcript thô cho các mẫu, rồi áp các bước hậu xử lý trong cfg["asr_post"]."""
    asr_cfg = dict(cfg.get("asr") or {"name": "gold"})
    with_history = asr_cfg.pop("history", True)
    hint_mode = asr_cfg.pop("hint", "none")
    merge = asr_cfg.pop("merge_with", None)  # ASR thứ hai: bản chép lượt cuối của nó thành phương án đầu (không đổi khoá cache)
    key = stage_key(split, asr_cfg, with_history, hint_mode)
    cache = Path(os.environ.get("LIFEMATE_CACHE", REPO / "cache")) / "asr" / split / f"{asr_cfg['name']}_{key}.jsonl"
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
                    hint = bias_hint(s, hint_mode, split)
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
    if merge:
        other = run_asr(split, samples, {"asr": {"name": asr_cfg["name"], "history": with_history, **merge}}, log)
        for sid, row in out.items():
            # thời gian chép của ASR thứ hai cũng là thời gian của mẫu: time_budget và luật 30 giây tính cả hai lượt nghe
            row["sec"] = round(row.get("sec", 0.0) + other[sid].get("sec", 0.0), 3)
            extra = other[sid]["final"]
            alts = list((row.get("alts") or {}).get("final") or [])
            if extra and _plain(extra) not in {_plain(x) for x in [row["final"], *alts]}:
                row["alts"] = {**(row.get("alts") or {}), "final": [extra, *alts]}
    for post in cfg.get("asr_post") or []:
        step = build("asr_post", post)
        for s in samples:
            out[s["id"]] = step.apply(s, out[s["id"]])
    return out
