"""Đo ASR trên train: WER, CER của lượt cuối, và độ giữ thực thể (chuỗi trong tham số gold có nghe ra đúng không).

`wer` so chuỗi sau khi hạ chữ thường và bỏ dấu câu. `wer_num` còn đổi mọi số viết bằng chữ số sang chữ ở cả hai phía
("9 giờ" = "chín giờ", "50.000" = "năm mươi nghìn"), để mô hình viết số kiểu khác không bị tính là nghe sai.
"""
from __future__ import annotations

import re

import jiwer

from ..scoring import fold

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _under_1000(n: int, full: bool) -> list[str]:
    h, t, u = n // 100, n // 10 % 10, n % 10
    out = []
    if h or full:
        out += [_DIGITS[h], "trăm"]
    if t == 0:
        if u and (h or full):
            out.append("lẻ")
    elif t == 1:
        out.append("mười")
    else:
        out += [_DIGITS[t], "mươi"]
    if u:
        out.append("mốt" if u == 1 and t > 1 else "lăm" if u == 5 and t > 0 else _DIGITS[u])
    return out


def num_words(n: int) -> str:
    if n == 0:
        return "không"
    out, started = [], False
    for unit, size in (("tỷ", 10**9), ("triệu", 10**6), ("nghìn", 10**3), ("", 1)):
        part = n // size % 1000
        if part:
            out += _under_1000(part, started) + ([unit] if unit else [])
            started = True
    return " ".join(out)


def _spell_numbers(s: str) -> str:
    s = re.sub(r"(\d)[.,](?=\d{3}\b)", r"\1", s)  # 50.000 -> 50000
    s = re.sub(r"(\d{1,2}):(\d{2})", lambda m: f"{m.group(1)} giờ {int(m.group(2))}" if int(m.group(2)) else f"{m.group(1)} giờ", s)
    return re.sub(r"\d+", lambda m: num_words(int(m.group(0))) if len(m.group(0)) <= 12 else m.group(0), s)


def norm(s: str, numbers: bool = False) -> str:
    s = s.lower()
    if numbers:
        s = _spell_numbers(s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def entity_spans(sample) -> list[str]:
    """Giá trị chuỗi trong call gold mà người dùng đã nói ra nguyên văn (tên, truy vấn, địa điểm)."""
    said = fold(sample.get("audio_text", ""))
    out = []
    for c in sample.get("label", {}).get("tool_calls", []):
        for v in (c.get("arguments") or {}).values():
            if isinstance(v, str) and 2 <= len(v) <= 40 and fold(v) in said and not re.match(r"^\d{4}-", v):
                out.append(v)
    return out


def asr_metrics(samples: list[dict], transcripts: dict[str, dict]) -> dict:
    refs, hyps, refs_n, hyps_n, ent_tot, ent_ok, ent_exact = [], [], [], [], 0, 0, 0
    for s in samples:
        if "audio_text" not in s:
            return {}
        hyp = transcripts[s["id"]]["final"]
        refs.append(norm(s["audio_text"]) or "_")
        hyps.append(norm(hyp) or "_")
        refs_n.append(norm(s["audio_text"], True) or "_")
        hyps_n.append(norm(hyp, True) or "_")
        for e in entity_spans(s):
            ent_tot += 1
            ent_ok += fold(e) in fold(hyp)  # đúng khi bỏ dấu: đủ cho tool tìm kiếm
            ent_exact += norm(e) in norm(hyp)  # đúng cả dấu: cần cho tham số khớp tuyệt đối
    return {
        "wer": round(100 * jiwer.wer(refs, hyps), 2),
        "wer_num": round(100 * jiwer.wer(refs_n, hyps_n), 2),
        "cer": round(100 * jiwer.cer(refs, hyps), 2),
        "entity_recall_folded": round(100 * ent_ok / max(1, ent_tot), 2),
        "entity_recall_exact": round(100 * ent_exact / max(1, ent_tot), 2),
        "entities": ent_tot,
    }


def per_sample(sample, tr) -> dict:
    if "audio_text" not in sample:
        return {}
    r, h = norm(sample["audio_text"], True) or "_", norm(tr["final"], True) or "_"
    return {"wer": round(100 * jiwer.wer(r, h), 1)}
