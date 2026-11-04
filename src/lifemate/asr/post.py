"""Hậu xử lý transcript, chạy sau ASR và không cần chạy lại ASR. Chọn bằng danh sách `asr_post` trong config."""
from __future__ import annotations

import json
import re
import unicodedata

from ..registry import register


class PostStep:
    def apply(self, sample: dict, tr: dict) -> dict:
        raise NotImplementedError


@register("asr_post", "strip_fillers")
class StripFillers(PostStep):
    """Bỏ từ đệm đầu câu ("ờ", "à", "ơi") nhưng giữ cụm đổi ý ("à thôi", "à mà")."""

    def __init__(self, **_):
        self.rx = re.compile(r"^(?:(?:ờ|ừ|à|ơ|ơi|ừm|ờm)(?!\s*(?:thôi|mà))[\s,.…]+)+", re.I)

    def apply(self, sample, tr):
        return {**tr, "final": self.rx.sub("", tr["final"]).strip()}


# -- khoá âm vị theo phương ngữ ---------------------------------------------------------------
# Gộp các cặp âm mà giọng Nam / Trung hay nhập làm một (ý tưởng của arXiv 2605.24451, bản luật tay):
# phụ âm đầu v/d/gi/r → z, tr/ch → c, s/x → s, l/n → l (một số vùng Bắc); vần cuối n/ng → n, t/c → t;
# bỏ thanh điệu (hỏi/ngã nhập ở giọng Trung và Nam).
_ONSETS = [("ngh", "ng"), ("gi", "z"), ("tr", "c"), ("ch", "c"), ("kh", "k"), ("ph", "f"), ("th", "t"),
           ("qu", "kw"), ("gh", "g"), ("v", "z"), ("d", "z"), ("r", "z"), ("x", "s"), ("c", "k"), ("q", "k")]


def _strip_tone(s: str) -> str:
    t = unicodedata.normalize("NFD", s.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn" or c in "̛̆̂")  # giữ ă â ơ ư ê ô
    return unicodedata.normalize("NFC", t).replace("đ", "d")


def phonetic_key(syllable: str, merge_ln: bool = False) -> str:
    s = _strip_tone(re.sub(r"[^\wÀ-ỹđĐ]", "", syllable))
    for a, b in _ONSETS:
        if s.startswith(a):
            s = b + s[len(a):]
            break
    if merge_ln and s.startswith("n") and not s.startswith("ng"):
        s = "l" + s[1:]
    s = re.sub(r"ng$|nh$", "n", s)
    s = re.sub(r"ch$|c$", "t", s)
    return s


@register("asr_post", "phonetic_entity_fix")
class PhoneticEntityFix(PostStep):
    """Sửa tên riêng nghe sai bằng khoá âm vị phương ngữ, đối chiếu với tên trong hồ sơ và lịch sử."""

    def __init__(self, max_ngram=3, merge_ln=False, **_):
        self.n = max_ngram
        self.merge_ln = merge_ln

    def entities(self, sample) -> list[str]:
        p = sample["context"].get("user_profile", {})
        names = [p.get("name", ""), p.get("city", "")]
        blob = " ".join(t.get("text", "") + " " + json.dumps(t.get("tool_results") or [], ensure_ascii=False)
                        for t in sample.get("history", []) if t.get("role") == "assistant")
        names += re.findall(r"(?:[A-ZĐÀ-Ỹ][\wÀ-ỹ]+)(?: [A-ZĐÀ-Ỹ][\wÀ-ỹ]+)*", blob)
        return [n for n in dict.fromkeys(names) if n and len(n) > 2]

    def key(self, words):
        return " ".join(phonetic_key(w, self.merge_ln) for w in words)

    def apply(self, sample, tr):
        ents = {}
        for e in self.entities(sample):
            ws = e.split()
            if 1 <= len(ws) <= self.n:
                ents.setdefault(self.key(ws), e)
        words = tr["final"].split()
        out, i, fixes = [], 0, []
        while i < len(words):
            for n in range(min(self.n, len(words) - i), 0, -1):
                span = words[i:i + n]
                core = [re.sub(r"[^\wÀ-ỹđĐ]", "", w) for w in span]
                ent = ents.get(self.key(core))
                if ent and " ".join(core) != ent and " ".join(core).lower() != ent.lower():
                    tail = re.sub(r"^[\wÀ-ỹđĐ]+", "", span[-1])
                    out.append(ent + tail)
                    fixes.append((" ".join(span), ent))
                    i += n
                    break
            else:
                out.append(words[i])
                i += 1
        return {**tr, "final": " ".join(out), "fixes": tr.get("fixes", []) + fixes}
