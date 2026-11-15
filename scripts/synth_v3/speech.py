"""Lời nói tiếng Việt kiểu bản chép ASR (số đọc thành chữ ~87%) và nhiễu ASR lấy từ nhầm lẫn thật của fold ASR (config 181).

Bảng nhầm lẫn chỉ dựng từ train_rest: gold `audio_text` so với bản chép theo fold (ERR: gold → primary) và primary so với
các phương án N-best (NB: primary → alt). Không đọc public_test.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
WEEKDAYS = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ Nhật"]
P_WORDS = 0.87


def w99(n: int) -> str:
    if n < 10:
        return DIGITS[n]
    t, u = divmod(n, 10)
    if n < 20:
        return "mười" + ("" if u == 0 else " " + ("lăm" if u == 5 else DIGITS[u]))
    return f"{DIGITS[t]} mươi" + ("" if u == 0 else " " + {1: "mốt", 4: "tư", 5: "lăm"}.get(u, DIGITS[u]))


def w999(n: int) -> str:
    if n < 100:
        return w99(n)
    h, r = divmod(n, 100)
    if r == 0:
        return f"{DIGITS[h]} trăm"
    return f"{DIGITS[h]} trăm " + (f"linh {DIGITS[r]}" if r < 10 else w99(r))


def num(n: int, words: bool) -> str:
    return w999(n) if words and n < 1000 else str(n)


class Say:
    """Một giá trị nói ra: chữ ở bản chép chính và (nếu là số) dạng chữ số mà một phương án N-best có thể ra."""

    def __init__(self, text: str, digits: str | None = None):
        self.text, self.digits = text, digits


def money(v: int, rng: random.Random) -> Say:
    m, r = divmod(v, 1_000_000)
    words = rng.random() < P_WORDS
    k = rng.choice(["nghìn", "nghìn", "ngàn"])

    def fmt(wd: bool) -> str:
        if m and not r:
            return f"{num(m, wd)} triệu"
        if m and r == 500_000:
            return f"{num(m, wd)} triệu rưỡi"
        if m and r % 100_000 == 0:
            return (
                f"{num(m, wd)} triệu {num(r // 100_000, wd)}"
                if r // 100_000 < 10
                else f"{num(m, wd)} triệu {num(r // 1000, wd)} {k}"
            )
        if m:
            return f"{num(m, wd)} triệu {num(r // 1000, wd)} {k}"
        return f"{num(v // 1000, wd)} {k}"

    return Say(fmt(words), fmt(not words))


def integer(n: int, unit: str, rng: random.Random) -> Say:
    words = rng.random() < P_WORDS
    u = f" {unit}" if unit else ""
    return Say(num(n, words) + u, num(n, not words) + u)


def part(h: int) -> str:
    return "sáng" if h < 11 else "trưa" if h < 13 else "chiều" if h < 18 else "tối"


def clock(h: int, m: int, rng: random.Random, words: bool) -> str:
    h12 = h if h <= 12 else h - 12
    if m == 0:
        core = f"{num(h12, words)} giờ"
    elif m == 30:
        core = rng.choice([f"{num(h12, words)} rưỡi", f"{num(h12, words)} giờ rưỡi"])
    else:
        core = f"{num(h12, words)} giờ {num(m, words)}"
    return f"{core} {part(h)}"


def day(now: datetime, d, rng: random.Random, words: bool) -> str:
    k = (d - now.date()).days
    monday = now.date() - timedelta(days=now.date().weekday())
    opts = {0: ["hôm nay"], 1: ["mai", "ngày mai"], 2: ["ngày mốt", "ngày kia"], -1: ["hôm qua"]}.get(k, [])
    if k >= 2:
        week = "tuần này" if d < monday + timedelta(days=7) else "tuần sau" if d < monday + timedelta(days=14) else None
        if week:
            opts.append(f"{WEEKDAYS[d.weekday()]} {week}")
    if k >= 1 and not opts[1:]:
        opts.append(f"mùng {num(d.day, words)}" if d.day <= 10 else f"ngày {num(d.day, words)}")
    return rng.choice(opts)


def when(now: datetime, dt: datetime, rng: random.Random) -> Say:
    words = rng.random() < P_WORDS
    seed = rng.random()

    def fmt(wd: bool) -> str:
        r = random.Random(seed)
        c = clock(dt.hour, dt.minute, r, wd)
        k = (dt.date() - now.date()).days
        if k == 0 and r.random() < 0.7:
            return f"{c} nay"
        if k == 1 and r.random() < 0.6:
            return f"{c} mai"
        d = day(now, dt.date(), r, wd)
        return r.choice([f"{c} {d}", f"{d} lúc {c}"])

    return Say(fmt(words), fmt(not words))


def date_say(now: datetime, d, rng: random.Random) -> Say:
    words = rng.random() < P_WORDS
    seed = rng.random()
    return Say(day(now, d, random.Random(seed), words), day(now, d, random.Random(seed), not words))


def duration(minutes: int, rng: random.Random) -> Say:
    words = rng.random() < P_WORDS
    if minutes % 60 == 0:
        h = minutes // 60
        return Say(f"{num(h, words)} tiếng", f"{num(h, not words)} tiếng")
    if minutes == 30:
        return Say("nửa tiếng", "30 phút" if words else "ba mươi phút")
    return Say(f"{num(minutes, words)} phút", f"{num(minutes, not words)} phút")


# ---------------------------------------------------------------------------------------------------------------------
TOK = re.compile(r"\S+")
PUNCT = ".,!?…:;\"'"


def fold(s: str) -> str:
    t = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(c for c in t if not unicodedata.combining(c))


def _toks(s: str) -> list[str]:
    return [t.strip(PUNCT).lower() for t in s.split()]


def build_confusions(gold_path: str, asr_path: str, ids: set[str]) -> tuple[dict, dict]:
    """ERR: gold → primary, NB: primary → phương án khác; khoá là cụm 1-2 từ (thường, bỏ dấu câu)."""
    G = {}
    for line in open(gold_path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in ids:
            G[r["id"]] = r["final"]
    err, nb = defaultdict(Counter), defaultdict(Counter)

    def pairs(a: str, b: str, into):
        ta, tb = _toks(a), _toks(b)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
            if op == "replace" and 1 <= i2 - i1 <= 2 and 1 <= j2 - j1 <= 2:
                x, y = " ".join(ta[i1:i2]), " ".join(tb[j1:j2])
                if x and y and x != y and not re.search(r"\d", x + y):
                    into[x][y] += 1

    for line in open(asr_path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in G:
            continue
        pairs(G[r["id"]], r["final"], err)
        for alt in r.get("alts", {}).get("final", []):
            pairs(r["final"], alt, nb)

    def keep(tab):  # bỏ cặp lệch hàng của difflib: giữ cặp gần âm (sau bỏ dấu) và gặp ít nhất 2 lần
        out = {}
        for x, ys in tab.items():
            ys = {
                y: c for y, c in ys.items() if c >= 2 and difflib.SequenceMatcher(a=fold(x), b=fold(y)).ratio() >= 0.6
            }
            if ys:
                out[x] = ys
        return out

    return keep(err), keep(nb)


def _sub_once(text: str, table: dict, rng: random.Random, protected: list[tuple[int, int]], min_count=1) -> str | None:
    """Thay một cụm 1-2 từ (ngoài vùng bảo vệ) bằng một biến thể trong bảng, giữ dấu câu và chữ hoa đầu câu."""
    spans = [(m.start(), m.end()) for m in TOK.finditer(text)]
    cands = []
    for n in (1, 2):
        for i in range(len(spans) - n + 1):
            a, b = spans[i][0], spans[i + n - 1][1]
            if any(not (b <= pa or a >= pb) for pa, pb in protected):
                continue
            raw = text[a:b]
            core = raw.strip(PUNCT)
            key = core.lower()
            opts = {y: c for y, c in table.get(key, {}).items() if c >= min_count}
            if opts:
                cands.append((a, b, raw, core, opts))
    if not cands:
        return None
    a, b, raw, core, opts = rng.choice(cands)
    ys, ws = zip(*opts.items())
    y = rng.choices(ys, ws)[0]
    if core[:1].isupper():
        y = y[:1].upper() + y[1:]
    lead = raw[: len(raw) - len(raw.lstrip(PUNCT))]
    trail = raw[len(raw.rstrip(PUNCT)) :]
    return text[:a] + lead + y + trail + text[b:]


# luật nhầm âm tiếng Việt cho tên riêng khi bảng không có (phụ âm đầu, vần cuối, dấu thanh)
INITIAL = [
    ("tr", "ch"),
    ("ch", "tr"),
    ("gi", "d"),
    ("d", "gi"),
    ("r", "d"),
    ("s", "x"),
    ("x", "s"),
    ("l", "n"),
    ("n", "l"),
    ("v", "d"),
]
FINAL = [("ng", "n"), ("n", "ng"), ("c", "t"), ("t", "c"), ("nh", "n")]
TONES = {
    "ã": "á",
    "á": "ã",
    "ả": "ã",
    "ạ": "ả",
    "ấ": "ẩ",
    "ẩ": "ấ",
    "ỳ": "ý",
    "ũ": "ú",
    "ủ": "ũ",
    "ỏ": "õ",
    "õ": "ỏ",
    "ẽ": "é",
    "ể": "ễ",
    "ễ": "ể",
    "ỡ": "ở",
    "ở": "ỡ",
    "ử": "ữ",
    "ữ": "ử",
    "ị": "ỉ",
    "ỉ": "ĩ",
    "ĩ": "ỉ",
    "ỗ": "ổ",
    "ổ": "ỗ",
    "ằ": "ầ",
    "ầ": "ằ",
    "ắ": "ấ",
    "ờ": "ơ",
    "ư": "ừ",
    "ơ": "ờ",
    "í": "ì",
    "ì": "í",
    "é": "è",
    "è": "é",
    "ó": "ò",
    "ò": "ó",
    "ú": "ù",
    "ù": "ú",
    "à": "á",
    "ồ": "ố",
    "ố": "ồ",
    "ệ": "ễ",
    "ặ": "ắ",
}


def corrupt_word(w: str, table: dict, rng: random.Random) -> str | None:
    key = w.lower()
    if key in table:
        ys, ws = zip(*table[key].items())
        y = rng.choices(ys, ws)[0]
        if " " not in y and fold(y)[:1] == fold(key)[:1] or rng.random() < 0.3:
            return y[:1].upper() + y[1:] if w[:1].isupper() else y
    opts = []
    for a, b in INITIAL:
        if key.startswith(a) and not (a == "d" and key.startswith("đ")):
            rest = key[len(a) :]
            if b == "gi" and rest.startswith(("i", "ì", "í", "ỉ", "ĩ", "ị")):
                b = "g"
            opts.append(b + rest)
    for a, b in FINAL:
        if key.endswith(a) and len(key) > len(a) + 1:
            opts.append(key[: -len(a)] + b)
    for i, ch in enumerate(key):
        if ch in TONES and len(TONES[ch]) == 1:
            opts.append(key[:i] + TONES[ch] + key[i + 1 :])
    opts = [o for o in opts if o != key]
    if not opts:
        return None
    y = rng.choice(opts)
    return y[:1].upper() + y[1:] if w[:1].isupper() else y
