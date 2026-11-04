"""Bộ chấm cục bộ theo PROPOSAL.md mục 1, và phân loại lỗi tự động.

Điểm mẫu = 0 nếu state_hash lệch, ngược lại = F1 trên danh sách call:
cùng tên tool, tham số không-fuzzy bằng nhau tuyệt đối, tham số fuzzy có token-F1 > 0,6, mỗi call gold khớp một lần.
Mẫu gold rỗng chỉ được 1 khi nộp rỗng. Kiểm: chấm chính gold của train phải ra 100.
"""
from __future__ import annotations

import re
import unicodedata

from . import data

FUZZY_THRESHOLD = 0.6


def fold(s) -> str:
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def tokens(s) -> list[str]:
    return re.findall(r"[^\W_]+", fold(s))


def token_f1(pred, gold) -> float:
    tp, tg = tokens(pred), tokens(gold)
    if not tp or not tg:
        return float(tp == tg)
    rest, common = list(tg), 0
    for t in tp:
        if t in rest:
            rest.remove(t)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(tp), common / len(tg)
    return 2 * p * r / (p + r)


def call_match(pred: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k, gv in ga.items():
        if k in fz:
            if token_f1(pa[k], gv) <= FUZZY_THRESHOLD:
                return False
        elif pa[k] != gv:
            return False
    return True


def match(pred_calls, gold_calls, fuzzy):
    """Ghép tham lam theo thứ tự nộp; trả về (danh sách cặp pred->gold, tập gold đã khớp)."""
    used, pairs = set(), []
    for i, p in enumerate(pred_calls):
        for j, g in enumerate(gold_calls):
            if j not in used and call_match(p, g, fuzzy):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs, used


def response_type(calls) -> str:
    if not calls:
        return "đứng yên"
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return "hỏi lại"
    return "làm"


def score_sample(split, pred_calls, pred_hash, gold_calls, gold_hash) -> dict:
    fuzzy = data.fuzzy_params(split)
    hash_ok = pred_hash == gold_hash
    pairs, used = match(pred_calls, gold_calls, fuzzy)
    m = len(pairs)
    u = len(pred_calls) - m
    if not gold_calls:
        f1 = 1.0 if not pred_calls else 0.0
    elif m == 0:
        f1 = 0.0
    else:
        p, r = m / (m + u), m / len(gold_calls)
        f1 = 2 * p * r / (p + r)
    out = {
        "score": f1 if hash_ok else 0.0,
        "hash_ok": hash_ok,
        "f1": f1,
        "M": m,
        "U": u,
        "G": len(gold_calls),
        "gold_type": response_type(gold_calls),
        "pred_type": response_type(pred_calls),
    }
    out["error"] = classify(split, out, pred_calls, gold_calls, pairs, used)
    return out


def _arg_kind(key: str, val) -> str:
    if key.endswith("_id") or key in ("id", "program"):
        return "sai ID"
    if isinstance(val, str) and re.match(r"^\d{4}-\d{2}-\d{2}", val):
        return "sai ngày giờ"
    if key in ("amount", "price", "quantity") or (isinstance(val, (int, float)) and not isinstance(val, bool)):
        return "sai số"
    return "sai tham số ghi"


def classify(split, s, pred_calls, gold_calls, pairs, used) -> str:
    """Một nhãn lỗi chính cho mỗi mẫu chưa đạt 1 điểm (mục 7 của proposal)."""
    if s["score"] == 1.0:
        return "đúng"
    if not gold_calls:
        return "gọi khi phải đứng yên"
    if s["gold_type"] != s["pred_type"]:
        return f"sai kiểu phản ứng ({s['gold_type']} → {s['pred_type']})"
    kind = lambda c: data.tool_kind(split, c["name"])
    pm = {i for i, _ in pairs}
    p_bad = [c for i, c in enumerate(pred_calls) if i not in pm]
    g_miss = [c for j, c in enumerate(gold_calls) if j not in used]
    if not s["hash_ok"]:
        gw = [c for c in g_miss if kind(c) == "write"]
        pw = [c for c in p_bad if kind(c) == "write"]
        if gw and pw:
            same = [(p, g) for g in gw for p in pw if p["name"] == g["name"]]
            if same:
                p, g = same[0]
                for k, gv in g["arguments"].items():
                    if p["arguments"].get(k) != gv and k not in data.fuzzy_params(split).get(g["name"], set()):
                        return _arg_kind(k, gv)
                return "sai tham số ghi"
            return "sai tool ghi"
        if gw:
            return "thiếu lệnh ghi"
        if pw:
            return "thừa lệnh ghi"
        return "lệch state (thứ tự / khác)"
    # hash đúng, F1 < 1 (call bị từ chối được runner gắn nhãn riêng)
    g_read = [c for c in g_miss if kind(c) == "read"]
    p_read = [c for c in p_bad if kind(c) == "read"]
    if g_read and p_read and {c["name"] for c in g_read} & {c["name"] for c in p_read}:
        return "sai tham số đọc"
    if p_read and not g_read:
        return "thừa call đọc"
    if g_read and not p_read:
        return "thiếu call đọc"
    if any(c["name"] == "sys.ask_missing_info" for c in g_miss + p_bad):
        return "sai ask_missing_info"
    fz = data.fuzzy_params(split)
    if any(fz.get(c["name"]) for c in g_miss):
        return "fuzzy dưới ngưỡng"
    return "khác"
