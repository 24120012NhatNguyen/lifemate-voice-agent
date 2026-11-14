"""Bảng lỗi của một lượt chạy trên tập val, theo các nhóm của brief LM-47 (bước 0): model sai gì trên app chưa thấy.

Mỗi mẫu chưa đạt 1 điểm nhận đúng một nhóm, xét theo thứ tự:

    làm → đứng yên | làm → hỏi lại | hỏi lại → làm | kiểu phản ứng khác      gold và dự đoán khác kiểu phản ứng
    thiếu lệnh ghi (không ghi gì | ghi thiếu)      một lệnh ghi của gold không có lệnh ghi nào của dự đoán thế chỗ
    sai tool ghi                                   lệnh ghi của gold bị thay bằng lệnh ghi của tool khác
    sai tham số lấy từ kết quả đọc                 cùng tool ghi, lệch ở tham số mà giá trị gold nằm trong kết quả tool đọc
                                                   của lượt này hoặc trong lịch sử (ID, hoặc giá trị chép từ bản ghi)
    sai định dạng giá trị                          cùng giá trị nhưng khác dạng: ngày ↔ ngày giờ, lệch múi giờ / giây,
                                                   số lệch bậc nghìn, chuỗi ↔ số, khác hoa thường / dấu
    sai giá trị nói ra                             lệch ở tham số lấy từ lời (giờ, số, enum, tên) mà không phải hai nhóm trên
    thừa / thiếu tham số                           cùng tool ghi, khác tập tham số (điền tham số tuỳ chọn gold bỏ trống…)
    văn bản tự do dưới ngưỡng                      chỉ lệch tham số fuzzy (token-F1 ≤ 0,6)
    thừa lệnh ghi
    sai tool đọc | sai tham số đọc | thiếu call đọc | thừa call đọc           lệnh ghi khớp hết, lệch ở call đọc
    sai câu hỏi lại | khác

Với mẫu có nhiều lỗi, nhóm là lỗi của lệnh ghi gold đầu tiên không khớp. Cột "ASR" đếm mẫu mà một giá trị chuỗi của
gold có trong lời gold nhưng không còn trong bản chép (kể cả các bản nghe khác): lỗi có thể do nghe sai.
Cột "luật" đếm mẫu mà câu hỏi lại do luật ask_missing / pick_one của GuardedEnv thêm vào, không phải do LLM.

    python scripts/unseen_errors.py RUN_DIR [RUN_DIR…] [--ids 5] [--by-app] [--detail] [--json FILE]

Chỉ in số tổng hợp và ID mẫu (không in câu nói hay tham số), nên bảng dán được vào PR.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from lifemate import data, splits
from lifemate.scoring import FUZZY_THRESHOLD, fold, match, response_type, token_f1

SPLIT = "train"
ASK = "sys.ask_missing_info"
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")
ORDER = ["làm → đứng yên", "làm → hỏi lại", "hỏi lại → làm", "kiểu phản ứng khác",
         "thiếu lệnh ghi: không ghi gì", "thiếu lệnh ghi: ghi thiếu", "sai tool ghi",
         "sai tham số lấy từ kết quả đọc", "sai định dạng giá trị", "sai giá trị nói ra", "thừa / thiếu tham số",
         "văn bản tự do dưới ngưỡng", "thừa lệnh ghi", "sai tool đọc", "sai tham số đọc", "thiếu call đọc",
         "thừa call đọc", "sai câu hỏi lại", "khác"]
BRIEF = {  # gộp về 7 nhóm brief nêu
    "làm → đứng yên": "làm → đứng yên", "làm → hỏi lại": "làm → hỏi lại", "hỏi lại → làm": "hỏi lại → làm",
    "thiếu lệnh ghi: không ghi gì": "thiếu lệnh ghi", "thiếu lệnh ghi: ghi thiếu": "thiếu lệnh ghi",
    "sai tool ghi": "sai tool", "sai tool đọc": "sai tool",
    "sai tham số lấy từ kết quả đọc": "sai tham số lấy từ kết quả đọc", "sai định dạng giá trị": "sai định dạng giá trị",
}


def leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)
    else:
        yield obj


def kind(name: str) -> str:
    return data.tool_kind(SPLIT, name)


def _when(v: str):
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def _digits(v) -> str:
    d = re.sub(r"\D", "", str(v))
    return "0" + d[2:] if d.startswith("84") and len(d) == 11 else d


def same_value_other_format(p, g) -> bool:
    """Cùng một giá trị viết khác dạng."""
    if isinstance(p, bool) or isinstance(g, bool):
        return str(p).lower() == str(g).lower() and type(p) is not type(g)
    if isinstance(p, (int, float)) and isinstance(g, (int, float)):
        if p == 0 or g == 0:
            return False
        r = max(abs(p), abs(g)) / min(abs(p), abs(g))
        return r in (10, 100, 1000, 10_000, 100_000, 1_000_000)
    if isinstance(p, (int, float)) != isinstance(g, (int, float)):  # chuỗi ↔ số
        return _digits(p) == _digits(g) and _digits(g) != ""
    if isinstance(p, list) != isinstance(g, list):
        one, many = (g, p) if isinstance(p, list) else (p, g)
        return len(many) == 1 and many[0] == one
    if isinstance(p, str) and isinstance(g, str):
        if ISO.match(p) and ISO.match(g):
            if p[:10] != g[:10]:
                return False
            if ("T" in p) != ("T" in g):
                return True  # ngày ↔ ngày giờ của cùng ngày
            a, b = _when(p), _when(g)
            if a is None or b is None:
                return False
            if (a.tzinfo is None) != (b.tzinfo is None):
                return a.replace(tzinfo=None, second=0) == b.replace(tzinfo=None, second=0)
            return a == b or a.replace(second=0) == b.replace(second=0)
        if ISO.match(g) and g[:10] in p or ISO.match(p) and p[:10] in g:
            return True
        if fold(p) == fold(g) or re.sub(r"[\W_]+", "", fold(p)) == re.sub(r"[\W_]+", "", fold(g)):
            return True
        if len(_digits(g)) >= 8 and _digits(p) == _digits(g):
            return True
    return False


def gold_sources(sample: dict, upto: int) -> set:
    """Mọi giá trị lá trong kết quả các call gold trước call thứ `upto` và trong lịch sử."""
    out = set()
    for r in (sample["label"].get("tool_results") or [])[:upto]:
        out |= {v for v in leaves(r) if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    for t in sample.get("history", []):
        for r in t.get("tool_results") or []:
            out |= {v for v in leaves(r) if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return out


def arg_diff(p: dict, g: dict, sample: dict, g_index: int) -> str:
    """Nhóm lỗi của một cặp lệnh ghi cùng tool mà không khớp."""
    name = g["name"]
    pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
    props = (data.tool_specs(SPLIT).get(name, {}).get("parameters") or {}).get("properties") or {}
    fz = data.fuzzy_params(SPLIT).get(name, set())
    src = gold_sources(sample, g_index)
    found = set()
    for k in sorted(set(pa) | set(ga)):
        if k not in pa or k not in ga:
            found.add("thừa / thiếu tham số")
        elif k in fz:
            if token_f1(pa[k], ga[k]) <= FUZZY_THRESHOLD:
                found.add("văn bản tự do dưới ngưỡng")
        elif pa[k] != ga[k]:
            gv = ga[k]
            vals = gv if isinstance(gv, list) else [gv]
            flat = [x for v in vals for x in (leaves(v) if isinstance(v, (dict, list)) else [v])]
            from_read = bool(props.get(k, {}).get("x-ref")) or k.endswith("_id") or any(
                isinstance(x, str) and not ISO.match(x) and x in src for x in flat)
            if same_value_other_format(pa[k], gv):
                found.add("sai định dạng giá trị")
            elif from_read:
                found.add("sai tham số lấy từ kết quả đọc")
            else:
                found.add("sai giá trị nói ra")
    for c in ("sai tham số lấy từ kết quả đọc", "sai định dạng giá trị", "sai giá trị nói ra", "thừa / thiếu tham số",
              "văn bản tự do dưới ngưỡng"):
        if c in found:
            return c
    return "khác"


def classify(row: dict, sample: dict) -> str:
    pred, gold = row["tool_calls"], row["gold_calls"]
    gt, pt = response_type(gold), response_type(pred)
    if gt != pt:
        name = f"{gt} → {pt}"
        return name if name in ORDER else "kiểu phản ứng khác"
    fuzzy = data.fuzzy_params(SPLIT)
    pairs, used = match(pred, gold, fuzzy)
    pm = {i for i, _ in pairs}
    p_un = [(i, c) for i, c in enumerate(pred) if i not in pm]
    g_un = [(j, c) for j, c in enumerate(gold) if j not in used]
    pw = [(i, c) for i, c in p_un if kind(c["name"]) == "write"]
    gw = [(j, c) for j, c in g_un if kind(c["name"]) == "write"]
    if gw:
        j, g = gw[0]
        same = next((c for _, c in pw if c["name"] == g["name"]), None)
        if same is not None:
            return arg_diff(same, g, sample, j)
        names_gold = Counter(c["name"] for _, c in gw)
        if any(c["name"] not in names_gold for _, c in pw):
            return "sai tool ghi"
        wrote = any(kind(c["name"]) == "write" for c in pred)
        return "thiếu lệnh ghi: ghi thiếu" if wrote else "thiếu lệnh ghi: không ghi gì"
    if pw:
        return "thừa lệnh ghi"
    pr = [c for _, c in p_un if kind(c["name"]) == "read"]
    gr = [c for _, c in g_un if kind(c["name"]) == "read"]
    if gr and pr:
        return "sai tham số đọc" if {c["name"] for c in gr} & {c["name"] for c in pr} else "sai tool đọc"
    if gr:
        return "thiếu call đọc"
    if pr:
        return "thừa call đọc"
    if any(c["name"] == ASK for _, c in p_un + g_un):
        return "sai câu hỏi lại"
    return "khác"


def asr_lost(row: dict, alts: list[str]) -> bool:
    """Một giá trị chuỗi của gold có trong lời gold nhưng không còn trong bản chép nào."""
    ref = fold(row.get("reference") or "")
    heard = " | ".join(fold(x) for x in [row.get("transcript") or "", *alts])
    for c in row["gold_calls"]:
        for v in leaves(c.get("arguments") or {}):
            if isinstance(v, str) and not ISO.match(v) and len(fold(v)) >= 3 and fold(v) in ref and fold(v) not in heard:
                return True
    return False


def rule_asked(row: dict) -> bool:
    return any(str(g.get("decision", "")).startswith(("hoãn", "chạy (hỏi lại đã hoãn)")) for g in row.get("guard_log") or [])


def _app(name: str) -> str:
    return name.split(".", 1)[0]


def _is_id(tool: str, key: str) -> bool:
    p = ((data.tool_specs(SPLIT).get(tool, {}).get("parameters") or {}).get("properties") or {}).get(key, {})
    return key.endswith("_id") or bool(p.get("x-ref"))


def _source_tool(tool: str, key: str) -> str | None:
    """Tool đọc mà mô tả tham số chỉ ra ("ID …, lấy từ app.tool")."""
    p = ((data.tool_specs(SPLIT).get(tool, {}).get("parameters") or {}).get("properties") or {}).get(key, {})
    m = re.search(r"lấy từ ([a-z_0-9]+\.[a-z_0-9]+)", p.get("description", ""))
    return m.group(1) if m else None


def _value_type(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "số"
    if isinstance(v, str) and ISO.match(v):
        return "ngày giờ" if "T" in v else "ngày"
    if isinstance(v, (list, dict)):
        return "danh sách / object"
    return "chuỗi"


def details(row: dict, sample: dict, cat: str) -> list[str]:
    """Các nhãn cấu trúc của một mẫu sai (không lộ nội dung): dùng để hiểu nguyên nhân trong từng nhóm."""
    pred, gold = row["tool_calls"], row["gold_calls"]
    held = set(splits.HELDOUT_APPS)
    specs = data.tool_specs(SPLIT)
    out = []
    pred_reads = [c["name"] for c in pred if kind(c["name"]) == "read"]
    gold_reads = [c["name"] for c in gold if kind(c["name"]) == "read"]
    gold_writes = [c for c in gold if kind(c["name"]) == "write"]
    seen_obs = set()
    for s in row.get("steps") or []:
        seen_obs |= {v for v in leaves(s.get("observation")) if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    if cat == "làm → hỏi lại":
        asks = [c for c in pred if c["name"] == ASK]
        a = (asks[-1].get("arguments") or {}) if asks else {}
        tool, params = str(a.get("tool_name", "")), a.get("params") if isinstance(a.get("params"), list) else []
        out.append("hỏi do luật của GuardedEnv" if rule_asked(row) else "hỏi do LLM")
        out.append("tool hỏi là tool ghi của gold" if tool in {c["name"] for c in gold_writes} else
                   "tool hỏi thuộc app giữ lại nhưng không phải tool gold" if _app(tool) in held else "tool hỏi thuộc app đã thấy / lạ")
        ids = [p for p in params if isinstance(p, str) and _is_id(tool, p)]
        if ids:
            src = {_source_tool(tool, p) for p in ids} - {None}
            out.append("hỏi tham số ID: " + ("đã đọc tool nguồn" if src & set(pred_reads) else
                                             "chưa đọc tool nguồn" if src else "không rõ tool nguồn"))
        else:
            req = set((specs.get(tool, {}).get("parameters") or {}).get("required") or [])
            out.append("hỏi tham số không phải ID: " + ("bắt buộc" if set(map(str, params)) & req else "tuỳ chọn / lạ"))
        out.append("trước khi hỏi: " + ("có đọc" if pred_reads else "không đọc gì"))
        out.append("gold: " + ("đọc rồi ghi" if gold_reads else "ghi thẳng"))
    elif cat == "làm → đứng yên":
        out.append("call đầu của gold là " + kind(gold[0]["name"]))
        out.append(f"gold có {min(len(gold), 4)}{'+' if len(gold) >= 4 else ''} call")
        out.append("app của gold: " + ("chỉ app giữ lại" if {_app(c['name']) for c in gold} <= held else "có cả app đã thấy"))
    elif cat.startswith("thiếu lệnh ghi"):
        missing = [c for c in gold_writes if c["name"] not in {p["name"] for p in pred}]
        out.append("lệnh ghi thiếu thuộc " + ("app giữ lại" if any(_app(c["name"]) in held for c in missing) else "app đã thấy"))
        if cat.endswith("không ghi gì"):
            out.append("đã đọc đủ các tool đọc của gold rồi dừng" if gold_reads and set(gold_reads) <= set(pred_reads) else
                       "gold không cần đọc, model đọc rồi dừng" if not gold_reads else "đọc thiếu / đọc tool khác rồi dừng")
            ids = [(c["name"], k) for c in missing for k in (c.get("arguments") or {}) if _is_id(c["name"], k)]
            if ids:
                got = [all(v in seen_obs for v in ([c["arguments"][k]] if not isinstance(c["arguments"][k], list) else c["arguments"][k])
                           if isinstance(v, str)) for c in missing for k in (c.get("arguments") or {}) if _is_id(c["name"], k)]
                out.append("ID cần cho lệnh ghi: " + ("đã có trong kết quả model đọc" if all(got) else "chưa có trong kết quả model đọc"))
        else:
            out.append(f"gold {min(len(gold_writes), 4)} lệnh ghi, model {sum(1 for c in pred if kind(c['name']) == 'write')}")
    elif cat == "sai tool ghi":
        g = next((c for c in gold_writes if c["name"] not in {p["name"] for p in pred}), None)
        extra = [c["name"] for c in pred if kind(c["name"]) == "write" and c["name"] not in {x["name"] for x in gold}]
        if g and extra:
            out.append("thay bằng tool khác của cùng app" if any(_app(x) == _app(g["name"]) for x in extra) else
                       "thay bằng tool của app đã thấy" if any(_app(x) not in held for x in extra) else "thay bằng tool app giữ lại khác")
    elif cat == "sai tham số lấy từ kết quả đọc":
        fuzzy = data.fuzzy_params(SPLIT)
        _, used = match(pred, gold, fuzzy)
        for j, g in enumerate(gold):
            if j in used or kind(g["name"]) != "write":
                continue
            p = next((c for c in pred if c["name"] == g["name"]), None)
            if p is None:
                continue
            for k, gv in (g.get("arguments") or {}).items():
                pv = (p.get("arguments") or {}).get(k)
                if pv is None or pv == gv or k in fuzzy.get(g["name"], set()):
                    continue
                vals = pv if isinstance(pv, list) else [pv]
                seen = all(v in seen_obs for v in vals if isinstance(v, (str, int, float)))
                said = all(fold(str(v)) in fold(row.get("transcript") or "") for v in vals if isinstance(v, (str, int, float)))
                out.append(("tham số ID" if _is_id(g["name"], k) else "tham số giá trị") + ": " +
                           ("model chọn mục khác trong kết quả đã đọc" if seen else "model lấy từ lời nói" if said else
                            "model tự đặt (không có trong kết quả đã đọc)"))
            break
    elif cat == "hỏi lại → làm":
        for c in gold:
            if c["name"] == ASK:
                a = c.get("arguments") or {}
                tool, params = str(a.get("tool_name", "")), a.get("params") if isinstance(a.get("params"), list) else []
                out.append("gold hỏi tham số ID (nhiều mục khớp / không tìm thấy)" if any(_is_id(tool, str(p)) for p in params)
                           else "gold hỏi tham số không phải ID (thiếu thông tin)")
                out.append("model có ghi bằng chính tool đó" if tool in {p["name"] for p in pred} else "model ghi tool khác / chỉ đọc")
                break
    elif cat == "sai giá trị nói ra":
        fuzzy = data.fuzzy_params(SPLIT)
        _, used = match(pred, gold, fuzzy)
        for j, g in enumerate(gold):
            if j in used or kind(g["name"]) != "write":
                continue
            p = next((c for c in pred if c["name"] == g["name"]), None)
            if p is None:
                continue
            for k, gv in (g.get("arguments") or {}).items():
                pv = (p.get("arguments") or {}).get(k)
                if pv is None or pv == gv or k in fuzzy.get(g["name"], set()):
                    continue
                enum = bool(((specs.get(g["name"], {}).get("parameters") or {}).get("properties") or {}).get(k, {}).get("enum"))
                out.append("kiểu giá trị lệch: " + ("enum" if enum else _value_type(gv)))
            break
    return out


def heldout_app(sample: dict) -> str:
    held = set(splits.HELDOUT_APPS)
    for c in sample["label"]["tool_calls"]:
        name = c["name"] if c["name"] != ASK else str((c.get("arguments") or {}).get("tool_name", ""))
        if name.split(".", 1)[0] in held:
            return name.split(".", 1)[0]
    return "-"


def analyse(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").open(encoding="utf-8") if line.strip()]
    alts = {}
    tp = run_dir / "transcripts.jsonl"
    if tp.exists():
        for line in tp.open(encoding="utf-8"):
            t = json.loads(line)
            alts[t["id"]] = list((t.get("alts") or {}).get("final") or [])
    by_id = data.by_id(SPLIT)
    out = {"name": run_dir.name, "n": len(rows), "score": round(100 * sum(r["score"] for r in rows) / len(rows), 2),
           "cats": defaultdict(lambda: {"n": 0, "lost": 0.0, "asr": 0, "rule": 0, "ids": [], "detail": Counter()}),
           "apps": defaultdict(lambda: {"n": 0, "score": 0.0, "cats": Counter()}), "rows": {}}
    for r in rows:
        s = by_id[r["id"]]
        app = heldout_app(s)
        a = out["apps"][app]
        a["n"] += 1
        a["score"] += r["score"]
        if r["score"] == 1.0:
            continue
        cat = classify(r, s)
        c = out["cats"][cat]
        c["n"] += 1
        c["lost"] += 1 - r["score"]
        c["asr"] += asr_lost(r, alts.get(r["id"], []))
        c["rule"] += rule_asked(r)
        c["ids"].append(r["id"])
        c["detail"].update(details(r, s, cat))
        a["cats"][BRIEF.get(cat, "khác")] += 1
        out["rows"][r["id"]] = cat
    return out


def table(runs: list[dict], n_ids: int) -> str:
    head = "| Nhóm lỗi | " + " | ".join(f"{r['name'].split('_', 1)[1]}: mẫu | điểm mất | ASR | luật" for r in runs) + " |"
    lines = [head, "|---|" + "---:|" * (4 * len(runs))]

    def cells(r, cats):
        n = sum(r["cats"][c]["n"] for c in cats if c in r["cats"])
        lost = sum(r["cats"][c]["lost"] for c in cats if c in r["cats"])
        asr = sum(r["cats"][c]["asr"] for c in cats if c in r["cats"])
        rule = sum(r["cats"][c]["rule"] for c in cats if c in r["cats"])
        return f"{n} | {100 * lost / r['n']:.2f} | {asr} | {rule}"

    for cat in ORDER:
        if any(cat in r["cats"] for r in runs):
            lines.append(f"| {cat} | " + " | ".join(cells(r, [cat]) for r in runs) + " |")
    lines.append("| **tổng mẫu sai** | " + " | ".join(cells(r, ORDER) for r in runs) + " |")
    text = "\n".join(lines)
    if n_ids:
        text += "\n\nID tiêu biểu (mỗi nhóm tối đa %d, theo lượt chạy):\n" % n_ids
        for r in runs:
            text += f"\n- {r['name']}:\n"
            for cat in ORDER:
                if cat in r["cats"]:
                    text += f"  - {cat}: {', '.join(r['cats'][cat]['ids'][:n_ids])}\n"
    return text


def brief_table(runs: list[dict]) -> str:
    groups = ["làm → đứng yên", "làm → hỏi lại", "hỏi lại → làm", "thiếu lệnh ghi", "sai tool",
              "sai tham số lấy từ kết quả đọc", "sai định dạng giá trị", "khác"]
    lines = ["| Nhóm (theo brief) | " + " | ".join(f"{r['name'].split('_', 1)[1]}: mẫu | điểm mất" for r in runs) + " |",
             "|---|" + "---:|" * (2 * len(runs))]
    for g in groups:
        cells = []
        for r in runs:
            cats = [c for c in r["cats"] if BRIEF.get(c, "khác") == g]
            n = sum(r["cats"][c]["n"] for c in cats)
            lost = sum(r["cats"][c]["lost"] for c in cats)
            cells.append(f"{n} | {100 * lost / r['n']:.2f}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def detail_text(runs: list[dict]) -> str:
    """Nhãn cấu trúc trong từng nhóm (số mẫu có nhãn đó), để hiểu nguyên nhân mà không cần đọc nội dung mẫu."""
    lines = []
    for r in runs:
        lines.append(f"- {r['name']}:")
        for cat in ORDER:
            d = r["cats"].get(cat, {}).get("detail")
            if d:
                lines.append(f"  - {cat} ({r['cats'][cat]['n']}): " + "; ".join(f"{k} {v}" for k, v in d.most_common()))
    return "\n".join(lines)


def app_table(runs: list[dict]) -> str:
    apps = sorted({a for r in runs for a in r["apps"]})
    lines = ["| App giữ lại của gold | " + " | ".join(f"{r['name'].split('_', 1)[1]}: mẫu | điểm | nhóm lỗi nhiều nhất" for r in runs) + " |",
             "|---|" + "---:|---:|---|" * len(runs)]
    for a in apps:
        cells = []
        for r in runs:
            x = r["apps"].get(a)
            if not x:
                cells.append("0 | - | -")
                continue
            top = ", ".join(f"{k} {v}" for k, v in x["cats"].most_common(3))
            cells.append(f"{x['n']} | {100 * x['score'] / x['n']:.1f} | {top}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="thư mục lượt chạy (có samples.jsonl)")
    ap.add_argument("--ids", type=int, default=5, help="số ID tiêu biểu mỗi nhóm (0: không in)")
    ap.add_argument("--by-app", action="store_true", help="thêm bảng theo app giữ lại của gold")
    ap.add_argument("--detail", action="store_true", help="thêm nhãn cấu trúc trong từng nhóm (vì sao sai)")
    ap.add_argument("--json", default=None, help="ghi số liệu ra file JSON (nhóm của từng mẫu sai)")
    a = ap.parse_args(argv)
    runs = [analyse(Path(p)) for p in a.runs]
    for r in runs:
        print(f"{r['name']}: n = {r['n']}, điểm {r['score']}")
    print()
    print(brief_table(runs))
    print()
    print(table(runs, a.ids))
    if a.detail:
        print("\nChi tiết theo nhóm (nhãn cấu trúc, số mẫu):\n")
        print(detail_text(runs))
    if a.by_app:
        print()
        print(app_table(runs))
    if len(runs) == 2 and set(runs[0]["rows"]) | set(runs[1]["rows"]):
        both = set(runs[0]["rows"]) & set(runs[1]["rows"])
        same = sum(1 for i in both if runs[0]["rows"][i] == runs[1]["rows"][i])
        print(f"\nSai ở cả hai lượt: {len(both)} mẫu (cùng nhóm lỗi: {same}); chỉ sai ở lượt 1: "
              f"{len(set(runs[0]['rows']) - both)}; chỉ sai ở lượt 2: {len(set(runs[1]['rows']) - both)}")
    if a.json:
        Path(a.json).write_text(json.dumps(
            [{"name": r["name"], "n": r["n"], "score": r["score"], "rows": r["rows"],
              "cats": {k: {x: v[x] for x in ("n", "lost", "asr", "rule")} for k, v in r["cats"].items()}} for r in runs],
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
