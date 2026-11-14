"""Kiểm `resolve.pick` (luật ask_resolve) trên call gold của train, theo BRIEF_LM47 mục 000.

    dương   call ghi của gold có tham số ID lấy từ một danh sách vừa đọc (≥ 2 mục, ID chưa có trong lịch sử): pick phải
            ra đúng ID gold hoặc không chọn
    âm      gold hỏi lại một tham số ID sau khi đã đọc danh sách nguồn: pick không được chọn

    đúng = dương chọn đúng; sai = dương chọn sai + âm bị chọn; tỉ lệ đúng = đúng / (đúng + sai) (cần ≥ 95%)

    python scripts/resolve_check.py [--subset train_rest] [--min 2] [--margin 2] [--show 5]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from lifemate import data, resolve, splits

SPLIT = "train"
ASK = "sys.ask_missing_info"


def id_param(tool: str, key: str, specs: dict) -> bool:
    p = ((specs.get(tool, {}).get("parameters") or {}).get("properties") or {}).get(key, {})
    return bool(p.get("x-ref")) or key.endswith("_id")


def history_ids(s: dict) -> set:
    out = set()
    for t in s.get("history", []):
        out |= {x.get("id") for r in t.get("tool_results") or [] for lst in resolve.lists_with_ids(r) for x in lst}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="train_rest")
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--loose", action="store_true", help="không đòi lời nói chỉ nhắc đúng một mục")
    ap.add_argument("--names", action="store_true", help="chỉ chấm trên các trường tên (name, title, label, subject)")
    a = ap.parse_args(argv)
    specs = data.tool_specs(SPLIT)
    st, shown = Counter(), Counter()
    for s in splits.select(SPLIT, a.subset):
        calls, results = s["label"]["tool_calls"], s["label"].get("tool_results") or []
        known = history_ids(s)
        for i, c in enumerate(calls):
            if c["name"] == ASK:
                ps = (c.get("arguments") or {}).get("params") or []
                tool = str((c.get("arguments") or {}).get("tool_name"))
                if len(ps) != 1 or not id_param(tool, str(ps[0]), specs):
                    continue
                lists = [lst for r in results[:i] for lst in resolve.lists_with_ids(r) if len(lst) >= 2]
                if not lists:
                    continue
                st["âm"] += 1
                hit = resolve.pick(lists[-1], s["audio_text"], a.min, a.margin, not a.loose, resolve.NAME_KEYS if a.names else None)
                if hit is not None:
                    st["âm_bị_chọn"] += 1
                    if shown["âm"] < a.show:
                        shown["âm"] += 1
                        print(f"  ÂM BỊ CHỌN {s['id']}: {s['audio_text'][:110]} → {hit['id']}")
                continue
            if specs.get(c["name"], {}).get("kind") != "write":
                continue
            for k, v in (c.get("arguments") or {}).items():
                if not isinstance(v, str) or not id_param(c["name"], k, specs) or v in known:
                    continue
                lists = [lst for r in results[:i] for lst in resolve.lists_with_ids(r)
                         if len(lst) >= 2 and any(x.get("id") == v for x in lst)]
                if not lists:
                    continue
                st["dương"] += 1
                hit = resolve.pick(lists[-1], s["audio_text"], a.min, a.margin, not a.loose, resolve.NAME_KEYS if a.names else None)
                if hit is None:
                    st["dương_không_chọn"] += 1
                elif hit["id"] == v:
                    st["dương_đúng"] += 1
                else:
                    st["dương_sai"] += 1
                    if shown["sai"] < a.show:
                        shown["sai"] += 1
                        print(f"  SAI {s['id']} {c['name']}.{k}: gold {v}, chọn {hit['id']} | {s['audio_text'][:100]}")
    wrong = st["dương_sai"] + st["âm_bị_chọn"]
    print(json.dumps({"tập": a.subset, "min": a.min, "margin": a.margin, "exclusive": not a.loose, "names": a.names, **st,
                      "tỉ_lệ_đúng_khi_chọn": round(st["dương_đúng"] / (st["dương_đúng"] + wrong), 4) if st["dương_đúng"] + wrong else None,
                      "tỉ_lệ_chọn_ở_dương": round(st["dương_đúng"] / st["dương"], 3) if st["dương"] else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
