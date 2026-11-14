"""In gọn chỗ lệch của từng mẫu sai trong một lượt chạy: nhóm lỗi (scripts/unseen_errors.py), lời nghe được và lời gold,
rồi các call lệch (tham số dự đoán → gold). Dùng để tìm luật tất định; chạy trên PC, in ra màn hình, không ghi file.

    python scripts/error_diffs.py RUN_DIR [--cats "sai giá trị nói ra,thừa / thiếu tham số"] [--max 80]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from lifemate import data
from lifemate.scoring import match

SPLIT = "train"
spec = importlib.util.spec_from_file_location("ue", Path(__file__).with_name("unseen_errors.py"))
ue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ue)


def short(c: dict) -> str:
    return f"{c['name']} {json.dumps(c.get('arguments') or {}, ensure_ascii=False)}"


def diff_args(p: dict, g: dict) -> str:
    pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
    out = []
    for k in sorted(set(pa) | set(ga)):
        if pa.get(k) != ga.get(k):
            out.append(f"{k}: {json.dumps(pa.get(k, '∅'), ensure_ascii=False)} → {json.dumps(ga.get(k, '∅'), ensure_ascii=False)}")
    return "; ".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cats", default="")
    ap.add_argument("--max", type=int, default=80)
    a = ap.parse_args(argv)
    run = Path(a.run)
    want = {x.strip() for x in a.cats.split(",") if x.strip()}
    alts = {}
    for x in (run / "transcripts.jsonl").open(encoding="utf-8"):
        if x.strip():
            t = json.loads(x)
            alts[t["id"]] = (t.get("alts") or {}).get("final") or []
    by_id = data.by_id(SPLIT)
    fuzzy = data.fuzzy_params(SPLIT)
    n = 0
    for x in (run / "samples.jsonl").open(encoding="utf-8"):
        r = json.loads(x)
        if r["score"] == 1.0:
            continue
        s = by_id[r["id"]]
        cat = ue.classify(r, s)
        if want and cat not in want:
            continue
        n += 1
        if n > a.max:
            break
        print(f"### {r['id']} [{cat}] now={s['context']['current_time'][:16]} score={r['score']:.2f}")
        print(f"  nghe: {r.get('transcript')}")
        if r.get("reference") != r.get("transcript"):
            print(f"  gold: {r.get('reference')}")
        for alt in alts.get(r["id"], [])[:2]:
            if alt not in (r.get("transcript"), r.get("reference")):
                print(f"  khác: {alt}")
        pred, gold = r["tool_calls"], r["gold_calls"]
        pairs, used = match(pred, gold, fuzzy)
        pm = {i for i, _ in pairs}
        pu = [c for i, c in enumerate(pred) if i not in pm]
        gu = [c for j, c in enumerate(gold) if j not in used]
        for g in gu:
            p = next((c for c in pu if c["name"] == g["name"]), None)
            if p is not None:
                pu.remove(p)
                print(f"  ≠ {g['name']}: {diff_args(p, g)}")
            else:
                print(f"  thiếu: {short(g)}")
        for p in pu:
            print(f"  thừa: {short(p)}")
    print(f"({n} mẫu)")


if __name__ == "__main__":
    main()
