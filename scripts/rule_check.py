"""Kiểm luật tầng D trên train trước khi đo (BRIEF_LM47 mục 000): luật bật trên lời gold và call gold của train.

Với mỗi luật (cờ guards), mỗi mẫu train: phát call gold qua GuardedEnv với guards của config 195 cộng luật đó, lời là
chữ gold. Đếm:

    sửa_gold      số call gold luật đổi (luật sai: gold đã đúng mà luật đổi)
    áp_dụng       số lần điều kiện của luật thoả trên call gold (luật "có ý kiến" về giá trị): đổi hoặc giữ nguyên
    đúng          áp_dụng − sửa_gold; tỉ lệ đúng = đúng / áp_dụng (cần ≥ 95%)
    cứu           với call gold bị làm hỏng theo kiểu LLM hay hỏng (đổi phút, đổi sáng/chiều), số lần luật
                  đưa về đúng gold: luật có ích tới đâu

    python scripts/rule_check.py [--subset train_rest|all] [--rules clock_face] [--show 5]
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime

from lifemate import data, fixes, normalize, splits
from lifemate.guards import GuardedEnv

SPLIT = "train"
BASE = {"clock_fix": True, "money_k": True, "span_min": True, "ask_missing": True, "pick_one": True}


def replay(sample: dict, calls: list, guards: dict) -> tuple[list, list]:
    env = data.tool_env(SPLIT, sample["id"])
    g = GuardedEnv(env, SPLIT, sample, sample["audio_text"], **guards)
    for c in calls:
        g.call(c["name"], copy.deepcopy(c["arguments"]))
    g.finish()
    return env.get_tool_calls(), [n for rec in g.log for n in rec.get("fixes") or []]


def perturb_clock(c: dict, props: dict) -> list[dict]:
    """Các bản hỏng của một call: phút lệch 20, hoặc giờ lệch 12 (sáng ↔ chiều), ở tham số ngày giờ không phải giờ kết thúc."""
    out = []
    for k, v in (c.get("arguments") or {}).items():
        if not (isinstance(v, str) and fixes._ISO.match(v) and "T" in v and fixes._temporal_prop(props.get(k, {}))
                and not fixes._end_like(k)):
            continue
        t = datetime.fromisoformat(v)
        for bad in (t.replace(minute=(t.minute + 20) % 60), t.replace(hour=(t.hour + 12) % 24)):
            out.append({**c, "arguments": {**c["arguments"], k: bad.isoformat()}})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="train_rest")
    ap.add_argument("--rules", default="clock_face")
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args(argv)
    pool = splits.select(SPLIT, a.subset)
    specs = data.tool_specs(SPLIT)
    for rule in [r.strip() for r in a.rules.split(",") if r.strip()]:
        st, shown = Counter(), 0
        for s in pool:
            gold = s["label"]["tool_calls"]
            base, _ = replay(s, gold, BASE)
            mine, notes = replay(s, gold, {**BASE, rule: True})
            mine_notes = [n for n in notes if n.startswith(rule)]
            changed = sum(1 for x, y in zip(base, mine) if x != y)
            st["mẫu"] += 1
            st["sửa_gold"] += changed
            if changed and shown < a.show:
                shown += 1
                print(f"  SAI {s['id']}: {s['audio_text'][:120]} | {mine_notes[:2]}")
            # cứu: làm hỏng từng call ghi của gold theo kiểu LLM hay hỏng, xem luật có đưa về đúng không
            for i, c in enumerate(gold):
                props = (specs.get(c["name"], {}).get("parameters") or {}).get("properties") or {}
                if specs.get(c["name"], {}).get("kind") != "write":
                    continue
                bads = perturb_clock(c, props) if rule == "clock_face" else perturb_phone(c, props) if rule == "phone_fix" else []
                for bad in bads:
                    calls = gold[:i] + [bad] + gold[i + 1:]
                    b0, _ = replay(s, calls, BASE)
                    b1, n1 = replay(s, calls, {**BASE, rule: True})
                    if any(n.startswith(rule) for n in n1):
                        st["áp_dụng_trên_bản_hỏng"] += 1
                        st["cứu"] += b1 == base and b0 != base
                        st["sửa_mà_vẫn_sai"] += b1 != base
            # áp dụng trên gold: luật có ý kiến về call gold (điều kiện thoả) mà giữ nguyên
            if rule == "clock_face":
                st["áp_dụng"] += sum(1 for c in gold if specs.get(c["name"], {}).get("kind") == "write"
                                    and applicable_clock(s, c, specs))
            elif rule == "phone_fix" and len(normalize.phones(s["audio_text"])) == 1:
                st["áp_dụng"] += sum(1 for c in gold if specs.get(c["name"], {}).get("kind") == "write"
                                    for k in (c.get("arguments") or {})
                                    if fixes._is_phone_param(k, ((specs[c["name"]].get("parameters") or {}).get("properties") or {}).get(k, {})))
        ok = st["áp_dụng"] - st["sửa_gold"]
        print(json.dumps({"luật": rule, "tập": a.subset, **st, "tỉ_lệ_đúng_trên_gold": round(ok / st["áp_dụng"], 4) if st["áp_dụng"] else None},
                         ensure_ascii=False))


def perturb_phone(c: dict, props: dict) -> list[dict]:
    """Số điện thoại hỏng kiểu LLM hay hỏng: thiếu một chữ số "0" khi đọc "không không không"."""
    out = []
    for k, v in (c.get("arguments") or {}).items():
        if isinstance(v, str) and fixes._is_phone_param(k, props.get(k, {})) and "00" in v:
            out.append({**c, "arguments": {**c["arguments"], k: v.replace("00", "0", 1)}})
    return out


def applicable_clock(s: dict, c: dict, specs: dict) -> bool:
    props = (specs.get(c["name"], {}).get("parameters") or {}).get("properties") or {}
    norm = fixes._norm(s["audio_text"])
    if fixes._OFFSET.search(norm) or fixes._REL_TIME.search(norm):
        return False
    cs = fixes.spoken_clocks(s["audio_text"])
    faces = {(int(x[:2]) % 12, x[3:]) for x in cs}
    if len(faces) != 1:
        return False
    (fh, _), = faces
    for k, v in (c.get("arguments") or {}).items():
        m = fixes._ISO.match(v) if isinstance(v, str) else None
        if m and m.group(2) and fixes._temporal_prop(props.get(k, {})) and not fixes._end_like(k) and int(m.group(2)[:2]) % 12 == fh:
            return True
    return False


if __name__ == "__main__":
    main()
