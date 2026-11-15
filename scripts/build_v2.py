"""Pro 15/11: dựng dữ liệu SFT v2 = dữ liệu 175/98f + bản lỗi ASR THẬT cho câu D1 (qua audio TTS) [+ bản lỗi của mẫu val].

Cùng nguyên tắc với bản lỗi theo fold của 175 (LM-51, build_sft.py --asr-only --asr-keep-entities-nbest):
  * mỗi câu có HAI bản cùng nhãn: bản sạch (đã có trong file gốc) và bản lỗi (thêm vào), không thay chỗ;
  * bản lỗi kèm khối "Bản nghe khác" như lúc thi (N-best 4 của ASR chính, ASR phụ đứng đầu nếu khác chữ);
  * thực thể chuỗi của call gold mà người dùng nói ra phải còn ở ít nhất một phương án, không thì bỏ câu đó.

tts  BASE.jsonl TTS_ASR.jsonl OUT.jsonl     : chỉ các dòng bản lỗi TTS-D1 (nối sau BASE bằng cat).
val  RUN_VAL_APP RUN_VAL_RANDOM OUT.jsonl   : bản lỗi của mẫu val từ transcripts.jsonl của hai lượt đo (ASR train_rest,
                                              chưa nghe audio val), dựng bằng build_sft như bản fold (chỉ cho bản full).
Chạy trong worktree lm51-asr (build_sft của 175):  PYTHONPATH=src:scripts .venv/bin/python build_v2.py ...
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

from lifemate.asr.metrics import fold, norm

Q = re.compile(r'(YÊU CẦU CUỐI CỦA NGƯỜI DÙNG \(bản chép lời\):\n)"(.*)"$', re.S)


def plain(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())


def gold_strings(msgs: list[dict]) -> list[str]:
    out = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        try:
            a = json.loads(m["content"])
        except ValueError:
            continue
        for v in (a.get("arguments") or {}).values():
            if isinstance(v, str) and 2 <= len(v) <= 40 and not re.match(r"^\d{4}-", v):
                out.append(v)
    return out


def tts(base, asr, out):
    heard = {r["src_id"]: r for r in map(json.loads, open(asr, encoding="utf-8"))}
    st = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for line in open(base, encoding="utf-8"):
            r = json.loads(line)
            h = heard.get(r["id"])
            if h is None:
                continue
            st["có audio TTS"] += 1
            u = r["messages"][1]["content"]
            m = Q.search(u)
            if not m or "Bản nghe khác" in u:
                st["bỏ: không tách được câu cuối"] += 1
                continue
            clean = m.group(2)
            final, alts = h["final"].strip(), [a for a in h["alts"] if a.strip()]
            if not final:
                st["bỏ: ASR rỗng"] += 1
                continue
            sec = (h.get("second") or "").strip()
            if sec and plain(sec) not in {plain(x) for x in [final, *alts]}:
                alts = [sec, *alts]
            if plain(final) == plain(clean) and not alts:
                st["bỏ: nghe đúng hệt, không có bản khác"] += 1
                continue
            hyps = [final, *alts]
            lost = [e for e in gold_strings(r["messages"]) if fold(e) in fold(clean) and all(norm(e) not in norm(x) for x in hyps)]
            if lost:
                st["bỏ: mọi phương án mất thực thể gold"] += 1
                continue
            new = u[:m.start()] + m.group(1) + f'"{final}"'
            if alts:
                new += "\n\nBản nghe khác của cùng câu: " + " | ".join(f'"{a}"' for a in alts)
            r = dict(r, id=r["id"] + "_tts", asr=True)
            r["messages"] = [r["messages"][0], dict(r["messages"][1], content=new), *r["messages"][2:]]
            st["ghi"] += 1
            st["top-1 khác chữ đúng"] += plain(final) != plain(clean)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(st, ensure_ascii=False))


def val(run_app, run_rand, out):
    import random

    import build_sft as B
    from lifemate import config, splits
    from lifemate.agent.agents import ReactJSON
    from lifemate.registry import build

    cfg = config.load(B.REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
    agent = build("agent", cfg["agent"])
    assert isinstance(agent, ReactJSON)
    heard = {}
    for d in (run_app, run_rand):
        for line in open(f"{d}/transcripts.jsonl", encoding="utf-8"):
            t = json.loads(line)
            heard[t["id"]] = t
    pool = [s for sub in ("val_app", "val_random") for s in splits.select(B.SPLIT, sub)]
    specs = B.data.tool_specs(B.SPLIT)
    st = Counter()
    rng = random.Random(2026)
    with open(out, "w", encoding="utf-8") as f:
        for s in pool:
            h = heard[s["id"]]
            hyps = [h["final"], *(h.get("alts", {}).get("final", []))]
            if any(all(norm(e) not in norm(x) for x in hyps) for e in B.entity_spans(s)):
                st["bỏ: mọi phương án mất thực thể gold"] += 1
                continue
            if B.guard_warnings(cfg, s, h):
                st["bỏ: gold bị middleware cảnh báo"] += 1
                continue
            calcs = B.calc_steps(s, h["final"], specs) if agent.calc else None
            msgs = B.conversation(agent, s, h, None, calcs)
            f.write(json.dumps({"id": s["id"] + "_valasr", "renamed": False, "asr": True, "messages": msgs},
                               ensure_ascii=False) + "\n")
            st["ghi"] += 1
    rng.random()
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    {"tts": tts, "val": val}[sys.argv[1]](*sys.argv[2:])
