"""Kiểm synth v3 và hai file train ghép (CPU):

1. Render byte-match trên dòng gốc: dựng lại các dòng train_rest sạch (bản chép gold) và dòng ASR-fold (cache config 181)
   bằng build_sft.conversation + config 10, so từng byte với train_para_v1_asrfold.jsonl.
2. Mọi dòng synth v3: phát lại gold trên ToolEnv thật dựng lại từ metadata (kết quả, danh sách call, state_hash, cảnh báo
   middleware), rồi render lại và so byte với synth_v3.jsonl.
3. Độ dài token (tokenizer Qwen3.5, định dạng scaffold của train_sft_q35r) < 8192 cho mọi dòng synth v3.
4. Không trùng app/tool của val_app (HELDOUT_APPS) và không trùng tên app train; hai file ghép = file gốc − synth cũ + v3.

    cd fable-eval/lm51-asr && LIFEMATE_DATA=... .venv/bin/python ../tools-pro/synth_v3/check_v3.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LM51 = HERE.parents[1]
SFT = LM51 / "sft"
TOK = LM51 / "models/qwen35-4b-text/tokenizer.json"
sys.path[:0] = [str(HERE), str(LM51 / "src"), str(LM51 / "scripts")]

import gen_v3 as G  # noqa: E402
import apps_v3 as A  # noqa: E402
import build_sft  # noqa: E402
from lifemate import data, splits  # noqa: E402

SCAFFOLD = "<think>\n\n</think>\n\n"


def load_cache(p: Path) -> dict:
    return {json.loads(x)["id"]: json.loads(x) for x in p.open(encoding="utf-8")}


def main():
    ctx = G.load_ctx(0)
    out = {}
    # 1. byte-match dòng gốc
    pool = {s["id"]: s for s in splits.select("train", "train_rest")}
    gold, asr = load_cache(G.GOLD), load_cache(G.ASR181)
    c = Counter()
    for ln in (SFT / "train_para_v1_asrfold.jsonl").open(encoding="utf-8"):
        r = json.loads(ln)
        if "asr" not in r or r.get("renamed") or r["id"] not in pool:
            continue
        tr = (asr if r["asr"] else gold)[r["id"]]
        msgs = build_sft.conversation(ctx.agent, pool[r["id"]], tr, None)
        ok = json.dumps(msgs, ensure_ascii=False) == json.dumps(r["messages"], ensure_ascii=False)
        c[("asr" if r["asr"] else "clean", ok)] += 1
    out["render_bytematch_originals"] = {f"{k}_{'ok' if ok else 'diff'}": v for (k, ok), v in sorted(c.items())}
    # 2. phát lại + render lại synth v3
    rows = {json.loads(x)["id"]: json.loads(x) for x in (SFT / "synth_v3.jsonl").open(encoding="utf-8")}
    samples = [json.loads(x) for x in (SFT / "synth_v3.samples.jsonl").open(encoding="utf-8")]
    bad_replay, bad_render, overlap = [], [], []
    real_apps = {a["id"] for a in data.catalog("train")["apps"]}
    held_tools = {
        t["name"].split(".", 1)[1]: set(t["parameters"].get("properties", {}))
        for t in data.tool_specs("train").values()
        if t["name"].split(".")[0] in A.HELDOUT
    }
    for s in samples:
        errs = G.replay(s, ctx.cfg.get("guards"))
        if errs:
            bad_replay.append((s["id"], errs[:2]))
        msgs = build_sft.conversation(ctx.agent, s, s["synth3"]["transcript"], None, specs=G.specs_of(s))
        if json.dumps(msgs, ensure_ascii=False) != json.dumps(rows[s["id"]]["messages"], ensure_ascii=False):
            bad_render.append(s["id"])
        for m in s["synth3"]["apps"]:
            app = m["names"]["app"]
            if app in real_apps or app in A.HELDOUT:
                overlap.append((s["id"], app))
            for t in A.catalog(A.BY_KEY[m["domain"]], m["names"])["tools"]:
                short = t["name"].split(".", 1)[1]
                if short in held_tools and set(t["parameters"]["properties"]) == held_tools[short]:
                    overlap.append((s["id"], t["name"]))
        # câu nói không được nhắc app đang có trong val_app
    out["synth_replay"] = {"rows": len(samples), "replay_fail": len(bad_replay), "examples": bad_replay[:5]}
    out["synth_render_bytematch"] = {"rows": len(samples), "diff": len(bad_render), "examples": bad_render[:5]}
    out["heldout_or_train_name_overlap"] = overlap[:10]
    out["heldout_or_train_name_overlap_count"] = len(overlap)
    # 3. độ dài token
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK))
    lens = []
    for r in rows.values():
        text = "".join(
            f"<|im_start|>{m['role']}\n"
            + (SCAFFOLD if m["role"] == "assistant" else "")
            + m["content"].strip()
            + "<|im_end|>\n"
            for m in r["messages"]
        )
        lens.append(len(tok.encode(text, add_special_tokens=False).ids))
    lens.sort()
    out["tokens"] = {
        "max": lens[-1],
        "p50": lens[len(lens) // 2],
        "p95": lens[int(0.95 * (len(lens) - 1))],
        "over_8192": sum(x >= 8192 for x in lens),
    }
    # 4. thành phần hai file ghép
    comp = {}
    for src, dst in [
        ("train_para_v1_asrfold.jsonl", "train_para_v1_asrfold_sv3.jsonl"),
        ("train_full_para_v1_asrfold_val.jsonl", "train_full_para_v1_asrfold_val_sv3.jsonl"),
    ]:
        a = [json.loads(x) for x in (SFT / src).open(encoding="utf-8")]
        b = [json.loads(x) for x in (SFT / dst).open(encoding="utf-8")]
        keep_a = [json.dumps(r, ensure_ascii=False) for r in a if r.get("synth") is None]
        keep_b = [
            json.dumps(r, ensure_ascii=False)
            for r in b
            if not str(r.get("synth", "")).startswith("v3:") and r.get("synth") is None
        ]
        v3b = [r for r in b if str(r.get("synth", "")).startswith("v3:")]
        comp[dst] = {
            "rows": len(b),
            "non_synth_identical_in_order": keep_a == keep_b,
            "old_synth_left": sum(1 for r in b if r.get("synth") and not str(r["synth"]).startswith("v3:")),
            "synth_v3_rows": len(v3b),
            "synth_v3_equal_file": [r["id"] for r in v3b] == list(rows),
        }
    out["spliced_files"] = comp
    ok = (
        all(k.endswith("_ok") for k in out["render_bytematch_originals"])
        and not bad_replay
        and not bad_render
        and not overlap
        and out["tokens"]["over_8192"] == 0
        and all(
            v["non_synth_identical_in_order"] and v["old_synth_left"] == 0 and v["synth_v3_equal_file"]
            for v in comp.values()
        )
    )
    out["ALL_OK"] = ok
    (SFT / "synth_v3.check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
