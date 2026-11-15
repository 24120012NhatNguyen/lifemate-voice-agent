"""Ghép synth v3 vào hai file train (thay các dòng synth_* cũ), ghi thống kê và sha256.

cd fable-eval/lm51-asr && LIFEMATE_DATA=... .venv/bin/python ../tools-pro/synth_v3/finalize_v3.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LM51 = HERE.parents[1]
SFT = LM51 / "sft"
sys.path[:0] = [str(HERE), str(LM51 / "src"), str(LM51 / "scripts")]

import gen_v3 as G  # noqa: E402
import measure_real as M  # noqa: E402
import apps_v3 as A  # noqa: E402
from lifemate import data, splits  # noqa: E402

PAIRS = [
    ("train_para_v1_asrfold.jsonl", "train_para_v1_asrfold_sv3.jsonl"),
    ("train_full_para_v1_asrfold_val.jsonl", "train_full_para_v1_asrfold_val_sv3.jsonl"),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def splice(src: Path, dst: Path, v3: list[str]) -> dict:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    first, kept, removed = None, [], 0
    for ln in lines:
        if json.loads(ln).get("synth") is not None:
            removed += 1
            if first is None:
                first = len(kept)
            continue
        kept.append(ln)
    first = len(kept) if first is None else first
    out = kept[:first] + v3 + kept[first:]
    dst.write_text("".join(out), encoding="utf-8")
    return {
        "src": src.name,
        "src_rows": len(lines),
        "removed_synth_v1": removed,
        "added_synth_v3": len(v3),
        "rows": len(out),
        "synth_share": round(len(v3) / len(out), 4),
    }


def stats(samples: list[dict]) -> dict:
    C, N, H, E, NA, sub, nw = Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    real_calls, pn = 0, 0
    for s in samples:
        sp = G.specs_of(s)
        C[M.action_class(s, sp)] += 1
        N[min(len(s["label"]["tool_calls"]), 6)] += 1
        H[min(sum(t["role"] == "user" for t in s["history"]), 3)] += 1
        z = s["synth3"]["noise"]
        E[z["entity_mode"]] += 1
        NA[z["n_alts"]] += 1
        pn += z["primary_noise"]
        sub[s["synth3"]["sub"]] += 1
        real_calls += any(
            c["name"].split(".")[0] in {a["id"] for a in data.catalog("train")["apps"]} - {"sys"}
            for c in s["label"]["tool_calls"]
        )
        import re

        t = s["audio_text"]
        nw["digit_groups"] += len(re.findall(r"\d+", t))
        nw["number_words"] += len(re.findall(r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|trăm)\b", t))
    n = len(samples)
    pct = lambda c: {str(k): f"{v} ({100 * v / n:.1f}%)" for k, v in sorted(c.items(), key=lambda x: str(x[0]))}
    # đặc trưng schema của các app tổng hợp đã dùng
    tools = {}
    for s in samples:
        for m in s["synth3"]["apps"]:
            for t in A.catalog(A.BY_KEY[m["domain"]], m["names"])["tools"]:
                tools[t["name"]] = t
    T = list(tools.values())
    P = [(t, p, sc) for t in T for p, sc in t["parameters"]["properties"].items()]
    ref_like = [
        x
        for x in P
        if "lấy từ" in x[2].get("description", "") or "x-ref" in x[2] or "phải có trong" in x[2].get("description", "")
    ]
    return {
        "n": n,
        "action": pct(C),
        "n_calls": pct(N),
        "history_user_turns": pct(H),
        "entity_noise": pct(E),
        "n_alts": pct(NA),
        "alts_block_share": round(sum(v for k, v in NA.items() if k > 0) / n, 4),
        "primary_noise_share": round(pn / n, 4),
        "rows_calling_real_train_app": real_calls,
        "sub_kinds": dict(sub.most_common()),
        "numbers_in_clean_text": dict(nw),
        "schema": {
            "apps": len({t["app"] for t in T}),
            "tools": len(T),
            "read_share": round(sum(t["kind"] == "read" for t in T) / len(T), 3),
            "params": len(P),
            "optional_share": round(sum(p not in t["parameters"]["required"] for t, p, _ in P) / len(P), 3),
            "enum_share": round(sum("enum" in sc for *_, sc in P) / len(P), 3),
            "fuzzy_share": round(sum(sc.get("x-match") == "fuzzy" for *_, sc in P) / len(P), 3),
            "xref_share": round(sum("x-ref" in sc for *_, sc in P) / len(P), 3),
            "ref_like_params": len(ref_like),
            "ref_like_without_id_suffix_or_xref": sorted(
                {f"{t['name']}.{p}" for t, p, sc in ref_like if not p.endswith("_id") or "x-ref" not in sc}
            ),
            "cross_app_xref": sorted(
                {f"{t['name']}.{p}" for t, p, sc in P if sc.get("x-ref", "").split(".")[0] not in ("", t["app"])}
            ),
            "no_param_tools": sum(not t["parameters"]["properties"] for t in T),
        },
    }


def main():
    v3_path = SFT / "synth_v3.jsonl"
    v3 = v3_path.read_text(encoding="utf-8").splitlines(keepends=True)
    samples = [json.loads(x) for x in (SFT / "synth_v3.samples.jsonl").open(encoding="utf-8")]
    report = {"synth_v3": {"path": str(v3_path), "rows": len(v3)}, "files": []}
    for src, dst in PAIRS:
        info = splice(SFT / src, SFT / dst, v3)
        report["files"].append(info)
    report["stats_synth_v3"] = stats(samples)
    report["stats_real_train_rest_action"] = real_action()
    sums = {}
    for p in [v3_path, SFT / "synth_v3.samples.jsonl", *[SFT / d for _, d in PAIRS]]:
        sums[p.name] = sha(p)
        (p.parent / (p.name + ".sha256")).write_text(f"{sums[p.name]}  {p.name}\n")
    report["sha256"] = sums
    (SFT / "synth_v3.stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def real_action() -> dict:
    specs = data.tool_specs("train")
    pool = splits.select("train", "train_rest")
    C = Counter(M.action_class(s, specs) for s in pool)
    n = len(pool)
    return {k: f"{v} ({100 * v / n:.1f}%)" for k, v in sorted(C.items())}


if __name__ == "__main__":
    main()
