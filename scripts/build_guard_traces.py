"""LM-92: SFT traces that teach the agent to read the notes guards.py injects into tool observations.

The production SFT data (build_sft.py) has no row with a guard note, because gold samples that trigger guard warnings
are dropped. At inference (config 240/255, prod guards) two guards rewrite observations:

  (a) empty_retry  a filtered read returns an empty list -> the guard re-calls without the string filter and returns the
                   full list with a note ("không có mục nào khớp ...; chọn đúng mục người dùng nói ...").
  (b) ask_resolve  an ask for a single ID parameter of a write tool whose source list was read this turn and the
                   utterance names exactly one item -> the guard returns {"ok": true, "note": "không cần hỏi lại ..."}.

For each eligible gold sample this script builds one trace per type:
  gold steps before the bad step  ->  the bad assistant step (marked "train": false: in context, no loss)
  ->  the observation the REAL GuardedEnv returns (with the exact note)  ->  the remaining gold steps (trained) -> stop.

System/user prompts and step formatting are those of scripts/build_sft.py (config 10), verified byte-identical to the
production rows of sft/train_para_v1*.jsonl. Guard behaviour is the real GuardedEnv of this tree with the guard flags of
config 240 (production). A case is kept only when, under those guards, the gold prefix and continuation produce exactly
the gold tool results and the bad step returns the expected note.

(a) filter value: the fold-ASR transcript's mis-hearing of the gold name when it differs (row uses the ASR transcript),
    else a controlled consonant/final/tone perturbation applied to the name both in the utterance and in the filter.

Run from the pro-rules tree (branch pro/docker-safety):
    cd fable-work/pro-rules && LIFEMATE_DATA=... PYTHONPATH=src .venv/bin/python \
        ../../fable-eval/tools-pro/build_guard_traces.py --subset train_rest --out ../../fable-eval/pro-threshold/sft/guard_traces_rest.jsonl
    ... --subset all --out ../../fable-eval/pro-threshold/sft/guard_traces_full.jsonl
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from lifemate import config, data, resolve, splits
from lifemate.asr.base import run_asr
from lifemate.config import REPO
from lifemate.guards import GuardedEnv, _ISO_DATE
from lifemate.registry import build

sys.path.insert(0, str(REPO / "scripts"))
import build_sft as B  # noqa: E402  (same rendering as production SFT)

SPLIT = "train"
ASK = "sys.ask_missing_info"
RENDER_CFG = "configs/experiments/10_qwen3-4b_gold_react.yaml"
GUARD_CFG = "configs/repro-guard-240.yaml"
NAME_PARAMS = ("query", "name", "keyword", "search", "title", "room", "location", "near", "merchant", "provider",
               "program", "contact", "store", "place")


# ---------------------------------------------------------------- text helpers
def cfold(s: str) -> str:
    """Length-preserving fold (NFC input): lowercase, strip diacritics, đ -> d. One char in, one char out."""
    out = []
    for ch in s:
        b = unicodedata.normalize("NFD", ch.lower())
        b = "".join(c for c in b if not unicodedata.combining(c)) or ch
        out.append("d" if b == "đ" else b[0])
    return "".join(out)


def find_span(text: str, value: str) -> tuple[int, int] | None:
    """Span of value in text, diacritic/case-insensitive, on word boundaries."""
    t, v = cfold(unicodedata.normalize("NFC", text)), cfold(unicodedata.normalize("NFC", value))
    for m in re.finditer(re.escape(v), t):
        a, b = m.span()
        if (a == 0 or not t[a - 1].isalnum()) and (b == len(t) or not t[b].isalnum()):
            return a, b
    return None


INITIALS = [("tr", "ch"), ("ch", "tr"), ("gi", "d"), ("d", "gi"), ("s", "x"), ("x", "s"), ("n", "l"), ("l", "n"),
            ("v", "d"), ("b", "v"), ("r", "g"), ("h", "kh"), ("kh", "h"), ("m", "n"), ("t", "th"), ("th", "t"),
            ("ph", "b"), ("q", "c"), ("k", "c"), ("c", "k"), ("ng", "nh"), ("nh", "ng")]
FINALS = [("ng", "n"), ("nh", "n"), ("n", "ng"), ("c", "t"), ("t", "c"), ("i", "y"), ("y", "i"), ("m", "n")]
TONES = "̣̀́̃̉"  # huyền sắc ngã hỏi nặng


def _retone(syl: str, rng: random.Random) -> str | None:
    """Same syllable with a different tone (or none): the tone mark moves on the vowel that carries it."""
    d = unicodedata.normalize("NFD", syl)
    cur = [k for k, c in enumerate(d) if c in TONES]
    base = "".join(c for c in d if c not in TONES)
    opts = [t for t in TONES if t not in [d[k] for k in cur]] + ([""] if cur else [])
    rng.shuffle(opts)
    vowels = [k for k, c in enumerate(base) if c.lower() in "aeiouy"]
    if not vowels:
        return None
    if cur:  # position in base of the toned vowel
        k = cur[0] - 1
        v = max(x for x in vowels if x <= k) if any(x <= k for x in vowels) else vowels[-1]
    else:
        v = vowels[-2] if len(vowels) >= 2 and base[-1].lower() in "aeiouy" else vowels[-1]
    j = v + 1
    while j < len(base) and unicodedata.combining(base[j]):
        j += 1
    return unicodedata.normalize("NFC", base[:j] + opts[0] + base[j:])


def perturb_options(span: str, rng: random.Random) -> list[str]:
    """Plausible mis-hearings of a name span: change one syllable (initial, final consonant, or tone)."""
    words = span.split(" ")
    idx = [i for i, w in enumerate(words) if w[:1].isupper()] or list(range(len(words)))
    idx = sorted(idx, key=lambda i: -i)  # proper name part is usually last
    out = []
    for i in idx:
        w = words[i]
        low, cap = w.lower(), w[:1].isupper()
        cands = []
        for a, b in INITIALS:
            if low.startswith(a) and not (a == "n" and low.startswith(("ng", "nh"))) and not (a == "t" and low.startswith(("th", "tr"))) \
                    and not (a == "c" and low.startswith("ch")) and not (a == "k" and low.startswith("kh")) \
                    and not (a == "g" and low.startswith("gi")) and not (a == "p" and low.startswith("ph")):
                rest = low[len(a):]
                if (b == "k") != (rest[:1] in ("i", "e", "ê", "y", "í", "ì", "é", "è", "ế", "ề")):
                    continue  # Vietnamese spelling: k only before i/e/ê/y, c elsewhere
                cands.append(b + rest)
        for a, b in FINALS:
            if low.endswith(a) and len(low) > len(a) + 1 and not (a == "n" and low.endswith(("ng", "nh"))):
                cands.append(low[: -len(a)] + b)
        rng.shuffle(cands)
        t = _retone(low, rng)
        if t and t != low:
            cands.append(t)
        for c in cands:
            c = c[:1].upper() + c[1:] if cap else c
            out.append(" ".join(words[:i] + [c] + words[i + 1:]))
    seen, res = set(), []
    for o in out:
        if o != span and o not in seen:
            seen.add(o)
            res.append(o)
    return res


def asr_variant(clean: str, heard: str, value: str) -> str | None:
    """The fold-ASR words aligned to the gold value's words in the clean utterance, if they differ."""
    sp = find_span(clean, value)
    if not sp:
        return None
    cw = [(m.start(), m.end()) for m in re.finditer(r"[^\W_]+", clean)]
    hw = [m.group() for m in re.finditer(r"[^\W_]+", heard)]
    ci = [k for k, (a, b) in enumerate(cw) if a >= sp[0] and b <= sp[1]]
    if not ci:
        return None
    sm = difflib.SequenceMatcher(a=[cfold(clean[a:b]) for a, b in cw], b=[cfold(w) for w in hw], autojunk=False)
    lo, hi = None, None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i2 <= ci[0] or i1 > ci[-1]:
            continue
        if tag == "equal":
            a, b = j1 + max(ci[0] - i1, 0), j1 + min(ci[-1] - i1 + 1, i2 - i1)
        elif i2 - i1 == j2 - j1:  # same length: word-for-word substitution
            a, b = j1 + max(ci[0] - i1, 0), j1 + min(ci[-1] - i1 + 1, i2 - i1)
        elif i1 >= ci[0] and i2 <= ci[-1] + 1:  # block inside the value span
            a, b = j1, j2
        else:  # block straddles the span edge with unequal length: alignment is ambiguous
            return None
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    if lo is None or hi <= lo or hi - lo > len(ci) + 2:
        return None
    var = " ".join(hw[lo:hi])
    if cfold(var) == cfold(value) or not var.strip():
        return None
    return var


# ---------------------------------------------------------------- guard replay
def free_string(spec: dict, p: str) -> bool:
    params = spec.get("parameters") or {}
    prop = (params.get("properties") or {}).get(p) or {}
    keep = set(params.get("required") or [])
    for grp in params.get("x-one-of") or []:
        keep |= set(grp) if isinstance(grp, list) else {grp}
    return prop.get("type") == "string" and "enum" not in prop and "format" not in prop and p not in keep


def genv(sample, tr, guards):
    return GuardedEnv(data.tool_env(SPLIT, sample["id"]), SPLIT, sample, tr["final"], turns=tr.get("turns"), **guards)


def replay(env, calls, results) -> bool:
    """Run gold calls through the guarded env; True when every observation equals the gold result."""
    for c, r in zip(calls, results):
        if env.call(c["name"], c["arguments"]) != r:
            return False
    return True


def note_empty_retry(name: str, clean: dict, wide: dict) -> str:
    """Independent copy of the guards.py empty_retry note formatter (checked against the real guard output)."""
    dropped = ", ".join(f"{k}={clean[k]!r}" for k in clean if k not in wide)
    return (f"không có mục nào khớp {dropped}; đây là kết quả gọi lại {name} không lọc, "
            "chọn đúng mục người dùng nói (tên có thể bị nghe sai)")


# ---------------------------------------------------------------- rendering
def render(agent, sample, tr, steps) -> list[dict]:
    """steps: [(name, arguments, observation, train)], rendered like build_sft.conversation, then a stop."""
    msgs = B.conversation(agent, {**sample, "label": {"tool_calls": [], "tool_results": []}}, tr, None)[:2]
    for name, args, obs, train in steps:
        m = {"role": "assistant", "content": B.action(name, args)}
        if not train:
            m["train"] = False
        msgs.append(m)
        msgs.append({"role": "user", "content": f"KẾT QUẢ {name}: {agent._obs(obs)}"})
    msgs.append({"role": "assistant", "content": B.action()})
    return msgs


def gold_steps(sample, lo=0, hi=None):
    calls, res = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    return [(c["name"], c["arguments"], r, True) for c, r in list(zip(calls, res))[lo:hi]]


# ---------------------------------------------------------------- trace builders
def trace_empty_retry(sample, tr_clean, tr_asr, guards, specs, rng, stats):
    calls, res = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    for i, c in enumerate(calls):
        spec = specs.get(c["name"], {})
        if spec.get("kind") != "read" or not any(isinstance(v, list) and v for v in res[i].values()):
            continue
        props = (spec.get("parameters") or {}).get("properties") or {}
        cands = []  # (param, correct value, base args without the param)
        for p, v in (c["arguments"] or {}).items():
            if free_string(spec, p) and isinstance(v, str) and not _ISO_DATE.match(v):
                cands.append((p, v, {k: x for k, x in c["arguments"].items() if k != p}))
        if not cands:  # gold reads the whole list: the model adds a name filter of the item it then uses
            later = {v for c2 in calls[i + 1:] for v in (c2["arguments"] or {}).values() if isinstance(v, str)}
            items = [x for v in res[i].values() if isinstance(v, list) for x in v if isinstance(x, dict) and x.get("id") in later]
            for p in (p for p in props if free_string(spec, p) and items):
                if isinstance(items[0].get(p), str):  # filter param named like an item field (room, location...)
                    nm = items[0][p]
                elif any(k in p for k in NAME_PARAMS):
                    nm = next((items[0][k] for k in ("name", "title", "label") if isinstance(items[0].get(k), str)), None)
                else:
                    nm = None
                if nm:
                    cands.append((p, nm, dict(c["arguments"] or {})))
        for p, v, base in cands:
            # variant: real fold-ASR mis-hearing first, then perturbation of the span heard in the clean utterance
            options = []
            if tr_asr is not None:
                var = asr_variant(sample["audio_text"], tr_asr["final"], v)
                if var:
                    options.append(("asr", var, tr_asr))
            sp = find_span(tr_clean["final"], v)
            if sp:
                span = tr_clean["final"][sp[0]:sp[1]]
                for var in perturb_options(span, rng)[:12]:
                    tr2 = {**copy.deepcopy(tr_clean), "final": tr_clean["final"][:sp[0]] + var + tr_clean["final"][sp[1]:]}
                    tr2.pop("alts", None)
                    options.append(("perturb", var, tr2))
            for src, var, tr in options:
                bad = {**base, p: var} if p in (c["arguments"] or {}) else {**base, p: var}
                # keep the gold key order: replace the value in place when the gold call had the filter
                if p in (c["arguments"] or {}):
                    bad = {k: (var if k == p else x) for k, x in c["arguments"].items()}
                env = genv(sample, tr, guards)
                if not replay(env, calls[:i], res[:i]):
                    stats["a: gold prefix differs under prod guards"] += 1
                    return None
                # the correct filter must find the item (the param really filters by name)
                if p not in (c["arguments"] or {}):
                    probe = env.env.call(c["name"], {**base, p: v})  # raw env, no guard bookkeeping
                    if not (isinstance(probe, dict) and any(isinstance(x, list) and x for x in probe.values())):
                        break
                    env = genv(sample, tr, guards)
                    replay(env, calls[:i], res[:i])
                obs = env.call(c["name"], bad)
                rec = env.log[-1] if env.log else {}
                if not (isinstance(obs, dict) and str(obs.get("note", "")).startswith("không có mục nào khớp")):
                    stats[f"a: guard did not fire ({src})"] += 1
                    continue
                clean_args = env.log[-2]["arguments"]
                wide = rec["arguments"]
                assert obs["note"] == note_empty_retry(c["name"], clean_args, wide), "note formatter mismatch"
                stats["a: note equal to independent formatter"] += 1
                if "warnings" in obs:
                    stats["a: bad step had warnings"] += 1
                    continue
                if not replay(env, calls[i + 1:], res[i + 1:]):
                    stats["a: gold continuation differs"] += 1
                    continue
                steps = gold_steps(sample, 0, i) + [(c["name"], bad, obs, False)] + gold_steps(sample, i + 1)
                return {"type": "empty_retry", "src": src, "step": i, "tool": c["name"], "param": p, "gold_value": v,
                        "bad_value": var, "tr": tr, "steps": steps, "with_filter_in_gold": p in (c["arguments"] or {})}
    return None


def trace_ask_resolve(sample, tr, guards, specs, stats):
    calls, res = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    asked = {(c["arguments"] or {}).get("tool_name") for c in calls if c["name"] == ASK}
    for k, c in enumerate(calls):
        spec = specs.get(c["name"], {})
        if spec.get("kind") != "write" or c["name"] in asked or k == 0:
            continue
        props = (spec.get("parameters") or {}).get("properties") or {}
        for p, v in (c["arguments"] or {}).items():
            if not isinstance(v, str) or not ((props.get(p) or {}).get("x-ref") or p.endswith("_id")):
                continue
            env = genv(sample, tr, guards)
            if not replay(env, calls[:k], res[:k]):
                stats["b: gold prefix differs under prod guards"] += 1
                return None
            ask = {"tool_name": c["name"], "params": [p]}
            obs = env.call(ASK, ask)
            note = obs.get("note", "") if isinstance(obs, dict) else ""
            if not note.startswith("không cần hỏi lại"):
                stats["b: guard did not fire"] += 1
                continue
            if f"{p}={v} (" not in note:
                stats["b: guard id differs from gold"] += 1
                continue
            item = next((x for r in res[:k] for lst in resolve.lists_with_ids(r) for x in lst
                         if x.get(p) == v or (not isinstance(x.get(p), str) and x.get("id") == v)), None)
            if item is not None:  # independent copy of the guards.py ask_resolve formatter
                label = " ".join(str(item[q]) for q in item if any(n in q for n in resolve.NAME_KEYS) and isinstance(item[q], str))
                expect = (f"không cần hỏi lại: lời người dùng gọi đúng một mục trong kết quả vừa đọc, "
                          f"{p}={v} ({label}); gọi {c['name']} với {p}={v}")
                assert note == expect, (note, expect)
                stats["b: note equal to independent formatter"] += 1
            if not replay(env, calls[k:], res[k:]):
                stats["b: gold continuation differs"] += 1
                continue
            steps = gold_steps(sample, 0, k) + [(ASK, ask, obs, False)] + gold_steps(sample, k)
            return {"type": "ask_resolve", "src": "clean", "step": k, "tool": c["name"], "param": p, "gold_value": v,
                    "tr": tr, "steps": steps}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="train_rest", choices=["train_rest", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-config", default="181_asr_folds",
                    help="fold-ASR cache (lm51-asr) for train_rest mis-hearings; none to disable")
    ap.add_argument("--asr-cache", type=Path, default=Path(__file__).with_name("asr181_train_rest.json"),
                    help="reviewed train_rest fold transcript export; required unless --asr-config none")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    rcfg = config.load(REPO / RENDER_CFG)
    agent = build("agent", rcfg["agent"])
    guards = config.load(REPO / GUARD_CFG)["guards"]
    assert guards.get("empty_retry") and guards.get("ask_resolve")
    specs = data.tool_specs(SPLIT)
    pool = splits.select(SPLIT, a.subset, a.limit)
    trs = run_asr(SPLIT, pool, rcfg, log=lambda m: print(m, file=sys.stderr))
    heard = {}
    if a.asr_config != "none":
        heard = json.loads(a.asr_cache.read_text(encoding="utf-8"))
    stats = Counter()
    out = Path(a.out)
    rows = []
    for s in pool:
        tr = trs[s["id"]]
        if B.guard_warnings(rcfg, s, tr):  # same filter as build_sft.py
            stats["skip: gold warned under config 10"] += 1
            continue
        if not s["label"]["tool_calls"]:
            continue
        rng = random.Random(f"lm92:{s['id']}")
        for t in (trace_empty_retry(s, tr, heard.get(s["id"]), guards, specs, rng, stats),
                  trace_ask_resolve(s, tr, guards, specs, stats)):
            if t is None:
                continue
            msgs = render(agent, s, t["tr"], t["steps"])
            # self-check: the same renderer with only gold steps reproduces build_sft.conversation exactly
            assert render(agent, s, tr, gold_steps(s)) == B.conversation(agent, s, tr, None), s["id"]
            bad = [m for m in msgs if m.get("train") is False]
            assert len(bad) == 1
            obs_msg = msgs[msgs.index(bad[0]) + 1]["content"]
            note = next(st[2]["note"] for st in t["steps"] if not st[3])
            if note not in obs_msg:
                stats[f"skip: note truncated by obs_chars ({t['type']})"] += 1
                continue
            stats[f"rows {t['type']}"] += 1
            stats[f"rows {t['type']} src={t['src']}"] += 1
            if t["type"] == "empty_retry":
                stats[f"rows empty_retry gold_had_filter={t['with_filter_in_gold']}"] += 1
            stats[f"tool {t['tool']}"] += 1
            rows.append({"id": f"guard_{t['type']}_{s['id']}", "renamed": False, "asr": t["src"] == "asr",
                         "guard": t["type"], "source_id": s["id"],
                         "meta": {k: t[k] for k in ("src", "step", "tool", "param", "gold_value") if k in t}
                         | ({"bad_value": t["bad_value"]} if "bad_value" in t else {}),
                         "messages": msgs})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    stats["rows"] = len(rows)
    stats["samples in pool"] = len(pool)
    meta = {"out": str(out), "sha256": sha, "subset": a.subset, "render_config": RENDER_CFG, "guard_config": GUARD_CFG,
            "guards": guards, "stats": dict(sorted(stats.items()))}
    out.with_suffix(".stats.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    Path(str(out) + ".sha256").write_text(f"{sha}  {out.name}\n", encoding="utf-8")
    print(json.dumps(meta["stats"], ensure_ascii=False), "->", out, sha)


if __name__ == "__main__":
    main()
