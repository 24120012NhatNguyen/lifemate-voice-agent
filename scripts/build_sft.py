"""Dựng dữ liệu SFT (proposal nấc 3) cho agent `react_json` từ quỹ đạo gold của `train_rest`.

Mỗi mẫu thành một hội thoại nhiều lượt, đúng các tin nhắn ReactJSON gửi lúc chạy: system và user dựng bằng chính
agent của config, mỗi call gold là một lượt assistant {"thought", "action", "name", "arguments"}, theo sau là lượt
user "KẾT QUẢ ..." lấy từ `label.tool_results`, và kết thúc bằng một hành động stop. Mẫu gold rỗng chỉ có một stop.

--rename P: mỗi mẫu có xác suất P được đổi tên nhất quán app, tool và tham số trong schema, luật, lịch sử và đáp án
(proposal 5.5), để mô hình đọc schema thay vì nhớ tên.

--asr-config C: mỗi mẫu có xác suất --asr-frac dùng bản chép ASR của config C (cả lượt cuối và lịch sử) thay cho chữ
đúng, còn đáp án vẫn là call gold (proposal nấc 3), để agent học làm đúng khi nghe sai thay vì đứng yên.
Config có `agent.calc: true`: trước mỗi call ghi có giá trị không nói thẳng ra mà tính được bằng một phép tính từ số đã có
(lời nói, kết quả tool, lịch sử), chèn một bước `sys.calc` với biểu thức đó và kết quả của nó (src/lifemate/numbers.py).

--asr-keep-entities chỉ dùng bản chép đó khi mọi giá trị chuỗi người dùng nói ra trong call gold vẫn còn nguyên văn
trong nó: nếu không, đáp án đòi một tên mà bản chép đã mất và mô hình học đoán (docs/BASELINE.md mục 8).

--synth N thêm N mẫu dùng app tổng hợp kiểu CRUD (roadmap đề xuất 5, src/lifemate/synth/): mỗi mẫu mượn ngữ cảnh một
mẫu của tập đang dựng, cài thêm 1 đến 3 app tổng hợp với tên tool và tham số ngẫu nhiên, và yêu cầu cuối nhắm vào
chúng; gold chạy thật trên môi trường của app tổng hợp nên kết quả tool và state_hash là thật.

--synth-replay FILE sinh lại đúng các mẫu tổng hợp của một lượt cũ (docs/repro/synth_v1_orders.json cho 600 mẫu của
sft/train_synth.jsonl): bộ sinh thời đó duyệt thẳng `set`, nên cùng seed mà mỗi tiến trình ra một bộ mẫu khác; file
này ghi lại thứ tự duyệt của lượt đó, do scripts/synth_orders.py dò ra. Số mẫu, seed và bảng kịch bản lấy từ FILE.

    .venv/bin/python scripts/build_sft.py
    .venv/bin/python scripts/build_sft.py --rename 0.5 --out sft/train_rename.jsonl
    .venv/bin/python scripts/build_sft.py --drop-heldout --out sft/train_drop.jsonl
    .venv/bin/python scripts/build_sft.py --subset all --out sft/train_full.jsonl   (bản nộp: không còn tập đo sạch)
    .venv/bin/python scripts/build_sft.py --asr-config configs/experiments/54_qwen3-4b-sft_phowhisper-large-ft_react.yaml \
        --out sft/train_asrmix.jsonl
    .venv/bin/python scripts/build_sft.py --synth 600 --out sft/train_synth.jsonl
    .venv/bin/python scripts/build_sft.py --synth-replay docs/repro/synth_v1_orders.json --out sft/train_synth.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter

from lifemate import config, context, data, splits, synth
from lifemate.agent.agents import CALC, ReactJSON
from lifemate.asr.base import run_asr
from lifemate.asr.metrics import entity_spans, norm
from lifemate.config import REPO
from lifemate.guards import GuardedEnv
from lifemate.numbers import calc, derive
from lifemate.registry import build

SPLIT = "train"
SUBSET = "train_rest"  # mặc định không dùng val_random hay val_app: đó là tập đo; --subset all chỉ cho bản nộp
HELDOUT = set(splits.HELDOUT_APPS)

# Từ thay thế theo từng mảnh tên (tách bởi "_"): giữ nghĩa, chỉ đổi chữ. Mảnh "id" không đổi để giữ quy ước *_id.
TOOL_SYN = {
    "list": ["browse", "show", "fetch_all"], "get": ["fetch", "read", "view", "show"], "set": ["adjust", "control", "change"],
    "create": ["add", "new", "make"], "search": ["find", "lookup", "query"], "cancel": ["abort", "revoke", "void"],
    "add": ["create", "insert", "new"], "send": ["post", "dispatch", "deliver"], "update": ["edit", "modify", "change"],
    "delete": ["remove", "erase", "drop"], "remove": ["delete", "erase", "drop"], "read": ["view", "open", "fetch"],
    "check": ["verify", "inspect"], "place": ["submit", "make"], "order": ["purchase", "request"], "log": ["record", "track"],
    "status": ["state", "info"], "start": ["begin", "launch"], "stop": ["halt", "end"], "complete": ["finish", "close"],
    "book": ["reserve", "order"], "reserve": ["book", "hold"], "pay": ["settle", "charge"], "save": ["store", "keep"],
    "play": ["stream"], "recent": ["latest", "last"], "schedule": ["plan", "arrange"], "transfer": ["send", "move"],
    "convert": ["exchange", "calc"], "run": ["execute", "trigger"], "control": ["operate", "manage"],
}
PARAM_SYN = {
    "date": ["day"], "time": ["at", "when", "datetime"], "type": ["kind", "category"], "from": ["source", "src"],
    "to": ["target", "dest"], "title": ["heading", "subject", "name"], "amount": ["value", "sum", "money"],
    "content": ["text", "body", "message"], "query": ["keyword", "q", "term"], "phone": ["phone_number", "tel", "mobile"],
    "mode": ["setting", "option"], "label": ["tag", "name"], "limit": ["max", "top", "count"], "note": ["memo", "comment"],
    "number": ["no", "num"], "body": ["text", "content"], "name": ["title", "label"], "location": ["place", "area", "where"],
    "start": ["begin", "from"], "end": ["finish", "until"], "minutes": ["mins", "duration"], "address": ["addr", "place"],
    "enabled": ["active", "on"], "repeat": ["recurrence", "recur"], "priority": ["importance", "level"],
    "quantity": ["qty", "count"], "destination": ["dest", "dropoff"], "origin": ["source", "pickup"],
    "passengers": ["pax", "people"], "attendees": ["participants", "guests"], "volume": ["level", "loudness"],
    "power": ["state", "switch"], "currency": ["ccy"], "email": ["mail", "email_address"], "subject": ["topic", "title"],
}
SYLLABLES = ["ba", "bo", "ca", "da", "di", "do", "ga", "ha", "ki", "ko", "la", "li", "lu", "ma", "mi", "mo", "na", "ni",
             "no", "pa", "pi", "ra", "ri", "sa", "so", "ta", "ti", "to", "va", "vi", "vo", "xu", "za", "zo"]
KEEP_APPS = {"sys"}  # sys.ask_missing_info giữ nguyên: lúc thi luôn có đúng tên này


def _respell(name: str, table: dict, p: float, rng: random.Random) -> str:
    return "_".join(rng.choice(table[t]) if t in table and rng.random() < p else t for t in name.split("_"))


def _unique(make, taken: set, fallback: str, tries=8) -> str:
    for _ in range(tries):
        cand = make()
        if cand not in taken:
            return cand
    return fallback


class Renamer:
    """Một bộ tên mới cho cả catalog: app thành tên kiểu thương hiệu, tool và tham số thay từ đồng nghĩa."""

    def __init__(self, specs: dict, rng: random.Random, p_tool=0.7, p_param=0.5):
        apps = sorted({n.split(".", 1)[0] for n in specs})
        taken = set(apps)
        self.app = {}
        for a in apps:
            if a in KEEP_APPS:
                self.app[a] = a
                continue
            brand = _unique(lambda: "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))), taken, a + "_x")
            taken.add(brand)
            self.app[a] = brand
        self.tool, self.param = {}, {}
        for n in sorted(specs):
            app, short = n.split(".", 1)
            if app in KEEP_APPS:
                self.tool[n], self.param[n] = n, {}
                continue
            used = {self.tool[m].split(".", 1)[1] for m in self.tool if m.startswith(app + ".")}
            new_short = _unique(lambda: _respell(short, TOOL_SYN, p_tool, rng), used, short)
            self.tool[n] = f"{self.app[app]}.{new_short}"
            props = list(specs[n].get("parameters", {}).get("properties", {}))
            pm: dict[str, str] = {}
            for p in props:
                taken_p = set(pm.values()) | (set(props) - {p})
                pm[p] = _unique(lambda: _respell(p, PARAM_SYN, p_param, rng), taken_p, p)
            self.param[n] = pm

    # -- văn bản: thay tên tool đầy đủ và tiền tố app (x-ref "app.collection") -------------------------------
    def text(self, s: str) -> str:
        def sub(m):
            full = m.group(0)
            if full in self.tool:
                return self.tool[full]
            app, rest = full.split(".", 1)
            return f"{self.app[app]}.{rest}" if app in self.app else full
        return re.sub(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b", sub, s)

    def spec(self, n: str, spec: dict) -> dict:
        out = copy.deepcopy(spec)
        pm = self.param[n]
        changed = {p: q for p, q in pm.items() if p != q}
        word = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, changed)) + r")(?!\w)") if changed else None

        def desc(s: str) -> str:  # mô tả nhắc tên tham số ("bắt buộc: amount") cũng phải đổi theo
            s = self.text(s)
            return word.sub(lambda m: changed[m.group(1)], s) if word else s

        out["name"] = self.tool[n]
        if "app" in out:
            out["app"] = self.app.get(out["app"], out["app"])
        out["description"] = desc(out.get("description", ""))
        params = out.get("parameters") or {}
        props = params.get("properties", {})
        new_props = {}
        for p, s in props.items():
            s = dict(s)
            if "description" in s:
                s["description"] = desc(s["description"])
            if "x-ref" in s:
                s["x-ref"] = self.text(s["x-ref"])
            new_props[pm.get(p, p)] = s
        if props:
            params["properties"] = new_props
        if "required" in params:
            params["required"] = [pm.get(p, p) for p in params["required"]]
        if "x-one-of" in params:
            params["x-one-of"] = [[pm.get(p, p) for p in group] for group in params["x-one-of"]]
        return out

    def call(self, c: dict) -> dict:
        name, args = c["name"], dict(c.get("arguments") or {})
        pm = self.param.get(name, {})
        args = {pm.get(k, k): v for k, v in args.items()}
        if name == "sys.ask_missing_info":
            target = args.get("tool_name")
            tpm = self.param.get(target, {})
            if target in self.tool:
                args["tool_name"] = self.tool[target]
            if isinstance(args.get("params"), list):
                args["params"] = [tpm.get(p, p) for p in args["params"]]
        return {**c, "name": self.tool.get(name, name), "arguments": args}

    def sample(self, s: dict) -> dict:
        s = copy.deepcopy(s)
        apps = s["context"]["installed_apps"]
        s["context"]["installed_apps"] = {
            self.app.get(a, a): {**v, "tools": [self.tool.get(t, t) for t in v["tools"]]} for a, v in apps.items()}
        for turn in s.get("history", []):
            if turn.get("tool_calls"):
                turn["tool_calls"] = [self.call(c) for c in turn["tool_calls"]]
        s["label"]["tool_calls"] = [self.call(c) for c in s["label"]["tool_calls"]]
        return s


def action(name: str | None = None, arguments: dict | None = None) -> str:
    """Đáp án một bước, cùng thứ tự khoá với prompt và JSON Schema của ReactJSON. Gold không có suy nghĩ nên để rỗng."""
    act = {"thought": "", "action": "stop"} if name is None else {"thought": "", "action": "call", "name": name, "arguments": arguments}
    return json.dumps(act, ensure_ascii=False)


def guard_warnings(cfg: dict, sample: dict, transcript: dict) -> list[str]:
    """Phát lại gold qua middleware của config: gold có call bị cảnh báo (ID tự đặt) thì không dạy cho mô hình."""
    env = GuardedEnv(data.tool_env(SPLIT, sample["id"]), SPLIT, sample, transcript["final"], **(cfg.get("guards") or {}))
    for c in sample["label"]["tool_calls"]:
        env.call(c["name"], c["arguments"])
    return [w for g in env.log for w in g.get("warnings", [])]


def heldout_in_history(sample: dict) -> bool:
    return any(c["name"].split(".")[0] in HELDOUT for t in sample.get("history", []) for c in t.get("tool_calls") or [])


def without_heldout(sample: dict) -> dict:
    """Bản sao mẫu đã gỡ app của val_app khỏi installed_apps. Gold của train_rest không bao giờ gọi các app này, nên nếu
    để nguyên, mô hình thấy chúng hàng trăm lần mà chưa lần nào được gọi và học cách bỏ qua chúng (docs/BASELINE.md mục 7)."""
    s = copy.deepcopy(sample)
    s["context"]["installed_apps"] = {a: v for a, v in s["context"]["installed_apps"].items() if a not in HELDOUT}
    return s


def calc_steps(sample: dict, said: str, specs: dict) -> dict[int, list[str]]:
    """Biểu thức sys.calc cần chèn trước từng call gold (theo chỉ số), cho các tham số call ghi tính được từ số đã có."""
    sources = [sample["context"].get("user_profile", {}), sample.get("history", [])]
    dates = context.timetable_dates(sample["context"]["current_time"])
    out = {}
    for i, (c, r) in enumerate(zip(sample["label"]["tool_calls"], sample["label"]["tool_results"])):
        if specs.get(c["name"], {}).get("kind") == "write":
            for v in (c.get("arguments") or {}).values():
                e = derive(v, said, sources, sample["context"]["current_time"], dates)
                if e:
                    assert calc(e) == v, (sample["id"], e, v)
                    out.setdefault(i, []).append(e)
            sources.append(c.get("arguments") or {})
        sources.append(r)
    return out


def conversation(agent: ReactJSON, sample: dict, transcript: dict, renamer: Renamer | None,
                 calcs: dict[int, list[str]] | None = None, specs: dict | None = None) -> list[dict]:
    """specs: catalog thay cho catalog train (mẫu tổng hợp mang theo app của nó)."""
    rules = None
    if renamer:
        orig = data.tool_specs(SPLIT)
        specs = {renamer.tool[n]: renamer.spec(n, t) for n, t in orig.items()}
        rules = renamer.text(agent.rules)
        sample = renamer.sample(sample)
    msgs = [{"role": "system", "content": agent.system_prompt(SPLIT, sample, specs=specs, rules=rules)},
            {"role": "user", "content": agent.user_prompt(sample, transcript)}]
    for i, (c, obs) in enumerate(zip(sample["label"]["tool_calls"], sample["label"]["tool_results"])):
        for e in (calcs or {}).get(i, []):
            msgs.append({"role": "assistant", "content": action(CALC, {"expression": e})})
            msgs.append({"role": "user", "content": f"KẾT QUẢ {CALC}: {agent._obs({'result': calc(e)})}"})
        msgs.append({"role": "assistant", "content": action(c["name"], c["arguments"])})
        msgs.append({"role": "user", "content": f"KẾT QUẢ {c['name']}: {agent._obs(obs)}"})
    msgs.append({"role": "assistant", "content": action()})
    return msgs


def synth_samples(a, hosts: list[dict]) -> list[dict]:
    """Mẫu tổng hợp của lượt dựng: --synth N sinh mới (tất định theo --seed), --synth-replay phát lại một lượt cũ."""
    if a.synth_replay:
        rep = json.loads((REPO / a.synth_replay).read_text(encoding="utf-8"))
        if (a.synth or rep["subset"] != a.subset or a.drop_heldout or a.limit
                or a.config != rep["config"] or a.seed != rep["seed"]):
            sys.exit("--synth-replay: cần đúng subset, config, seed của metadata; không kèm --synth, --drop-heldout, --limit")
        kinds = {"v1": synth.scenarios.KINDS_V1, "v2": synth.scenarios.KINDS}[rep["kinds"]]
        return synth.generate(hosts, rep["n"], seed=rep["seed"], kinds=kinds, orders=rep["orders"])
    return synth.generate(hosts, a.synth, seed=a.seed) if a.synth else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/10_qwen3-4b_gold_react.yaml",
                    help="config của agent sẽ chạy: lấy prompt, luật, ngữ cảnh và ASR từ đây")
    ap.add_argument("--out", default="sft/train.jsonl")
    ap.add_argument("--rename", type=float, default=0.0, help="xác suất một mẫu được đổi tên app, tool, tham số")
    ap.add_argument("--drop-heldout", action="store_true",
                    help="gỡ app của val_app khỏi installed_apps và bỏ mẫu có lịch sử gọi chúng, để val_app đo đúng app chưa thấy")
    ap.add_argument("--asr-config", default=None, help="config có ASR thật: lấy bản chép của nó cho một phần mẫu")
    ap.add_argument("--asr-frac", type=float, default=0.5, help="xác suất một mẫu dùng bản chép của --asr-config")
    ap.add_argument("--asr-keep-entities", action="store_true",
                    help="bản chép ASR làm mất thực thể của call gold thì dùng chữ đúng cho mẫu đó")
    ap.add_argument("--subset", default=SUBSET, choices=[SUBSET, "all"],
                    help="all: cả 1.545 mẫu train, để huấn luyện bản nộp sau khi đã chọn công thức trên train_rest")
    ap.add_argument("--synth", type=int, default=0, help="thêm N mẫu dùng app tổng hợp (src/lifemate/synth/)")
    ap.add_argument("--synth-replay", default=None,
                    help="JSON của scripts/synth_orders.py: sinh lại đúng các mẫu tổng hợp của một lượt cũ (thay cho --synth)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.subset == "all" and a.drop_heldout:
        sys.exit("--drop-heldout chỉ có nghĩa khi còn giữ val_app để đo")

    cfg = config.load(REPO / a.config)
    agent = build("agent", cfg["agent"])
    if not isinstance(agent, ReactJSON):
        sys.exit(f"config {a.config} không dùng agent react_json")
    pool = splits.select(SPLIT, a.subset, a.limit)
    if a.subset == SUBSET:
        held = {s["id"] for sub in ("val_random", "val_app") for s in splits.select(SPLIT, sub)}
        assert not held & {s["id"] for s in pool}, "train_rest trùng tập đo"
    trs = run_asr(SPLIT, pool, cfg, log=lambda m: print(m, file=sys.stderr))
    heard = run_asr(SPLIT, pool, config.load(REPO / a.asr_config), log=lambda m: print(m, file=sys.stderr)) if a.asr_config else None

    rng = random.Random(a.seed)
    specs = data.tool_specs(SPLIT)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with out.open("w", encoding="utf-8") as f:
        for s in pool:
            if a.drop_heldout:
                if heldout_in_history(s):
                    stats["bỏ: lịch sử gọi app giữ lại"] += 1
                    continue
                s = without_heldout(s)
            use_asr = heard is not None and rng.random() < a.asr_frac
            if use_asr and a.asr_keep_entities and any(norm(e) not in norm(heard[s["id"]]["final"]) for e in entity_spans(s)):
                stats["chép ASR mất thực thể, dùng chữ đúng"] += 1
                use_asr = False
            tr = (heard if use_asr else trs)[s["id"]]
            if guard_warnings(cfg, s, tr):
                stats["bỏ: gold bị middleware cảnh báo"] += 1
                continue
            renamer = Renamer(specs, rng) if rng.random() < a.rename else None
            calcs = calc_steps(s, tr["final"], specs) if agent.calc else None
            msgs = conversation(agent, s, tr, renamer, calcs)
            calls = s["label"]["tool_calls"]
            stats["mẫu"] += 1
            stats["đổi tên"] += renamer is not None
            stats["chép ASR"] += use_asr
            stats["gold rỗng"] += not calls
            stats["có hỏi lại"] += any(c["name"] == "sys.ask_missing_info" for c in calls)
            stats["lượt assistant"] += len(calls) + 1 + sum(map(len, (calcs or {}).values()))
            stats["mẫu có sys.calc"] += bool(calcs)
            stats["bước sys.calc"] += sum(map(len, (calcs or {}).values()))
            f.write(json.dumps({"id": s["id"], "renamed": renamer is not None, "asr": use_asr, "messages": msgs},
                               ensure_ascii=False) + "\n")
        hosts = [without_heldout(s) for s in pool if not heldout_in_history(s)] if a.drop_heldout else pool
        for s in synth_samples(a, hosts):
            if synth.replay(s, cfg.get("guards")):
                stats["bỏ: mẫu tổng hợp phát lại lệch"] += 1
                continue
            msgs = conversation(agent, s, synth.transcript(s), None, specs=synth.specs(s))
            calls = s["label"]["tool_calls"]
            stats["mẫu tổng hợp"] += 1
            stats[f"tổng hợp: {s['synth']['kind']}"] += 1
            stats["lượt assistant"] += len(calls) + 1
            f.write(json.dumps({"id": s["id"], "synth": s["synth"]["kind"], "messages": msgs}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats), ensure_ascii=False), "→", out)


if __name__ == "__main__":
    main()
