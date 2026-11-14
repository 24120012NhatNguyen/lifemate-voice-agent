"""App tổng hợp: gold phải chạy lại ra đúng kết quả và state của chính nó, và hội thoại SFT phải đúng từng chữ ReactJSON."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from lifemate import config, data, splits, synth
from lifemate.agent.agents import ReactJSON
from lifemate.config import REPO
from lifemate.guards import GuardedEnv
from lifemate.registry import build
from lifemate.synth import apps
from lifemate.synth.domains import DOMAINS

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("build_sft", REPO / "scripts" / "build_sft.py")
build_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft)

CFG = config.load(REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
HOSTS = splits.select("train", "train_rest")
SAMPLES = synth.generate(HOSTS, 300, seed=0)


class ReplayLLM:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, msgs, **_):
        self.seen.append([dict(m) for m in msgs])
        return SimpleNamespace(content=self.replies.pop(0)), {}


def test_gold_replays_to_its_own_results_and_state():
    for s in SAMPLES:
        assert synth.replay(s, CFG["guards"]) == [], s["id"]


def test_gold_only_calls_installed_tools_and_asks_with_schema_names():
    for s in SAMPLES:
        installed = set(data.installed_tools(s))
        specs = synth.specs(s)
        for c in s["label"]["tool_calls"]:
            assert c["name"] in installed, (s["id"], c["name"])
            if c["name"] == "sys.ask_missing_info":
                target = c["arguments"]["tool_name"]
                assert target in installed, (s["id"], target)
                props = specs[target]["parameters"]["properties"]
                assert set(c["arguments"]["params"]) <= set(specs[target]["parameters"]["required"]) <= set(props), s["id"]


def test_sft_messages_match_what_react_json_sends():
    agent = build("agent", CFG["agent"])
    for s in SAMPLES[:120]:
        tr = synth.transcript(s)
        sp = synth.specs(s)
        conv = build_sft.conversation(agent, s, tr, None, specs=sp)
        agent.llm = ReplayLLM([m["content"] for m in conv if m["role"] == "assistant"])
        # lúc chạy, catalog đến từ data.tool_specs; mẫu tổng hợp mang theo catalog của nó
        agent.system_prompt = lambda split, sample, specs=None, rules=None: ReactJSON.system_prompt(
            agent, split, sample, specs=sp, rules=rules)
        env = apps.SynthEnv.of(s)
        agent.run("train", s, tr, GuardedEnv(env, "train", s, tr["final"], **CFG["guards"]))
        cuts = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        assert agent.llm.seen == [conv[:i] for i in cuts], s["id"]
        assert env.get_tool_calls() == s["label"]["tool_calls"], s["id"]
        assert env.get_state_hash() == s["synth"]["state_hash"], s["id"]
        system = conv[0]["content"]
        for inst in s["synth"]["apps"]:
            for t in inst["catalog"]["tools"]:
                assert f"\n{t['name']} [" in system, (s["id"], t["name"])


def test_covers_the_behaviours_the_gold_needs():
    seen = Counter()
    for s in SAMPLES:
        calls = s["label"]["tool_calls"]
        kinds = [synth.specs(s).get(c["name"], {}).get("kind", "meta") for c in calls]
        seen["rỗng"] += not calls
        seen["hỏi lại"] += "meta" in kinds
        seen["đọc rồi ghi"] += any(a == "read" and b == "write" for a, b in zip(kinds, kinds[1:]))
        seen["ghi không đọc, ID từ lịch sử"] += bool(s["history"]) and bool(kinds) and kinds[0] == "write"
        seen["chỉ đọc"] += bool(kinds) and set(kinds) == {"read"}
        seen["app thật"] += any(c["name"] in data.tool_specs("train") for c in calls if c["name"] != "sys.ask_missing_info")
        seen["nhiều app tổng hợp"] += len(s["synth"]["apps"]) > 1
        seen["hai lệnh ghi"] += kinds.count("write") >= 2
        if s["synth"]["kind"] == "pick_detail":  # hai mục cùng tên gốc, lời nói đủ chi tiết: đọc rồi ghi, không hỏi
            seen["chọn theo chi tiết"] += "meta" not in kinds and "write" in kinds and "read" in kinds
    for k in ["rỗng", "hỏi lại", "đọc rồi ghi", "ghi không đọc, ID từ lịch sử", "chỉ đọc", "app thật", "nhiều app tổng hợp",
              "hai lệnh ghi", "chọn theo chi tiết"]:
        assert seen[k] >= 5, (k, dict(seen))


def test_pick_detail_reads_a_list_with_two_lookalikes_and_acts_without_asking():
    picked = [s for s in SAMPLES if s["synth"]["kind"] == "pick_detail"]
    assert picked
    for s in picked:
        calls = s["label"]["tool_calls"]
        assert all(c["name"] != "sys.ask_missing_info" for c in calls), s["id"]
        kinds = [synth.specs(s)[c["name"]]["kind"] for c in calls]
        assert "read" in kinds and "write" in kinds, (s["id"], kinds)
        # lời nói mang đuôi phân biệt của make_ambiguous ("lớn" / "nhỏ", "cũ" / "mới", "1" / "2", "ở nhà" / "ở công ty")
        low = s["audio_text"].lower()
        assert any(a in low or b in low for a, b in synth.scenarios.AMBIG), s["id"]


def test_synthetic_apps_never_reuse_a_real_or_held_out_app_name():
    real = synth.scenarios.real_app_ids() | set(splits.HELDOUT_APPS)
    for d in DOMAINS:
        assert not set(d.ids) & real, d.ids
    for s in SAMPLES:
        for inst in s["synth"]["apps"]:
            assert inst["app"] not in real
            assert inst["app"] in s["context"]["installed_apps"]


def test_tool_and_param_names_vary_between_instances():
    names = Counter()
    for s in SAMPLES:
        for inst in s["synth"]["apps"]:
            names[inst["domain"]] += 0
            for t in inst["catalog"]["tools"]:
                names[(inst["domain"], t["name"].split(".")[1], json.dumps(sorted(t["parameters"]["properties"])))] += 1
    for d in {k for k in names if isinstance(k, str)}:
        assert len([k for k in names if isinstance(k, tuple) and k[0] == d]) > len(
            next(x for x in DOMAINS if x.ids[0] == d).tools), d


def test_generation_is_deterministic():
    assert synth.generate(HOSTS, 30, seed=7) == synth.generate(HOSTS, 30, seed=7)


def digest(samples) -> str:
    return hashlib.sha256(json.dumps(samples, ensure_ascii=False).encode()).hexdigest()


def test_samples_do_not_depend_on_the_hash_seed_of_the_process():
    """Trước LM-48 bộ sinh duyệt thẳng `set` các tên trường, nên cùng seed mà mỗi tiến trình Python (PYTHONHASHSEED) ra
    một bộ mẫu khác, kể cả thứ tự khoá trong call gold. Giờ mọi tập đi qua một hàm xếp thứ tự."""
    code = ("import hashlib, json; from lifemate import splits, synth; "
            "ss = synth.generate(splits.select('train', 'train_rest'), 60, seed=0); "
            "print(hashlib.sha256(json.dumps(ss, ensure_ascii=False).encode()).hexdigest())")
    run = lambda seed: subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed},
                                      capture_output=True, text=True, check=True).stdout.strip()
    assert run("1") == run("2") == digest(SAMPLES[:60])


def test_the_two_set_walks_are_the_only_place_the_order_function_matters():
    by_abc = synth.generate(HOSTS, 40, seed=3)
    assert digest(by_abc) == digest(synth.generate(HOSTS, 40, seed=3, orders=None))
    backwards = []  # ghi lại một thứ tự khác (bảng chữ cái ngược), rồi phát lại: phải ra đúng bộ mẫu của thứ tự đó
    order = lambda xs: backwards.append(sorted(xs, reverse=True)) or backwards[-1]
    rng, avoid, out = synth.scenarios.random.Random(3), synth.scenarios.real_app_ids(), []
    while len(out) < 40:
        sample = synth.scenarios.attempt(rng, HOSTS, synth.scenarios.KINDS, avoid, order, len(out) + 1)
        if sample is not None:
            out.append(sample)
    recorded = [o for o in backwards if len(o) > 1]
    assert digest(out) != digest(by_abc) and recorded
    assert digest(synth.generate(HOSTS, 40, seed=3, orders=recorded)) == digest(out)


def test_replay_refuses_orders_recorded_for_another_run():
    replay = synth.scenarios.Replay([["b", "a"]])
    assert replay(set()) == [] and replay({"z"}) == ["z"] and replay({"a", "b"}) == ["b", "a"]
    with pytest.raises(ValueError):
        replay({"a", "b"})  # hết thứ tự đã ghi
    with pytest.raises(ValueError):
        synth.generate(HOSTS, 30, seed=0, orders=[["không", "phải"]] * 200)  # thứ tự của một tập khác


@pytest.mark.parametrize("name, subset, kinds", [("synth_v1_orders.json", "train_rest", "v1"),
                                                 ("synth_full_v2_orders.json", "all", "v2")])
def test_the_training_files_are_reproduced_from_their_recorded_set_orders(name, subset, kinds):
    """600 mẫu tổng hợp của hai file đã dùng để train sinh lại được từng byte từ mã, seed 2026, tập train và thứ tự duyệt
    dò lại bằng scripts/synth_orders.py: sft/train_synth.jsonl (sha256 cả file b251018b…, bảng kịch bản v1 trước khi có
    pick_detail, mẫu chủ train_rest) và phần tổng hợp của sft/train_full_para_v1.jsonl (d8363b21…, bảng v2, mẫu chủ all)."""
    rep = json.loads((REPO / "docs/repro" / name).read_text(encoding="utf-8"))
    assert (rep["n"], rep["seed"], rep["kinds"], rep["subset"]) == (600, 2026, kinds, subset)
    table = {"v1": synth.scenarios.KINDS_V1, "v2": synth.scenarios.KINDS}[kinds]
    samples = synth.generate(splits.select("train", subset), rep["n"], seed=rep["seed"], kinds=table, orders=rep["orders"])
    agent = build("agent", config.load(REPO / rep["config"])["agent"])
    lines = [json.dumps({"id": s["id"], "synth": s["synth"]["kind"],
                         "messages": build_sft.conversation(agent, s, synth.transcript(s), None, specs=synth.specs(s))},
                        ensure_ascii=False) for s in samples]
    assert hashlib.sha256("".join(x + chr(10) for x in lines).encode()).hexdigest() == rep["synth_lines_sha256"]
    assert any(s["synth"]["kind"] == "pick_detail" for s in samples) == (kinds == "v2")
    assert "pick_detail" not in synth.scenarios.KINDS_V1


def test_generation_reads_nothing_but_the_train_split(monkeypatch):
    """Tập public không được dùng để sinh dữ liệu dưới bất kỳ hình thức nào (FAQ của BTC), kể cả để tránh trùng tên app."""
    for name in ("catalog", "tool_specs", "samples", "by_id", "fuzzy_params", "zaic_tools", "tool_env", "tool_kind"):
        real = getattr(data, name)

        def guarded(split, *args, _real=real, _name=name, **kw):
            assert split == "train", f"bộ sinh gọi data.{_name}({split!r})"
            return _real(split, *args, **kw)

        monkeypatch.setattr(data, name, guarded)
    samples = synth.generate(HOSTS, 40, seed=5)
    assert len(samples) == 40 and all(not synth.replay(s, CFG["guards"]) for s in samples)
    assert synth.scenarios.real_app_ids() == {a["id"] for a in data.catalog("train")["apps"]}


def test_build_sft_reads_nothing_but_the_train_split(monkeypatch, tmp_path):
    """Cả đường dựng file SFT (mẫu thật + mẫu tổng hợp): không hàm đọc dữ liệu nào được gọi với split khác train."""
    for name in ("catalog", "tool_specs", "samples", "by_id", "fuzzy_params", "zaic_tools", "tool_env", "tool_kind"):
        real = getattr(data, name)

        def guarded(split, *args, _real=real, _name=name, **kw):
            assert split == "train", f"build_sft gọi data.{_name}({split!r})"
            return _real(split, *args, **kw)

        monkeypatch.setattr(data, name, guarded)
    out = tmp_path / "train.jsonl"
    build_sft.main(["--limit", "25", "--synth", "10", "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sum(r["id"].startswith("synth_") for r in rows) == 10 and 20 <= len(rows) <= 35
    assert not any("public" in r["id"] for r in rows)
