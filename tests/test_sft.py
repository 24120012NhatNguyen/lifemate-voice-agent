"""Dữ liệu SFT phải đúng từng chữ những gì ReactJSON gửi lúc chạy, và bản đổi tên phải nhất quán."""
import importlib.util
import json
import random
from types import SimpleNamespace

import pytest

from lifemate import config, data, splits
from lifemate.config import REPO
from lifemate.guards import GuardedEnv
from lifemate.runner import gold_hash
from lifemate.registry import build

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("build_sft", REPO / "scripts" / "build_sft.py")
build_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft)

CFG = config.load(REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
POOL = splits.select("train", "train_rest", limit=60)


def gold_transcript(s):
    users = [t for t in s.get("history", []) if t.get("role") == "user"]
    return {"final": s["audio_text"], "turns": {f"u{k + 1}": u["audio_text"] for k, u in enumerate(users)}, "alts": {}}


class ReplayLLM:
    """LLM giả trả lần lượt các hành động gold và ghi lại mọi tin nhắn agent gửi."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, msgs, **_):
        self.seen.append([dict(m) for m in msgs])
        return SimpleNamespace(content=self.replies.pop(0)), {}


def test_sft_messages_match_what_react_json_sends():
    agent = build("agent", CFG["agent"])
    for s in POOL:
        tr = gold_transcript(s)
        conv = build_sft.conversation(agent, s, tr, None)
        targets = [m["content"] for m in conv if m["role"] == "assistant"]
        agent.llm = ReplayLLM(targets)
        env = GuardedEnv(data.tool_env("train", s["id"]), "train", s, tr["final"], **CFG["guards"])
        agent.run("train", s, tr, env)
        # lần gọi LLM thứ k thấy đúng phần hội thoại đứng trước lượt assistant thứ k
        cuts = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        assert len(agent.llm.seen) == len(cuts), s["id"]
        for seen, i in zip(agent.llm.seen, cuts):
            assert seen == conv[:i], s["id"]


def test_rename_is_consistent_across_schema_and_targets():
    agent = build("agent", CFG["agent"])
    specs = data.tool_specs("train")
    rng = random.Random(0)
    for s in POOL:
        r = build_sft.Renamer(specs, rng)
        conv = build_sft.conversation(agent, s, gold_transcript(s), r)
        system = conv[0]["content"]
        new_specs = {r.tool[n]: r.spec(n, t) for n, t in specs.items()}
        for m in conv:
            if m["role"] != "assistant":
                continue
            act = json.loads(m["content"])
            if act["action"] == "stop":
                continue
            name = act["name"]
            assert f"\n{name} [" in system or name == "sys.ask_missing_info", (s["id"], name)
            props = new_specs[name]["parameters"].get("properties", {})
            assert set(act["arguments"]) <= set(props), (s["id"], name, act["arguments"])
        schema = system[system.index("## Tool đã cài"):system.index("## Định dạng")]
        for app in s["context"]["installed_apps"]:
            if app != "sys":
                assert f"\n{app}." not in schema, (s["id"], app)


def test_calc_steps_replay_exactly_and_stay_out_of_the_submission():
    cfg = config.load(REPO / "configs/experiments/70_qwen3-4b-sft-calc_gold_react.yaml")
    agent = build("agent", cfg["agent"])
    specs = data.tool_specs("train")
    with_calc = 0
    for s in splits.select("train", "train_rest", limit=400):
        tr = gold_transcript(s)
        calcs = build_sft.calc_steps(s, tr["final"], specs)
        if not calcs:
            continue
        with_calc += 1
        conv = build_sft.conversation(agent, s, tr, None, calcs)
        agent.llm = ReplayLLM([m["content"] for m in conv if m["role"] == "assistant"])
        env = GuardedEnv(data.tool_env("train", s["id"]), "train", s, tr["final"], **cfg["guards"])
        agent.run("train", s, tr, env)
        cuts = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        assert [seen for seen in agent.llm.seen] == [conv[:i] for i in cuts], s["id"]  # kết quả sys.calc trùng dữ liệu
        assert env.env.get_tool_calls() == s["label"]["tool_calls"], s["id"]  # sys.calc không vào file nộp
        assert env.env.get_state_hash() == gold_hash("train", s), s["id"]
    assert with_calc >= 15
