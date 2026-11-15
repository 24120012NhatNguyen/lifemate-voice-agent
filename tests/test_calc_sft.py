"""LM-98: opt-in augmentation and inference replay, CPU only."""
import copy
import importlib.util
from types import SimpleNamespace

import pytest

from lifemate import config, data, splits
from lifemate.agent.agents import CALC_LINE
from lifemate.config import REPO
from lifemate.guards import GuardedEnv
from lifemate.registry import build

spec = importlib.util.spec_from_file_location("build_calc_sft", REPO / "scripts/build_sft.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


def agent(enabled):
    cfg = config.load(REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
    cfg["agent"]["calc"] = enabled
    return build("agent", cfg["agent"])


def test_data_config_changes_only_calc():
    base = config.load(REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
    calc = config.load(REPO / "configs/experiments/lm98_calc_data.yaml")
    base.pop("name")
    calc.pop("name")
    assert calc["agent"].pop("calc") is True
    assert calc == base
    assert not build("agent", base["agent"]).calc


def test_disabled_transform_preserves_bytes_without_accessing_data(tmp_path, monkeypatch):
    source, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    blob = b' {"id":"opaque", "messages": []}  \n\n'
    source.write_bytes(blob)
    def forbidden(*_):
        raise AssertionError("disabled mode must not read competition data")
    monkeypatch.setattr(data, "by_id", forbidden)
    monkeypatch.setattr(data, "tool_specs", forbidden)
    sft.augment_sft(source, out, agent(False))
    assert out.read_bytes() == blob


def test_augmented_gold_replays_and_disabled_trace_is_identical():
    cfg = config.load(REPO / "configs/experiments/10_qwen3-4b_gold_react.yaml")
    off, on = agent(False), agent(True)
    specs = data.tool_specs("train")
    count = 0
    for sample in splits.select("train", "train_rest", limit=400):
        tr = {"final": sample["audio_text"], "turns": {f"u{i+1}": t["audio_text"] for i, t in
              enumerate(t for t in sample.get("history", []) if t.get("role") == "user")}}
        if not sft.calc_steps(sample, tr["final"], specs):
            continue
        row = {"id": sample["id"], "asr": False, "renamed": False,
               "messages": sft.conversation(off, sample, tr, None)}
        original = copy.deepcopy(row)
        unchanged, events = sft.calc_from_row(row, off, {}, {})
        assert unchanged == original and events == []
        changed, events = sft.calc_from_row(row, on, {sample["id"]: sample}, specs)
        assert row == original and events
        for runner, conv in [(off, unchanged["messages"]), (on, changed["messages"])]:
            replies = iter(m["content"] for m in conv if m["role"] == "assistant")
            seen = []
            def chat(msgs, **_):
                seen.append(copy.deepcopy(msgs))
                return SimpleNamespace(content=next(replies)), {}
            runner.llm = SimpleNamespace(chat=chat)
            env = GuardedEnv(data.tool_env("train", sample["id"]), "train", sample, tr["final"], **cfg["guards"])
            runner.run("train", sample, tr, env)
            assert seen == [conv[:i] for i, m in enumerate(conv) if m["role"] == "assistant"]
            assert env.env.get_tool_calls() == sample["label"]["tool_calls"]
        count += 1
        if count == 8:
            break
    assert count == 8


def test_fold_final_uses_stored_top1_and_preserves_nbest():
    sample = data.by_id("train")["train_0028"]
    off, on = agent(False), agent(True)
    tr = {"final": "Đặt báo thức trước họp một tiếng", "alts": {"final": ["đặt báo thức trước họp hai tiếng"]}}
    row = {"id": sample["id"] + "_valasr", "asr": True, "renamed": False,
           "messages": sft.conversation(off, sample, tr, None)}
    changed, events = sft.calc_from_row(row, on, {sample["id"]: sample}, data.tool_specs("train"))
    assert changed["messages"][1] == row["messages"][1]
    assert events and all(e["expression"].endswith(" - 60m") for e in events)
    assert CALC_LINE in changed["messages"][0]["content"]
    with pytest.raises(ValueError, match="already augmented"):
        sft.calc_from_row(changed, on, {}, {})


@pytest.mark.parametrize("provenance", [{"teacher": "paraphrase"}, {"synth": "write"}])
def test_other_mixture_rows_only_gain_schema(provenance):
    sample = data.by_id("train")["train_0028"]
    row = {"id": "non_gold", **provenance,
           "messages": sft.conversation(agent(False), sample, {"final": sample["audio_text"]}, None)}
    updated, events = sft.calc_from_row(row, agent(True), {}, {})
    assert events == []
    assert updated["messages"][1:] == row["messages"][1:]
    assert updated["messages"][0]["content"].replace(CALC_LINE, "", 1) == row["messages"][0]["content"]
