"""Chỉnh quyết định làm / hỏi lại / đứng yên theo xác suất của model (LM-25): đọc logprob, ép làm, ngưỡng."""
import json
import math
import re
from types import SimpleNamespace

import pytest

from lifemate import data, splits
from lifemate.agent import agents
from lifemate.agent.agents import ASK, decision_probs
from lifemate.agent.llm import LLM

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

SPECS = data.tool_specs("train")
SAMPLE = next(s for s in splits.select("train", "val_random")
              if any(SPECS.get(t, {}).get("kind") == "read" for t in data.installed_tools(s))
              and any(SPECS.get(t, {}).get("kind") == "write" for t in data.installed_tools(s)))
TOOLS = data.installed_tools(SAMPLE)
READ = next(t for t in TOOLS if SPECS[t]["kind"] == "read")
WRITE = next(t for t in TOOLS if SPECS[t]["kind"] == "write")


def stop():
    return json.dumps({"thought": "", "action": "stop"})


def call(name, **args):
    return json.dumps({"thought": "", "action": "call", "name": name, "arguments": args})


def tokens(text, alts):
    """Tách câu như tokenizer Qwen ở hai chỗ quyết định: ' "' rồi từ đầu của giá trị là một token riêng.
    alts: khoá ("action" / "name") → [(token, xác suất), ...] ở token đầu của giá trị."""
    cuts = []
    for key in alts:
        m = re.search(r'"' + key + r'"\s*:\s*"', text)
        if m:
            cuts.append((m.end(), key))
    out, pos = [], 0
    for vs, key in sorted(cuts):
        out.append((text[pos:vs - 2], -0.01, []))
        out.append((text[vs - 2:vs], -0.01, []))
        word = re.match(r"[a-z]+", text[vs:]).group(0)
        out.append((word, math.log(dict(alts[key]).get(word, 0.5)), [(t, math.log(p)) for t, p in alts[key]]))
        pos = vs + len(word)
    out.append((text[pos:], -0.01, []))
    return out


class FakeLLM:
    """replies: (câu, {khoá: [(token, xác suất)]}); ghi lại tham số từng lượt."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, **kw):
        self.requests.append({"messages": list(messages), **kw})
        text, alts = self.replies.pop(0)
        meta = {"sec": 1.0, "prompt_tokens": 1, "completion_tokens": 1}
        if kw.get("logprobs"):
            meta["logprobs"] = tokens(text, alts)
        return SimpleNamespace(content=text), meta


class FakeEnv:
    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append(name)
        return {"ok": True}


def run(replies, **decision):
    a = agents.ReactJSON(decision=decision or None)
    a.llm = FakeLLM(replies)
    env = FakeEnv()
    steps = a.run("train", SAMPLE, {"final": SAMPLE["audio_text"]}, env)["steps"]
    return a.llm.requests, env.calls, steps


def names(req):
    return req["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"]


def actions(req):
    return req["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]


ASKING = (call(ASK, tool_name=WRITE, params=["x"]), {"action": [("call", 0.9), ("stop", 0.1)],
                                                      "name": [("sys", 0.3), (WRITE[:3], 0.6), ("zz", 0.1)]})
STOPPING = (stop(), {"action": [("stop", 0.4), ("call", 0.6)]})
SURE_STOP = (stop(), {"action": [("stop", 0.95), ("call", 0.05)]})
READING = (call(READ), {"action": [("call", 0.9), ("stop", 0.1)], "name": [(READ[:3], 0.9)]})
WRITING = (call(WRITE), {"action": [("call", 0.9), ("stop", 0.1)], "name": [(WRITE[:3], 0.9)]})


def test_probs_normalise_over_the_valid_choices_only():
    p = decision_probs(tokens(*ASKING), ASKING[0], TOOLS)
    assert p["p_stop"] == pytest.approx(0.1) and p["p_call"] == pytest.approx(0.9)
    other = sum(q for t, q in ASKING[1]["name"] if t != "sys" and any(x.startswith(t) for x in TOOLS if x != ASK))
    assert p["p_ask_call"] == pytest.approx(0.3 / (0.3 + other), abs=1e-5)
    assert p["p_ask"] == pytest.approx(0.9 * p["p_ask_call"], abs=1e-5)
    assert p["p_act"] == pytest.approx(p["p_call"] - p["p_ask"], abs=1e-5)


def test_stop_has_no_tool_name_probability():
    p = decision_probs(tokens(*STOPPING), STOPPING[0], TOOLS)
    assert p == {"p_stop": pytest.approx(0.4), "p_call": pytest.approx(0.6)}


def test_alternatives_that_carry_the_quote_are_matched_after_the_shared_prefix():
    text = stop()
    toks = [('{"thought": "", "action":', -0.01, []),
            (' "stop', math.log(0.3), [(' "stop', math.log(0.3)), (' "call', math.log(0.6)), (' "', math.log(0.1))]),
            ('"}', -0.01, [])]
    assert decision_probs(toks, text, TOOLS) == {"p_stop": pytest.approx(1 / 3), "p_call": pytest.approx(2 / 3)}


def test_off_requests_no_logprobs_and_records_nothing():
    reqs, calls, steps = run([ASKING, SURE_STOP])
    assert "logprobs" not in reqs[0] and "p_stop" not in steps[0] and calls == [ASK]


def test_record_only_keeps_the_behaviour_and_drops_the_raw_logprobs():
    reqs, calls, steps = run([ASKING, SURE_STOP], record=True)
    assert reqs[0]["logprobs"] is True and reqs[0]["top_logprobs"] == 20
    assert calls == [ASK] and steps[0]["p_ask"] < 0.5 and "logprobs" not in steps[0]
    assert len(reqs) == 2


def test_force_act_first_forbids_stop_and_asking_on_the_first_decision_only():
    reqs, calls, _ = run([READING, SURE_STOP], force_act_first=True)
    assert actions(reqs[0]) == ["call"] and ASK not in names(reqs[0])
    assert actions(reqs[1]) == ["call", "stop"] and ASK in names(reqs[1])


def test_threshold_regenerates_a_doubtful_ask_once_before_calling_anything():
    reqs, calls, steps = run([ASKING, READING, SURE_STOP], mode="threshold", ask_min=0.5, stop_min=0.5)
    assert len(reqs) == 3 and calls == [READ]
    assert actions(reqs[1]) == ["call"] and ASK not in names(reqs[1])
    assert reqs[1]["messages"] == reqs[0]["messages"]
    o = steps[0]["decision_override"]
    assert o["from"] == "ask" and o["p"] < 0.5 and json.loads(o["raw"])["name"] == ASK
    assert steps[0]["name"] == READ and steps[0]["sec"] == 2.0


def test_threshold_keeps_a_confident_choice():
    _, calls, steps = run([SURE_STOP], mode="threshold", ask_min=0.5, stop_min=0.5)
    assert not calls and "decision_override" not in steps[0]


def test_first_scope_leaves_a_stop_after_a_read_alone():
    _, calls, steps = run([READING, STOPPING], mode="threshold", stop_min=0.5)
    assert calls == [READ] and "decision_override" not in steps[1]


def test_pre_write_scope_overrides_a_doubtful_stop_after_a_read():
    reqs, calls, steps = run([READING, STOPPING, WRITING, SURE_STOP], mode="threshold", stop_min=0.5,
                             scope="pre_write")
    assert calls == [READ, WRITE] and steps[1]["decision_override"]["from"] == "stop"


def test_pre_write_scope_ends_at_the_first_write_and_after_asking():
    _, calls, steps = run([WRITING, STOPPING], mode="threshold", stop_min=0.5, scope="pre_write")
    assert calls == [WRITE] and "decision_override" not in steps[1]
    _, calls, steps = run([ASKING, STOPPING], mode="threshold", stop_min=0.5, scope="pre_write")
    assert calls == [ASK] and "decision_override" not in steps[1]


def test_llm_passes_logprobs_through_meta():
    tok = SimpleNamespace(token="call", logprob=-0.1, top_logprobs=[SimpleNamespace(token="stop", logprob=-2.0)])
    resp = SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="x"),
                                                                logprobs=SimpleNamespace(content=[tok]))])
    llm = LLM()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: resp)))
    _, meta = llm.chat([], logprobs=True, top_logprobs=20)
    assert meta["logprobs"] == [("call", -0.1, [("stop", -2.0)])]


def test_history_alternatives_are_shown_under_their_user_turn_only_when_enabled():
    s = next(x for x in splits.select("train", "val_random") if any(t.get("role") == "user" for t in x.get("history", [])))
    tr = {"final": s["audio_text"], "turns": {"u1": "một"}, "alts": {"u1": ["một", "mốt", "mộc"], "final": ["x"]}}
    on = agents.ReactJSON(history_alts=True).user_prompt(s, tr)
    off = agents.ReactJSON().user_prompt(s, tr)
    assert 'USER: một\n  (nghe khác: "mốt" | "mộc")' in on
    assert "(nghe khác:" not in off


def test_a_byte_split_vietnamese_tail_keeps_the_decision_probabilities():
    text = call(WRITE, content="Mua sữa")
    toks = tokens(text, {"action": [("call", 0.9), ("stop", 0.1)]})
    tail, lp, alts = toks[-1]
    toks[-1] = (tail.replace("ữ", "�"), lp, alts)  # vLLM có thể trả mảnh byte UTF-8 ở phần arguments
    assert decision_probs(toks, text, TOOLS)["p_stop"] == pytest.approx(0.1)
    toks[0] = (toks[0][0] + "x", toks[0][1], [])        # lệch trước vị trí quyết định thì không đoán
    assert decision_probs(toks, text, TOOLS) is None


def test_later_steps_use_their_own_thresholds_and_the_override_budget():
    # bước sau: stop không bị ghi đè (later_stop_min 0), hỏi lại vẫn bị ghi đè theo later_ask_min
    _, calls, steps = run([READING, STOPPING], mode="threshold", scope="pre_write", ask_min=0.7, stop_min=0.7,
                          later_stop_min=0)
    assert calls == [READ] and "decision_override" not in steps[1]
    _, calls, steps = run([READING, ASKING, WRITING, SURE_STOP], mode="threshold", scope="pre_write",
                          ask_min=0.7, stop_min=0.7, later_stop_min=0, later_ask_min=0.7)
    assert calls == [READ, WRITE] and steps[1]["decision_override"]["from"] == "ask"
    # hết lượt ghi đè thì giữ lựa chọn của model
    _, calls, steps = run([STOPPING, READING, STOPPING], mode="threshold", scope="pre_write", stop_min=0.7,
                          max_overrides=1)
    assert calls == [READ] and steps[0]["decision_override"]["from"] == "stop"
    assert "decision_override" not in steps[1]
