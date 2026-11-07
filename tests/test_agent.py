"""Vòng ReAct của react_json: giới hạn thời gian mỗi mẫu, và hỏi lại khi bước đầu dừng mà máy có app chưa thấy."""
import json
from types import SimpleNamespace

import pytest

from lifemate import data, splits
from lifemate.agent import agents
from lifemate.agent.llm import LLM, LLMTimeout

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

STOP = json.dumps({"thought": "", "action": "stop"})


def call(name, **args):
    return json.dumps({"thought": "", "action": "call", "name": name, "arguments": args})


class FakeLLM:
    """Trả lần lượt các câu đã định (Exception thì ném ra); ghi lại tin nhắn và tham số của từng lượt."""

    def __init__(self, replies, sec=0.0, clock=None):
        self.replies, self.sec, self.clock, self.requests = list(replies), sec, clock, []

    def chat(self, messages, **kw):
        self.requests.append({"messages": list(messages), **kw})
        if self.clock:
            self.clock.t += self.sec
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply), {"sec": self.sec, "prompt_tokens": 1, "completion_tokens": 1}


class FakeEnv:
    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}


def unseen(sample):
    known = splits.known_apps("train_rest")
    return [app for app in sample["context"]["installed_apps"] if app not in known]


def tools_of(sample, apps):
    return [t for app in apps for t in sample["context"]["installed_apps"][app]["tools"]]


NEW = next(s for s in splits.select("train", "val_app") if unseen(s))
OLD = next(s for s in splits.select("train", "val_random") if not unseen(s))
NEW_TOOL = tools_of(NEW, unseen(NEW))[0]
OLD_TOOL = tools_of(OLD, list(OLD["context"]["installed_apps"]))[0]


def run(sample, replies, transcript=None, **opts):
    a = agents.ReactJSON(**opts)
    a.llm = FakeLLM(replies)
    env = FakeEnv()
    steps = a.run("train", sample, transcript or {"final": sample["audio_text"]}, env)["steps"]
    return a.llm.requests, env.calls, steps


def test_known_apps_leave_out_the_heldout_apps():
    assert not splits.known_apps("train_rest") & set(splits.HELDOUT_APPS)
    assert set(splits.HELDOUT_APPS) <= splits.known_apps("all")


def test_nudge_rechecks_a_bare_first_stop_with_the_unseen_tools():
    reqs, calls, steps = run(NEW, [STOP, call(NEW_TOOL), STOP], recheck_stop={"mode": "nudge"})
    assert [s["action"] for s in steps] == ["stop", "call", "stop"]
    assert steps[0]["recheck"] == "nudge" and calls == [(NEW_TOOL, {})]
    nudge = reqs[1]["messages"][-1]["content"]
    assert all(f"- {t}:" in nudge for t in tools_of(NEW, unseen(NEW)))
    seen = [a for a in NEW["context"]["installed_apps"] if a not in unseen(NEW)]
    assert not any(f"- {t}:" in nudge for t in tools_of(NEW, seen))


def test_a_second_stop_after_the_nudge_is_final():
    reqs, calls, steps = run(NEW, [STOP, STOP], recheck_stop={"mode": "nudge"})
    assert [s["action"] for s in steps] == ["stop", "stop"] and len(reqs) == 2 and not calls


@pytest.mark.parametrize("sample, replies, opts", [
    (OLD, [STOP], {"mode": "nudge"}),                                  # mọi app đã cài đều đã thấy
    (NEW, [STOP], None),                                               # tắt
    (NEW, [STOP], {"mode": "nudge", "known": list(NEW["context"]["installed_apps"])}),  # danh sách app (Docker)
    (NEW, [call(NEW_TOOL), STOP], {"mode": "nudge"}),                  # đã gọi rồi mới dừng
])
def test_no_recheck(sample, replies, opts):
    reqs, _, steps = run(sample, replies, recheck_stop=opts)
    assert len(reqs) == len(replies) and not any(s.get("recheck") for s in steps)


def test_force_constrains_only_the_next_turn_and_drops_the_stop():
    reqs, calls, steps = run(NEW, [STOP, call(NEW_TOOL), STOP], recheck_stop={"mode": "force"})
    props = [r["response_format"]["json_schema"]["schema"]["properties"] for r in reqs]
    assert props[1]["action"]["enum"] == ["call"]
    assert props[1]["name"]["enum"] == tools_of(NEW, unseen(NEW)) + ["sys.ask_missing_info"]
    assert props[0] == props[2] and props[2]["action"]["enum"] == ["call", "stop"]
    assert reqs[1]["messages"] == reqs[0]["messages"]  # câu stop không vào hội thoại
    assert steps[0]["recheck"] == "force" and calls == [(NEW_TOOL, {})]


def test_time_budget_counts_asr_and_stops_before_the_limit(monkeypatch):
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(agents.time, "perf_counter", lambda: clock.t)
    a = agents.ReactJSON(time_budget=10, min_step_sec=1.5)
    a.llm = FakeLLM([call(OLD_TOOL)] * 8, sec=3.0, clock=clock)
    env = FakeEnv()
    steps = a.run("train", OLD, {"final": "x", "sec": 2.0}, env)["steps"]
    # 2 s ASR, 3 s mỗi lượt: còn 8, 5, 2 s thì vẫn gọi; còn −1 s thì dừng
    assert [s["action"] for s in steps] == ["call"] * 3 + ["budget_stop"]
    assert [r["timeout"] for r in a.llm.requests] == [8.0, 5.0, 2.0] and len(env.calls) == 3


def test_a_turn_that_times_out_keeps_the_calls_made_so_far():
    _, calls, steps = run(OLD, [call(OLD_TOOL), LLMTimeout("quá giờ")], time_budget=26)
    assert [s["action"] for s in steps] == ["call", "timeout"] and len(calls) == 1


def test_llm_turns_an_openai_timeout_into_llmtimeout_without_retries():
    import openai

    def create(**_):
        raise openai.APITimeoutError.__new__(openai.APITimeoutError)  # không cần dựng request của thư viện HTTP

    llm, opts = LLM(), {}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.client = SimpleNamespace(with_options=lambda **o: opts.update(o) or client)
    with pytest.raises(LLMTimeout):
        llm.chat([], timeout=2.5)
    assert opts == {"timeout": 2.5, "max_retries": 0}
