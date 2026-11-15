"""Luật nhẹ Pro 15/11 (LIGHT_RULES_1115): guards.xref_guard, guards.explicit_defaults, guards.ask_schema. Mặc định tắt."""
import pytest

from lifemate import data
from lifemate.guards import DEFAULTS, GuardedEnv, id_collections, read_defaults
from lifemate.runner import gold_hash
from lifemate.scoring import score_faq, score_sample

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

TRAIN = data.samples("train")
SPECS = data.tool_specs("train")
LIGHT = {"xref_guard": True, "explicit_defaults": True, "ask_schema": True}


def _env(sid, **guards):
    s = data.by_id("train")[sid]
    env = data.tool_env("train", sid)
    return s, env, GuardedEnv(env, "train", s, s["audio_text"], **guards)


def test_flags_default_off():
    assert not any(DEFAULTS[k] for k in LIGHT)


def test_id_collections_only_list_members():
    out = {}
    id_collections({"ok": True, "events": [{"id": "evt_1"}], "item": {"id": "ord_1"}, "rules": [{"id": "r_1", "x": {"id": "n"}}]}, out)
    assert out == {"evt_1": {"events"}, "r_1": {"rules"}}  # "item" lẻ và dict lồng không gán bảng


def test_read_defaults_from_schema_text():
    assert read_defaults(SPECS["weather.get_forecast"], {"date": "2026-11-12"}, "Đà Nẵng") == {"location": "Đà Nẵng"}
    assert read_defaults(SPECS["weather.get_forecast"], {"location": "Huế"}, "Đà Nẵng") == {}
    assert read_defaults(SPECS["weather.get_forecast"], {}, None) == {}
    assert read_defaults(SPECS["maps.get_route"], {"origin": "a", "destination": "b"}, "Huế") == {"mode": "motorbike"}
    assert read_defaults(SPECS["maps.get_route"], {"origin": "a", "destination": "b", "mode": "walk"}, None) == {}
    # tool ghi không bao giờ được điền, kể cả khi mô tả nêu mặc định (shopping_list.add quantity, calendar end)
    assert read_defaults(SPECS["shopping_list.add"], {"name": "táo"}, "Huế") == {}
    assert read_defaults(SPECS["calendar.create_event"], {"title": "x", "start": "2026-10-01T09:00:00+07:00"}, "Huế") == {}


def test_explicit_defaults_same_observation():
    s, env, g = _env("train_0037", explicit_defaults=True)
    plain = data.tool_env("train", "train_0037").call("weather.get_forecast", {"date": "2026-11-12"})
    obs = g.call("weather.get_forecast", {"date": "2026-11-12"})
    assert obs == plain
    assert env.get_tool_calls()[0]["arguments"] == {"date": "2026-11-12", "location": s["context"]["user_profile"]["city"]}
    assert g.log[-1]["defaults"] == {"location": s["context"]["user_profile"]["city"]}


def test_explicit_defaults_off_keeps_call():
    _, env, g = _env("train_0037")
    g.call("weather.get_forecast", {"date": "2026-11-12"})
    assert env.get_tool_calls()[0]["arguments"] == {"date": "2026-11-12"}


def _event_id(obs):
    return next(e["id"] for e in obs["events"])


def test_xref_guard_blocks_id_from_another_collection():
    _, env, g = _env("train_1435", xref_guard=True)
    ev = _event_id(g.call("calendar.list_events", {}))
    obs = g.call("learning.log_progress", {"course_id": ev, "minutes": 30})
    assert obs["ok"] is True and obs["done"] is False and "learning.list_courses" in obs["message"]
    assert len(env.get_tool_calls()) == 1  # call bị chặn không vào bài nộp
    assert g.log[-1]["decision"] == "chặn: ID sai bảng (xref_guard)"
    # ID đúng bảng (hay ID chưa thấy ở đâu) vẫn chạy
    course = g.call("learning.list_courses", {})["courses"][0]["id"]
    g.call("learning.log_progress", {"course_id": course, "minutes": 30})
    g.call("learning.log_progress", {"course_id": "course_chua_thay", "minutes": 30})
    assert len(env.get_tool_calls()) == 4


def test_xref_guard_off_runs_call():
    _, env, g = _env("train_1435")
    ev = _event_id(g.call("calendar.list_events", {}))
    g.call("learning.log_progress", {"course_id": ev, "minutes": 30})
    assert len(env.get_tool_calls()) == 2


def test_ask_schema_drops_unknown_params():
    _, env, g = _env("train_0828", ask_schema=True)
    obs = g.call("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": ["tracking_id"]})
    assert "note" in obs and "address" in obs["note"] and env.get_tool_calls() == []
    # hỏi lần hai cùng tool: để đi như cũ (không vòng lặp)
    g.call("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": ["tracking_id"]})
    assert len(env.get_tool_calls()) == 1
    _, env, g = _env("train_0828", ask_schema=True)
    g.call("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": ["tracking_id", "date"]})
    assert env.get_tool_calls()[0]["arguments"]["params"] == ["date"]


def test_ask_schema_off_and_valid_asks_untouched():
    _, env, g = _env("train_0828")
    g.call("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": ["tracking_id"]})
    assert env.get_tool_calls()[0]["arguments"]["params"] == ["tracking_id"]
    _, env, g = _env("train_0828", ask_schema=True)
    g.call("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": ["date"]})
    assert env.get_tool_calls()[0]["arguments"]["params"] == ["date"]


def test_gold_replay_with_light_rules():
    """Phát lại call gold của cả train với ba cờ bật: không call nào bị chặn hay đổi câu hỏi, state không đổi, điểm FAQ 1.0;
    chỉ thêm tham số mặc định vào call đọc."""
    added = 0
    for s in TRAIN:
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"], **LIGHT)
        for c in s["label"]["tool_calls"]:
            g.call(c["name"], c["arguments"])
        calls = env.get_tool_calls()
        assert len(calls) == len(s["label"]["tool_calls"]), s["id"]
        for got, want in zip(calls, s["label"]["tool_calls"]):
            assert got["name"] == want["name"], s["id"]
            extra = {k: v for k, v in got["arguments"].items() if k not in want["arguments"]}
            assert {k: got["arguments"][k] for k in want["arguments"]} == want["arguments"], s["id"]
            assert not extra or SPECS[got["name"]]["kind"] == "read", s["id"]
            added += bool(extra)
        h = gold_hash("train", s)
        assert env.get_state_hash() == h, s["id"]
        assert score_faq("train", calls, h, s["label"]["tool_calls"], h)["score"] == 1.0, s["id"]
    assert added > 0


def test_gold_replay_flags_off_identical():
    for s in TRAIN[:300]:
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"])
        for c in s["label"]["tool_calls"]:
            g.call(c["name"], c["arguments"])
        assert env.get_tool_calls() == s["label"]["tool_calls"], s["id"]
        r = score_sample("train", env.get_tool_calls(), env.get_state_hash(), s["label"]["tool_calls"], gold_hash("train", s))
        assert r["score"] == 1.0, s["id"]


def test_day_filter_fills_single_spoken_day():
    from lifemate.guards import day_params
    assert day_params(SPECS["calendar.list_events"]) == ["date_from", "date_to"]
    assert day_params(SPECS["home_security.list_alerts"]) == []  # ISO có giờ: không đụng
    assert day_params(SPECS["calendar.create_event"]) == []  # tool ghi
    s = data.by_id("train")["train_0808"]
    env = data.tool_env("train", "train_0808")
    g = GuardedEnv(env, "train", s, "Ừ, ghi lại buổi chạy lúc sáu giờ sáng mai cho anh nhé", day_filter=True)
    g.call("calendar.list_events", {})
    args = env.get_tool_calls()[0]["arguments"]
    assert args["date_from"] == args["date_to"] and args["date_from"][:4] == "2026"
    assert g.log[-1]["day_filter"] == args
    # đã có ngày, hay lời nêu hai ngày / không ngày: giữ nguyên
    env = data.tool_env("train", "train_0808")
    g = GuardedEnv(env, "train", s, "mai với ngày kia", day_filter=True)
    g.call("calendar.list_events", {})
    assert env.get_tool_calls()[0]["arguments"] == {}
    env = data.tool_env("train", "train_0808")
    g = GuardedEnv(env, "train", s, "sáng mai", day_filter=True)
    g.call("calendar.list_events", {"date_from": "2026-10-01"})
    assert env.get_tool_calls()[0]["arguments"] == {"date_from": "2026-10-01"}


def test_gold_replay_day_filter_never_hides_used_ids():
    """Cờ day_filter trên lời gold: state không đổi, điểm FAQ 1.0 (kiểm ID bị giấu nằm trong LIGHT_RULES_1115, mục day_filter)."""
    for s in TRAIN:
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"], day_filter=True)
        for c in s["label"]["tool_calls"]:
            g.call(c["name"], c["arguments"])
        h = gold_hash("train", s)
        assert env.get_state_hash() == h, s["id"]
        assert score_faq("train", env.get_tool_calls(), h, s["label"]["tool_calls"], h)["score"] == 1.0, s["id"]
