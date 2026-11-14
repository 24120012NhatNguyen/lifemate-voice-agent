"""Cờ guards.empty_retry (Pro-R, 12/11): call đọc có lọc chuỗi tuỳ chọn trả rỗng thì gọi lại không lọc."""
import pytest

from lifemate import data
from lifemate.guards import GuardedEnv, all_empty, broaden
from lifemate.runner import gold_hash
from lifemate.scoring import score_sample

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

TRAIN = data.samples("train")
SPECS = data.tool_specs("train")


def test_all_empty():
    assert all_empty({"ok": True, "books": []})
    assert all_empty({"ok": True, "books": [], "warnings": ["x"]})
    assert not all_empty({"ok": True, "books": [{"id": "b"}]})
    assert not all_empty({"ok": True, "done": False, "message": "không có"})  # không có danh sách: không phải "rỗng"
    assert not all_empty({"ok": False, "error": "x"})


def test_broaden_drops_only_optional_string_filters():
    assert broaden(SPECS["loyalty.get_points"], {"program": "Cine Việt"}) == {}
    assert broaden(SPECS["library.search"], {"query": "Xe Finan"}) == {}
    # ngày giờ ISO giữ nguyên: danh sách rỗng theo ngày là câu trả lời có nghĩa (gold train_0640)
    assert broaden(SPECS["calendar.list_events"], {"date_from": "2026-10-01", "date_to": "2026-10-01"}) is None
    assert broaden(SPECS["home_security.list_alerts"], {"since": "2026-10-01T09:00:00+07:00"}) is None
    # tham số bắt buộc giữ nguyên
    assert broaden(SPECS["news.search"], {"query": "khách"}) is None
    assert broaden(SPECS["flights.search"], {"origin": "HAN", "destination": "DAD", "date": "2026-10-11"}) is None
    # số giữ nguyên, chuỗi tuỳ chọn bỏ
    assert broaden(SPECS["sms.read_recent"], {"contact_id": "c_807", "limit": 5}) == {"limit": 5}


def _env(sid, **guards):
    s = data.by_id("train")[sid]
    env = data.tool_env("train", sid)
    return s, env, GuardedEnv(env, "train", s, s["audio_text"], **guards)


def test_retry_lists_everything_and_records_both_calls():
    # train_1110: "Cine Việt" không khớp "CineViệt"; gold là loyalty.get_points {} rồi redeem rw_01_1
    _, env, g = _env("train_1110", empty_retry=True)
    obs = g.call("loyalty.get_points", {"program": "Cine Việt"})
    assert obs["ok"] and obs["programs"] and "không lọc" in obs["note"]
    assert "rw_01_1" in str(obs)
    assert [c["arguments"] for c in env.get_tool_calls()] == [{"program": "Cine Việt"}, {}]
    assert g.log[-1]["decision"] == "chạy (gọi lại không lọc)"
    assert "rw_01_1" in g.known_ids() or any("rw_01_1" in str(r.get("result")) for r in g.log)


def test_flag_off_keeps_old_behaviour():
    _, env, g = _env("train_1110")
    obs = g.call("loyalty.get_points", {"program": "Cine Việt"})
    assert obs == {"ok": True, "programs": []}
    assert len(env.get_tool_calls()) == 1


def test_no_second_retry_when_unfiltered_call_already_ran():
    _, env, g = _env("train_1110", empty_retry=True)
    g.call("loyalty.get_points", {})
    obs = g.call("loyalty.get_points", {"program": "Cine Việt"})
    assert obs.get("programs") == [] and len(env.get_tool_calls()) == 2


def test_gold_replay_unchanged_with_empty_retry():
    """Phát lại call gold của cả train với cờ bật: không mẫu nào mất điểm, không call nào được thêm."""
    for s in TRAIN:
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"], empty_retry=True)
        for c in s["label"]["tool_calls"]:
            g.call(c["name"], c["arguments"])
        assert len(env.get_tool_calls()) == len(s["label"]["tool_calls"]), s["id"]
        r = score_sample("train", env.get_tool_calls(), env.get_state_hash(), s["label"]["tool_calls"], gold_hash("train", s))
        assert r["score"] == 1.0, s["id"]
