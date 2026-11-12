"""Cổng của nấc 0: bộ chấm và khung chạy phải đúng trước khi so bất kỳ kiến trúc nào."""
import pytest

from lifemate import config, data, splits
from lifemate.asr.adapters import _distinct
from lifemate.asr.base import app_phrases, bias_hint, proper_names
from lifemate.asr.post import PhoneticEntityFix, phonetic_key
from lifemate.asr.train_data import ctc_text, utterances
from lifemate.context import timetable
from lifemate.guards import GuardedEnv
from lifemate.runner import gold_hash
from lifemate.scoring import score_sample, token_f1

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

TRAIN = data.samples("train")


def replay(sample, **guards):
    env = data.tool_env("train", sample["id"])
    g = GuardedEnv(env, "train", sample, sample["audio_text"], **guards)
    for c in sample["label"]["tool_calls"]:
        g.call(c["name"], c["arguments"])
    return score_sample("train", env.get_tool_calls(), env.get_state_hash(), sample["label"]["tool_calls"], gold_hash("train", sample))


def test_gold_scores_100_and_matches_final_state():
    for s in TRAIN:
        env = data.tool_env("train", s["id"])
        for c in s["label"]["tool_calls"]:
            env.call(c["name"], c["arguments"])
        st = env.get_state()
        assert all(st.get(app) == v for app, v in s["label"]["final_state"].items()), s["id"]
        r = score_sample("train", env.get_tool_calls(), env.get_state_hash(), s["label"]["tool_calls"], env.get_state_hash())
        assert r["score"] == 1.0, s["id"]


def test_score_formula():
    s = next(x for x in TRAIN if len(x["label"]["tool_calls"]) == 3)
    g, h = s["label"]["tool_calls"], gold_hash("train", s)
    extra = g + [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "x", "params": []}}]
    assert score_sample("train", extra, h, g, h)["score"] == pytest.approx(2 * 0.75 * 1 / 1.75)  # thừa 1 call, |G|=3
    assert score_sample("train", g, "khác", g, h)["score"] == 0.0
    empty = next(x for x in TRAIN if not x["label"]["tool_calls"])
    assert score_sample("train", [], gold_hash("train", empty), [], gold_hash("train", empty))["score"] == 1.0


def test_fuzzy_threshold():
    assert token_f1("mai chạy bộ dời sang 09:00 nhé", "mai chạy bộ dời sang 09:00 nhé") == 1.0
    assert token_f1("Em chào sếp, mai trời mưa nên lịch chạy bộ dời sang 9 giờ sáng nhé ạ", "mai chạy bộ dời sang 09:00 nhé") < 0.6


def test_default_guards_keep_gold_at_100():
    assert all(replay(s)["score"] == 1.0 for s in TRAIN)


def test_html_guards_block_history_ids():
    html = dict(id_guard="block", id_sources=["turn"], id_harvest="html", dup_guard="block")
    lost = sum(1 for s in TRAIN if replay(s, **html)["score"] < 1)
    assert 60 <= lost <= 80  # 72 mẫu mất điểm vì ID lấy từ lịch sử bị chặn


def test_splits_are_disjoint_and_stable():
    groups = {g: {s["id"] for s in splits.select("train", g)} for g in ("val_app", "val_random", "train_rest")}
    assert sum(map(len, groups.values())) == len(TRAIN)
    assert not (groups["val_app"] & groups["val_random"])
    assert len(groups["val_app"]) == 301


def test_config_extends_and_override(tmp_path):
    cfg = config.load(config.REPO / "configs/experiments/02_oracle_html_guards.yaml", ["limit=5"])
    assert cfg["name"] == "02_oracle_html_guards"
    assert cfg["guards"]["id_guard"] == "block" and cfg["guards"]["validate"] is True
    assert cfg["limit"] == 5


def test_timetable_covers_next_week():
    t = timetable("2026-11-28T07:10:00+07:00", "two_weeks")
    assert "2026-11-29" in t and "2026-12-07" in t


def test_phonetic_key_merges_dialect_pairs():
    assert phonetic_key("Vân") == phonetic_key("Dân")
    assert phonetic_key("Trang") == phonetic_key("Chan")
    s = {"context": {"user_profile": {"name": "Em Linh", "city": "Đà Nẵng"}},
         "history": [{"role": "assistant", "text": "Đã gọi cho Trần Văn Hùng."}]}
    out = PhoneticEntityFix().apply(s, {"final": "nhắn cho Chần Dăn Hùng nhé"})
    assert "Trần Văn Hùng" in out["final"]


def test_asr_train_data_excludes_val():
    train, dev = utterances("train"), utterances("dev")
    ids = {u["key"].split("/")[0] for u in train + dev}
    assert ids and not ids & {x["id"] for x in TRAIN if splits.group(x) != "train_rest"}
    assert not {u["key"] for u in train} & {u["key"] for u in dev}
    assert ctc_text("Nạp 50.000 lúc 9:30, nhé!") == "nạp năm mươi nghìn lúc chín giờ ba mươi nhé"


def test_hint_names_and_app_vocab():
    # lớp [À-Ỹ] cũ bắt cả chữ thường có dấu ("ền điện", "ủa Sơn Tùng")
    assert proper_names("Đã trả tiền điện của Sơn Tùng, nhắn Sếp Dũng và Mẹ.") == ["Sơn Tùng", "Sếp Dũng"]
    s = next(x for x in TRAIN if "tv" in x["context"]["installed_apps"])
    names, full = bias_hint(s, "profile_history"), bias_hint(s, "profile_history_apps")
    assert full.startswith(names) and "ghi hình" in full
    assert not any("_" in w or ":" in w for words in app_phrases("train").values() for w in words)


def test_nbest_alts_differ_in_words():
    assert _distinct("Giao hàng nhé.", ["giao hàng nhé", "Sao hàng nhé.", "Sao hàng, nhé", "Giao hàng nha."]) == \
        ["Sao hàng nhé.", "Giao hàng nha."]


def test_app_rank_counts_train_only(monkeypatch, tmp_path):
    """Thứ hạng mồi app luôn lấy từ train (hoặc file $LIFEMATE_APP_RANK), không đếm trên tập đang chạy."""
    import json as _json

    from lifemate.asr import base
    base._app_rank.cache_clear()
    f = tmp_path / "rank.json"
    f.write_text(_json.dumps({"zeta": 0, "alpha": 5}), encoding="utf-8")
    monkeypatch.setenv("LIFEMATE_APP_RANK", str(f))
    try:
        assert base._app_rank() == {"zeta": 0, "alpha": 5}
    finally:
        base._app_rank.cache_clear()
