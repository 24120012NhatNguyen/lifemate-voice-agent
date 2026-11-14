"""resolve.pick và luật ask_resolve của GuardedEnv (LM-47): trả ID của mục lời nói gọi tới thay cho câu hỏi lại."""
import pytest

from lifemate import data, resolve, splits
from lifemate.guards import GuardedEnv

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

ASK = "sys.ask_missing_info"
STRICT = dict(min_score=2, margin=2, exclusive=True, keys=resolve.NAME_KEYS)
ITEMS = [{"id": "c_1", "name": "Tiếng Nhật N5", "level": "cơ bản"},
         {"id": "c_2", "name": "Guitar đệm hát", "level": "cơ bản"},
         {"id": "c_3", "name": "Lập trình Python", "level": "nâng cao"}]


def test_pick_one_named_item():
    assert resolve.pick(ITEMS, "ghi bốn lăm phút học tiếng Nhật N5 tối qua giùm chị", **STRICT)["id"] == "c_1"
    assert resolve.pick(ITEMS, "ghi giờ học lập trình python hôm nay", **STRICT)["id"] == "c_3"


@pytest.mark.parametrize("said", [
    "ghi giờ học tối qua giùm chị",                         # không gọi tên mục nào
    "học tiếng Nhật N5 với guitar đệm hát",                 # gọi hai mục
    "ghi giờ học Nhật",                                     # chỉ một chữ phân biệt: chưa đủ chắc
])
def test_pick_abstains(said):
    assert resolve.pick(ITEMS, said, **STRICT) is None


def test_pick_needs_the_other_items_unnamed():
    """Mục khác cũng khớp một chữ ("tiếng" của "Tiếng Hàn") thì không chọn: luật chỉ chọn khi lời nói nhắc đúng một mục."""
    items = ITEMS + [{"id": "c_4", "name": "Tiếng Hàn sơ cấp"}]
    assert resolve.pick(items, "ghi bốn lăm phút học tiếng Nhật N5 tối qua giùm chị", **STRICT) is None


def test_pick_abstains_on_same_name():
    items = [{"id": "c_1", "name": "Minh Anh"}, {"id": "c_2", "name": "Minh Anh"}, {"id": "c_3", "name": "Lan"}]
    assert resolve.pick(items, "nhắn cho Minh Anh nhé", **STRICT) is None


def _case():
    """Mẫu train_rest đầu tiên mà gold đọc danh sách rồi ghi với ID lấy từ đó, và pick chọn đúng ID gold."""
    specs = data.tool_specs("train")
    for s in splits.select("train", "train_rest"):
        calls, results = s["label"]["tool_calls"], s["label"]["tool_results"]
        for i, c in enumerate(calls):
            if specs.get(c["name"], {}).get("kind") != "write" or i == 0:
                continue
            props = (specs[c["name"]].get("parameters") or {}).get("properties") or {}
            for k, v in c["arguments"].items():
                coll = str(props.get(k, {}).get("x-ref") or "").partition(".")[2]
                lst = results[i - 1].get(coll) if coll and isinstance(results[i - 1], dict) else None
                if isinstance(lst, list) and len(lst) >= 2 and (resolve.pick(lst, s["audio_text"], **STRICT) or {}).get("id") == v:
                    return s, calls[i - 1], c["name"], k, v
    pytest.skip("không có mẫu phù hợp")


def test_ask_resolve_answers_the_ask_with_the_named_id():
    s, read, tool, param, gold = _case()
    env = data.tool_env("train", s["id"])
    g = GuardedEnv(env, "train", s, s["audio_text"], ask_resolve=True)
    g.call(read["name"], read["arguments"])
    obs = g.call(ASK, {"tool_name": tool, "params": [param]})
    assert f"{param}={gold}" in obs["note"] and [c["name"] for c in env.get_tool_calls()] == [read["name"]]
    obs = g.call(ASK, {"tool_name": tool, "params": [param]})  # hỏi lần hai thì để qua
    assert obs.get("waiting_for_user") and env.get_tool_calls()[-1]["name"] == ASK


def test_ask_resolve_off_or_without_a_read_lets_the_ask_through():
    s, read, tool, param, _ = _case()
    for flags, reads in (({}, [read]), ({"ask_resolve": True}, [])):
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"], **flags)
        for r in reads:
            g.call(r["name"], r["arguments"])
        assert g.call(ASK, {"tool_name": tool, "params": [param]}).get("waiting_for_user")
