"""Luật sửa tham số call ghi (fixes.py) và cách GuardedEnv dùng chúng: đọc ngày giờ nói ra, không đụng giá trị có nguồn,
hỏi lại thay vì bịa số tiền hay giờ, và không đổi một call gold nào của train."""
import pytest

from lifemate import data, fixes as F
from lifemate.guards import GuardedEnv

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)

NOW = "2026-09-25T09:00:00+07:00"  # thứ Sáu
ALL = {"clock_fix": True, "money_k": True, "span_min": True}
SPECS = data.tool_specs("train")


@pytest.mark.parametrize("text, dates", [
    ("nhắc mình lúc 9 giờ tối mai", ["2026-09-26"]),
    ("tối nay với sáng ni", ["2026-09-25"]),
    ("thứ ba tuần sau", ["2026-09-29"]),
    ("thứ hai tuần sau nữa", ["2026-10-05"]),
    ("mười giờ sáng thứ ba, hôm hai chín ấy", ["2026-09-29"]),  # ngày cụ thể thắng thứ
    ("hạn tám giờ tối chủ nhật mười một tháng mười", ["2026-10-11"]),
    ("ngày ba mốt", ["2026-10-31"]),
    ("hôm qua chị Mai bảo", ["2026-09-24"]),  # "Mai" viết hoa giữa câu là tên
    ("hẹn hai ba người", []),
])
def test_spoken_dates(text, dates):
    assert F.spoken_dates(text, NOW) == dates


@pytest.mark.parametrize("text, clocks", [
    ("lúc 9 giờ kém 15 tối mai", ["20:45"]),
    ("Tối mai bảy giờ nhắc anh", ["19:00"]),  # buổi đứng trước
    ("Mai với thứ bảy, ba rưỡi chiều", ["15:30"]),  # "thứ bảy" không nuốt giờ đi sau
    ("đi chợ lúc tám giờ, không thì ba giờ chiều", ["08:00", "20:00", "15:00"]),  # "chiều" của giờ sau, không của "tám giờ"
    ("báo thức 6:30 nhé", ["06:30"]),
])
def test_spoken_clocks(text, clocks):
    assert F.spoken_clocks(text) == clocks


@pytest.mark.parametrize("text, minutes", [
    ("học từ bảy rưỡi tới tám giờ mười", 40),
    ("từ chín giờ tới mười giờ kém mười lăm", 45),
    ("từ chín giờ, à nhầm, chín rưỡi đến mười giờ hai mươi", 50),  # câu tự sửa: giờ sau cùng
    ("học bốn lăm phút", None),
])
def test_spoken_span(text, minutes):
    assert F.spoken_span(text) == minutes


def fix(name, args, said, seen=()):
    return F.fix_args(name, args, SPECS[name], said, NOW, set(seen), ALL)[0]


def test_clock_fix_reads_the_spoken_time():
    assert fix("reminders.create", {"content": "x", "time": "2026-09-26T18:45:00+07:00"},
               "nhắc lúc 9 giờ kém 15 tối mai")["time"] == "2026-09-26T20:45:00+07:00"
    # sai sáng chiều khi lời có nhiều giờ: chỉ sửa giờ cùng mặt đồng hồ
    assert fix("reminders.create", {"content": "x", "time": "2026-09-26T07:00:00+07:00"},
               "Tối mai bảy giờ nhắc uống canxi, với báo thức sáu giờ sáng")["time"] == "2026-09-26T19:00:00+07:00"


def test_clock_fix_moves_the_end_with_the_start():
    out = fix("calendar.create_event", {"title": "x", "start": "2026-09-29T08:00:00+07:00", "end": "2026-09-29T09:30:00+07:00"},
              "tạo lịch mười giờ sáng thứ ba tuần sau")
    assert (out["start"], out["end"]) == ("2026-09-29T10:00:00+07:00", "2026-09-29T11:30:00+07:00")


@pytest.mark.parametrize("said, seen", [
    ("nhắc anh đúng giờ họp tối mai", ["2026-09-26T18:45:00+07:00"]),  # giờ lấy từ kết quả tool
    ("nhắc anh trước giờ họp 15 phút, họp tám giờ tối mai", []),  # dời theo thời lượng
    ("nhắc bắt đầu nấu sao cho kịp bảy giờ tối mai ăn cơm", []),
])
def test_clock_fix_leaves_values_with_a_source(said, seen):
    args = {"content": "x", "time": "2026-09-26T18:45:00+07:00"}
    assert fix("reminders.create", args, said, seen) == args


def test_money_k_and_span_min():
    assert fix("ewallet.transfer", {"contact_id": "c_01", "amount": 350}, "chuyển cô Ly ba trăm rưỡi")["amount"] == 350000
    assert fix("exchange_rate.convert", {"from_currency": "USD", "to_currency": "VND", "amount": 500},
               "đổi năm trăm đô")["amount"] == 500  # ngoại tệ
    assert fix("learning.log_progress", {"course_id": "course_1", "minutes": 60},
               "học từ bảy rưỡi tới tám giờ mười")["minutes"] == 40


def missing(name, args, said, seen=()):
    return F.missing(name, args, SPECS[name], said, NOW, set(seen), final=said)


def test_missing_asks_for_an_amount_or_a_time_nobody_said():
    assert missing("ewallet.transfer", {"contact_id": "c_02", "amount": 100000}, "chuyển ví cho Minh chút tiền") == ["amount"]
    assert missing("ewallet.transfer", {"contact_id": "c_02", "amount": 200000}, "chuyển ví anh Long hai trăm nha") == []
    assert missing("reminders.create", {"content": "x", "time": "2026-09-26T14:00:00+07:00"}, "Nhắc tôi đi họp phụ huynh với.") == ["time"]
    assert missing("reminders.create", {"content": "x", "time": "2026-09-25T09:55:00+07:00"}, "giặt xong thì nhắc mình") == []
    # "Tí" là tên trong một lượt trước, không phải "tí tiền"
    assert F.missing("stocks.set_price_alert", {"symbol": "ZTP", "price": 23000, "direction": "below"},
                     data.tool_specs("public_test")["stocks.set_price_alert"], "nhắn thằng Tí. đặt cảnh báo giùm", NOW, set(),
                     final="đặt cảnh báo giùm") == []


@pytest.mark.parametrize("said, ask", [
    ("Mấy gói Taipei trừ tiền á, em hủy bớt một gói thôi.", ["subscription_id"]),
    ("Mấy đơn đồ ăn đang giao, huỷ giùm em một đơn thôi nghe.", ["subscription_id"]),
    ("đặt chuyến bay với chuyến tàu đó, mỗi cái một vé thôi", []),
    ("Sữa xanh mà tăng thì bán bớt một trăm cổ", []),
    ("trạm đó còn xe thì giữ cho chị một chiếc, không thì thôi", []),
    ("tiền bên thanh toán có đủ hai cái, con mở một cái thôi nha", []),  # mở mới, không chọn trong cái có sẵn
])
def test_pick_one(said, ask):
    assert F.pick_one("subscriptions.cancel", {"subscription_id": "sub_842"}, SPECS["subscriptions.cancel"], said) == ask


SAMPLE = data.by_id("train")["train_0244"]  # "... chuyển ví cho Minh chút tiền giùm tôi."


def test_guarded_env_defers_the_ask_to_the_end_and_asks_once():
    env = data.tool_env("train", SAMPLE["id"])
    g = GuardedEnv(env, "train", SAMPLE, SAMPLE["audio_text"], ask_missing=True)
    obs = g.call("ewallet.transfer", {"contact_id": "c_02", "amount": 100000})
    assert obs["ok"] and "ở cuối" in obs["note"] and env.get_tool_calls() == []
    g.call("ewallet.transfer", {"contact_id": "c_02", "amount": 50000})  # thử số khác: vẫn không chạy
    g.call("alarm.create", {"time": "2026-11-13T06:30:00+07:00"})  # phần đủ thông tin vẫn làm
    g.finish()
    assert env.get_tool_calls() == [
        {"name": "alarm.create", "arguments": {"time": "2026-11-13T06:30:00+07:00"}},
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}},
    ]


def test_finish_does_not_repeat_an_ask_the_llm_made_itself():
    env = data.tool_env("train", SAMPLE["id"])
    g = GuardedEnv(env, "train", SAMPLE, SAMPLE["audio_text"], ask_missing=True)
    g.call("ewallet.transfer", {"contact_id": "c_02", "amount": 100000})
    g.call("sys.ask_missing_info", {"tool_name": "ewallet.transfer", "params": ["amount"]})
    g.finish()
    assert [c["name"] for c in env.get_tool_calls()] == ["sys.ask_missing_info"]


def test_guarded_env_without_rules_runs_the_call_as_proposed():
    env = data.tool_env("train", SAMPLE["id"])
    GuardedEnv(env, "train", SAMPLE, SAMPLE["audio_text"]).call("ewallet.transfer", {"contact_id": "c_02", "amount": 100000})
    assert [c["name"] for c in env.get_tool_calls()] == ["ewallet.transfer"]


def test_no_rule_changes_a_gold_call_of_train():
    """Chạy call gold của cả 1.545 mẫu qua GuardedEnv với mọi luật bật: call, tham số và state phải y nguyên."""
    from lifemate.runner import gold_hash
    rules = dict(ALL, ask_missing=True, pick_one=True)
    changed = []
    for s in data.samples("train"):
        env = data.tool_env("train", s["id"])
        g = GuardedEnv(env, "train", s, s["audio_text"], validate=False, normalize=False, calendar_duration=False, **rules)
        for c in s["label"]["tool_calls"]:
            g.call(c["name"], c["arguments"])
        g.finish()
        if env.get_tool_calls() != s["label"]["tool_calls"] or env.get_state_hash() != gold_hash("train", s):
            changed.append(s["id"])
    assert not changed
