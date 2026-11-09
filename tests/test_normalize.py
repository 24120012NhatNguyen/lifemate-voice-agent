"""Tầng N (normalize.py): đọc số điện thoại, ngày theo mốc, giờ tính từ bây giờ, phép dời, số tiền, phần trăm; khối gợi
ý trong prompt; ba luật sửa tham số phone_fix, rel_day_fix, now_offset_fix; không đổi call gold nào của train."""
import pytest

from lifemate import normalize as Z

NOW = "2026-10-05T09:00:00+07:00"  # thứ Hai


@pytest.mark.parametrize("text, want", [
    ("gọi giúp tôi số không chín không một, hai ba bốn, năm sáu bảy bảy nhé", ["09012345677"]),
    ("nạp 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha", ["09505360367"]),
    ("xong gọi cho số trực 09889990002 nhé", ["09889990002"]),
    ("số mới là 0988 999 000 2", ["09889990002"]),
    ("đổi thành +84 912 345 678", ["0912345678"]),
    ("gọi số à không không chín không một hai ba bốn năm sáu bảy", ["0901234567"]),  # "không" phủ định liền trước
    ("hẹn hai ba bốn năm người đi ăn", []),  # đếm, không có từ gợi ý
    ("chuyển hai triệu ba trăm nghìn", []),  # lượng, không phải dãy chữ số
    ("mã đơn 12345", []),
])
def test_phones(text, want):
    assert Z.phones(text) == want


def test_phone_suffixes():
    assert Z.phone_suffixes("Số đuôi tám sáu tám gọi nhỡ anh mấy giờ thế?") == ["868"]
    assert Z.phone_suffixes("số đuôi 4567") == ["4567"]
    assert Z.phone_suffixes("đuôi xe") == []


def test_clauses_split_on_punctuation_and_connectives():
    assert Z.clauses("Bật đèn, rồi nhắc tôi lúc 9 giờ. Với cả tắt quạt") == [
        "bật đèn", "rồi nhắc tôi lúc 9 giờ", "với cả tắt quạt"]


@pytest.mark.parametrize("text, day, clocks", [
    ("thứ sáu này hẹn bưu điện lấy hàng nha. Với cả sáu giờ chiều hôm trước nhắc mình in nhãn", "2026-10-08", ("18:00",)),
    ("thứ sáu này đi khám, hôm sau nhắc mình uống thuốc lúc tám giờ sáng", "2026-10-10", ("08:00",)),
    ("chủ nhật này có tiệc, trước đó một ngày nhắc chị đi chợ", "2026-10-10", ()),
    ("thứ ba tuần sau họp, cùng hôm đó đặt xe lúc bảy giờ sáng", "2026-10-13", ("07:00",)),
])
def test_anchored(text, day, clocks):
    (a,) = Z.anchored(text, NOW)
    assert (a.day, a.clocks) == (day, clocks)


def test_anchored_needs_exactly_one_spoken_date():
    assert Z.anchored("sáu giờ chiều hôm trước nhắc mình in nhãn", NOW) == []  # mốc nằm trong dữ liệu
    assert Z.anchored("thứ sáu hay thứ bảy cũng được, hôm trước nhắc mình", NOW) == []  # hai ngày: không đoán


def test_anchored_falls_back_to_the_previous_turn():
    (a,) = Z.anchored("ghi vào lịch hai giờ chiều hôm đó là bưu tá qua lấy hàng", NOW,
                      past="Thứ sáu tuần sau anh có kiện hàng cần gửi")
    assert (a.day, a.clocks) == ("2026-10-16", ("14:00",))


@pytest.mark.parametrize("text, want", [
    ("Buổi đầu tám giờ sáng thứ hai tuần sau, em tạo sự kiện", ["2026-10-12T08:00:00+07:00"]),
    ("mai nhắc tôi, lúc chín giờ sáng nhé", ["2026-10-06T09:00:00+07:00"]),  # ngày ở vế khác
    ("mười giờ nhắc chị cầm ô", ["2026-10-05T10:00:00+07:00"]),  # chỉ nói giờ: hôm nay, khả năng còn ở phía trước
    ("Đặt báo thức sáng mai năm giờ, à không, năm rưỡi thôi nhé", ["2026-10-06T05:30:00+07:00"]),  # câu tự sửa
    ("nhắc tôi trước giờ họp một tiếng, lúc tám giờ", []),  # giờ theo mốc trong dữ liệu
])
def test_datetimes(text, want):
    assert [iso for iso, _ in Z.datetimes(text, NOW) if iso[11:13] != "17"] == want


@pytest.mark.parametrize("text, want", [
    ("45 phút nữa nhắc chị gọi lại cho khách nhé", ["2026-10-05T09:45:00+07:00"]),
    ("hai tiếng rưỡi nữa nhắc anh", ["2026-10-05T11:30:00+07:00"]),
    ("nửa tiếng nữa bật máy giặt", ["2026-10-05T09:30:00+07:00"]),
    ("ba ngày nữa nhắc anh gia hạn", ["2026-10-08"]),
    ("hai tiếng sau khi họp xong", []),  # "sau" theo mốc khác, không phải từ bây giờ
])
def test_now_offsets(text, want):
    assert Z.now_offsets(text, NOW) == want


def test_offsets():
    got = {(o.minutes, o.label) for o in Z.offsets(
        "đón sớm bốn mươi lăm phút, nhắc em mang laptop trước lúc xe tới mười lăm phút, trước hạn khiếu nại một ngày")}
    assert got == {(-45, "sớm bốn mươi lăm phút"), (-15, "trước lúc xe tới mười lăm phút"),
                   (-1440, "trước hạn khiếu nại một ngày")}
    assert [(o.minutes, o.label) for o in Z.offsets("qua hạn sớm nhất một ngày thì đặt lịch chín giờ sáng")] == [
        (1440, "qua hạn sớm nhất một ngày")]  # "sớm nhất" là tên mốc, "qua" là sau
    assert Z.offsets("thứ ba tuần sau") == []
    assert Z.offsets("45 phút nữa nhắc chị") == []


def test_amounts_percents_durations():
    assert Z.amounts("Nạp 50 ngàn, chuyển một triệu rưỡi") == [(50000, ""), (1500000, "")]
    assert Z.amounts("nâng mục tiêu thêm hai nghìn bước") == [(2000, "bước")]
    assert Z.amounts("chuyển hai trăm cho Lan") == []  # money_k lo
    assert Z.percents("để mờ 30 phần trăm thôi") == [30]
    assert Z.percents("giảm độ sáng xuống còn một nửa") == [50]
    assert Z.durations("học bốn lăm phút, chạy tiếng rưỡi") == [45, 90]
    assert Z.durations("thứ hai tuần sau") == []  # không phải "hai tuần"


def test_render_and_merge_mark_values_only_in_alternatives():
    fs = Z.merge(["gọi số không chín không một hai ba bốn năm sáu bảy bảy",
                  "gọi số không chín không hai hai ba bốn năm sáu bảy bảy"], NOW)
    assert [(f.kind, f.value, f.source) for f in fs if f.kind == "phone"] == [
        ("phone", "09012345677", "top"), ("phone", "09022345677", "alt")]
    block = Z.render(fs)
    assert block.startswith("GIÁ TRỊ MÁY ĐỌC TỪ LỜI NÓI")
    assert "09022345677 (chỉ có ở bản nghe khác)" in block
    assert Z.render([]) == ""


def test_hint_reads_transcript_alternatives_and_previous_turns():
    tr = {"final": "ghi lịch hai giờ chiều hôm đó", "alts": {"final": []}, "turns": {"u1": "thứ sáu tuần sau có hàng"}}
    assert "2026-10-16T14:00:00+07:00" in Z.hint(tr, NOW)


# ------------------------------------------------------------------ luật sửa tham số (cần dữ liệu BTC cho catalog)

def _specs():
    from lifemate import data
    try:
        data.data_root()
    except FileNotFoundError:
        pytest.skip("không có gói dữ liệu BTC")
    return data.tool_specs("train")


def _fix(name, args, said, seen=(), past=""):
    from lifemate import fixes as F
    rules = {"phone_fix": True, "rel_day_fix": True, "now_offset_fix": True}
    return F.fix_args(name, args, _specs()[name], said, NOW, set(seen), rules, past=past)[0]


def test_phone_fix():
    said = "Em gọi giúp tôi số không chín không một, hai ba bốn, năm sáu bảy bảy nhé"
    assert _fix("phone.call", {"phone_number": "0912345677"}, said) == {"phone_number": "09012345677"}
    assert _fix("phone.call", {"phone_number": "0911111111"}, said, seen={"0911111111"}) == {"phone_number": "0911111111"}
    assert _fix("phone.call", {"contact_id": "c_01"}, said) == {"contact_id": "c_01"}


def test_rel_day_fix_moves_the_date_and_keeps_the_pickup_on_the_anchor_day():
    said = "thứ sáu này hẹn bưu điện lấy gói hàng giúp mình nha. Với cả sáu giờ chiều hôm trước nhắc mình in nhãn"
    assert _fix("reminders.create", {"content": "in nhãn", "time": "2026-10-05T18:00:00+07:00"}, said)["time"] == \
        "2026-10-08T18:00:00+07:00"
    assert _fix("package_tracking.schedule_pickup", {"address": "x", "date": "2026-10-06"}, said)["date"] == "2026-10-09"


def test_rel_day_fix_leaves_values_with_a_source():
    said = "thứ sáu này hẹn bưu điện. Với cả sáu giờ chiều hôm trước nhắc mình in nhãn"
    v = "2026-10-05T18:00:00+07:00"
    assert _fix("reminders.create", {"content": "x", "time": v}, said, seen={v})["time"] == v


def test_now_offset_fix():
    said = "45 phút nữa nhắc chị gọi lại cho khách nhé."
    assert _fix("reminders.create", {"content": "x", "time": "2026-10-05T10:00:00+07:00"}, said)["time"] == \
        "2026-10-05T09:45:00+07:00"
    said2 = "45 phút nữa, à không, lúc mười giờ nhắc chị"  # có giờ nói ra: không sửa
    assert _fix("reminders.create", {"content": "x", "time": "2026-10-05T10:00:00+07:00"}, said2)["time"] == \
        "2026-10-05T10:00:00+07:00"


def test_new_rules_change_no_gold_call_of_train():
    """Bật cả ba luật mới cùng các luật của config 98, chạy call gold của 1.545 mẫu: call và state phải y nguyên."""
    from lifemate import data
    from lifemate.guards import GuardedEnv
    from lifemate.runner import gold_hash
    _specs()
    rules = dict(clock_fix=True, money_k=True, span_min=True, ask_missing=True, pick_one=True,
                 phone_fix=True, rel_day_fix=True, now_offset_fix=True)
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


@pytest.mark.parametrize('text, want', [
    ('gọi số 19001234', ['19001234']),
    ('gọi số một chín không không, một hai ba bốn', ['19001234']),
    ('gọi số 18001234', ['18001234']),
    ('mã đơn 19001234', []),
    ('gọi số 0901.234.567', ['0901234567']),
])
def test_phone_hotlines_and_grouped_digits(text, want):
    assert Z.phones(text) == want


def test_short_phone_suffix_with_copula_and_correction():
    assert Z.phone_suffixes('đuôi là sáu tám chứ không phải tám sáu') == ['68']
    assert Z.phone_suffixes('đuôi là sáu') == []


@pytest.mark.parametrize('text, want', [
    ('một tiếng mười lăm phút nữa nhắc tôi', '2026-10-05T10:15:00+07:00'),
    ('hai tiếng và ba mươi phút nữa nhắc tôi', '2026-10-05T11:30:00+07:00'),
    ('nửa giờ nữa nhắc tôi', '2026-10-05T09:30:00+07:00'),
    ('tiếng rưỡi nữa nhắc tôi', '2026-10-05T10:30:00+07:00'),
])
def test_compound_and_half_hour_offsets(text, want):
    assert Z.now_offsets(text, NOW) == [want]
    assert Z.durations(text) == []


def test_offset_keeps_seconds_from_current_time():
    assert Z.now_offsets('một phút nữa nhắc tôi', '2026-10-05T09:00:45+07:00') == [
        '2026-10-05T09:01:45+07:00']


def test_interval_and_compound_duration():
    assert Z.durations('học một tiếng mười lăm phút') == [75]
    assert Z.durations('học từ hai rưỡi đến ba giờ mười lăm chiều') == [45]
    assert Z.durations('học từ chín giờ, à nhầm, chín rưỡi đến mười giờ hai mươi') == [50]


@pytest.mark.parametrize('text, day', [
    ('hôm nay gặp cô Mai lúc chín giờ sáng', '2026-10-05'),
    ('hôm nay nhắc tôi chín giờ sáng, không thì thôi', '2026-10-05'),
    ('mai chín giờ sáng nhắc tôi, không thì chuyển tiền', '2026-10-06'),
])
def test_names_and_conditionals_do_not_drop_valid_datetime(text, day):
    assert [iso for iso, _ in Z.datetimes(text, NOW)] == [f'{day}T09:00:00+07:00']


def test_now_offset_fix_preserves_event_duration():
    from lifemate import fixes as F
    args = {'start': '2026-10-05T10:00:00+07:00', 'end': '2026-10-05T10:30:00+07:00'}
    said = '45 phút nữa tạo cuộc họp kéo dài nửa tiếng'
    out, _ = F._now_offset_fix(args, said, NOW, set())
    assert out == {'start': '2026-10-05T09:45:00+07:00', 'end': '2026-10-05T10:15:00+07:00'}
    out, _ = F._now_offset_fix(args, said, NOW, {args['end']})
    assert out['end'] == args['end']


def test_hiding_alternatives_keeps_previous_user_turns():
    from lifemate.agent.agents import LLMAgent
    agent = LLMAgent.__new__(LLMAgent)  # prompt only; no model/API initialization
    agent.opts = {'history': 'full'}
    agent.show_alts = False
    agent.normalize_hint = True
    transcript = {'final': 'ghi lịch hai giờ chiều hôm đó',
                  'turns': {'u1': 'thứ sáu tuần sau có hàng'},
                  'alts': {'final': ['ngày mai lúc ba giờ chiều']}}
    prompt = agent.user_prompt({'context': {'current_time': NOW}}, transcript)
    assert '2026-10-16T14:00:00+07:00' in prompt
    assert '2026-10-06T15:00:00+07:00' not in prompt


def test_empty_asr_history_does_not_fall_back_to_gold():
    from lifemate.guards import GuardedEnv
    _specs()
    sample = {'context': {'current_time': NOW, 'installed_apps': {}},
              'history': [{'role': 'user', 'audio_text': 'thứ sáu tuần sau có hàng'}]}
    g = GuardedEnv(None, 'train', sample, 'ghi lịch hai giờ chiều hôm đó', turns={})
    assert g.past_text == ''


@pytest.mark.parametrize('text, want', [
    ('đổi ba trăm đô Mỹ', [(300, 'đô')]),
    ('đổi năm trăm USD', [(500, 'usd')]),
    ('đổi tám mươi nghìn yên', [(80000, 'yên')]),
    ('chuyển ba trăm', []),
    ('ba trăm người', []),
])
def test_explicit_currency_allows_small_amounts(text, want):
    assert Z.amounts(text) == want


@pytest.mark.parametrize('said', [
    '45 phút nữa nhắc tôi',
    'thứ sáu này hẹn bưu điện, sáu giờ chiều hôm trước nhắc tôi',
])
def test_new_datetime_rules_leave_iso_shaped_free_text_alone(said):
    from lifemate import fixes as F
    value = '2026-10-05T18:00:00+07:00'
    args = {'content': value, 'time': value}
    out, _ = F.fix_args('reminders.create', args, _specs()['reminders.create'], said, NOW, set(),
                        {'clock_fix': True, 'rel_day_fix': True, 'now_offset_fix': True})
    assert out['content'] == value
    assert out['time'] != value


def test_fractional_currency_is_not_truncated():
    assert Z.amounts('đổi 1,5 đô') == [(1.5, 'đô')]


def test_phone_fix_does_not_choose_an_explicitly_rejected_number():
    assert _fix('phone.call', {'phone_number': '0911111111'},
                'đừng gọi số 0901234567, gọi lại số lạ vừa gọi nhỡ') == {'phone_number': '0911111111'}


def test_relative_date_does_not_replace_an_explicit_base_for_another_action():
    said = 'thứ sáu này đặt lấy hàng, trước đó một ngày nhắc tôi'
    assert _fix('package_tracking.schedule_pickup', {'address': 'x', 'date': '2026-10-09'}, said)['date'] == '2026-10-09'


# ------------------------------------------------------------------ cách nói từ bộ ca LM-31 (Antigravity) và vùng miền

@pytest.mark.parametrize("text, want", [
    ("gọi số không chín một hai ba mươi tư năm mươi sáu bảy mươi tám", ["0912345678"]),  # đọc theo cặp
    ("số không chín không năm một linh tám hai linh tư", ["0905108204"]),
    ("số không chín không ba bốn lẻ năm sáu lẻ bảy", ["0903405607"]),
    ("số không tám sáu tám một trăm hai mươi ba bốn trăm năm mươi sáu", ["0868123456"]),
    ("gọi số không chín một hai kép tám ba tư năm sáu", ["0912883456"]),
    ("gọi số không chín tám tám tứ quý chín không một", ["0988999901"]),
    ("gọi cộng tám tư chín một hai ba bốn năm sáu bảy tám", ["0912345678"]),
    ("gọi số không chín bẩy tư năm nhăm sáu sáu bẩy bẩy", ["0974556677"]),
    ("nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha", ["09505360367"]),
    ("gọi số không chín không một hai ba bốn năm sáu bảy, mười phút nữa gọi lại", ["0901234567"]),  # "mười" sau dấu phẩy
])
def test_phones_grouped_and_dialect(text, want):
    assert Z.phones(text) == want


@pytest.mark.parametrize("text, want", [
    ("hai mốt tháng mười", ["2026-10-21"]),  # không có "ngày" đứng trước
    ("hăm mốt tháng mười", ["2026-10-21"]),
    ("camera hôm kia báo mất kết nối", ["2026-10-03"]),
    ("trong hai tháng tới", []),
])
def test_spoken_dates_extra(text, want):
    assert Z.spoken_dates(text, NOW) == want


def test_midnight_g_clock_and_same_clause_anchor():
    assert [iso for iso, _ in Z.datetimes("nhắc tôi lúc nửa đêm", NOW)] == ["2026-10-06T00:00:00+07:00"]
    assert [iso for iso, _ in Z.datetimes("mười hai giờ đêm nay nhắc anh", NOW)] == ["2026-10-06T00:00:00+07:00"]
    assert [iso for iso, _ in Z.datetimes("6g15 sáng mai", NOW)] == ["2026-10-06T06:15:00+07:00"]
    (a,) = Z.anchored("trước thứ sáu tuần này một ngày nhắc chị", NOW)
    assert (a.day, a.base) == ("2026-10-08", "2026-10-09")


def test_decimal_percent_hăm_and_chục():
    assert Z.percents("hai phẩy năm phần trăm") == [2.5]
    assert Z.percents("2,5 phần trăm") == [2.5]
    assert Z.percents("không chấm tám phần trăm") == [0.8]
    assert Z.amounts("hăm nhăm nghìn") == [(25000, "")]
    assert Z.now_offsets("chục phút nữa", NOW) == ["2026-10-05T09:10:00+07:00"]


def test_slang_units_are_not_money():
    """Tiếng lóng tiền ("xị", "lít", "chai") không đọc: train và public không có lần nào, còn "lít", "chai" là đơn vị
    thật ("mười lăm lít xăng", "chai nước")."""
    assert Z.amounts("mười lăm lít xăng") == []
    assert Z.amounts("năm xị") == []


# ------------------------------------------------------------------ hồi quy từ dự đoán public của 98z (09/11)

def test_garbled_number_is_not_read():
    """public_0232: ASR "hai lăm lăm phút nữa" từng đọc ra 5 phút; cụm số đọc sai luật thì không đọc."""
    assert Z.now_offsets("rồi hai lăm lăm phút nữa nhắc chị đi", NOW) == []
    assert Z.now_offsets("hai lăm phút nữa nhắc chị", NOW) == ["2026-10-05T09:25:00+07:00"]


def test_day_before_a_data_entity_is_not_anchored_on_a_spoken_date():
    """public_0542: "năm giờ chiều hôm trước ngày bay" có mốc là ngày bay trong dữ liệu; không được neo vào ngày nói ở
    lượt trước. Tương tự "trước hạn khiếu nại một ngày"."""
    assert Z.anchored("hủy vé máy bay, rồi năm giờ chiều hôm trước ngày bay nhắc chị", NOW,
                      past="ngày hai hai tháng mười chị bay") == []
    assert Z.anchored("thứ sáu này đi khám, nhắc chị trước hạn nộp hồ sơ một ngày", NOW) == []
    v = "2026-11-05T17:00:00+07:00"
    assert _fix("reminders.create", {"content": "gọi", "time": v},
                "hủy vé máy bay, rồi năm giờ chiều hôm trước ngày bay nhắc chị", past="ngày hai hai tháng mười")["time"] == v


def test_rel_day_fix_ignores_an_anchor_inside_the_offset_phrase():
    """train_1353: "trước bữa Tí bay một ngày" nghe thành "trước bữa thứ bảy một ngày"; mốc trong cụm dời chỉ dùng cho
    gợi ý, luật sửa không dùng."""
    (a,) = Z.anchored("hẹn bưu tá lấy thùng bưởi, trước bữa thứ bảy một ngày nha", NOW)
    assert a.inner
    assert _fix("package_tracking.schedule_pickup", {"address": "x", "date": "2026-10-29"},
                "hẹn bưu tá lấy thùng bưởi, trước bữa thứ bảy một ngày nha")["date"] == "2026-10-29"
