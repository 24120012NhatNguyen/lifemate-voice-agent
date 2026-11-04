"""Máy tính ẩn sys.calc và bộ đọc số, giờ, thời lượng trong lời nói tiếng Việt."""
import pytest

from lifemate.numbers import calc, clock_times, durations, numbers, weekday_dates


@pytest.mark.parametrize("expr, want", [
    ("8000+1500", 9500), ("15*22800", 342000), ("200000*15%", 30000), ("(120000+85000)*3", 615000),
    ("2026-10-08T10:00:00+07:00 - 7m", "2026-10-08T09:53:00+07:00"), ("2026-10-28 - 1d", "2026-10-27"),
    ("21:45 - 21:00", 45), ("20:15 + 15m", "20:30"), ("2*30m", 60),
    ("2026-12-01T05:30:00+07:00 - 2026-11-30T21:35:00+07:00", 475),
])
def test_calc(expr, want):
    assert calc(expr) == want


def test_calc_rejects_code():
    assert str(calc("__import__('os').system('true')")).startswith("lỗi")


@pytest.mark.parametrize("text, want", [
    ("thêm ngàn rưỡi bước", [1500]), ("hai triệu tám", [2800000]), ("ba trăm hai mươi lăm nghìn", [325000]),
    ("50.000 đồng", [50000]), ("bốn lăm phần trăm", [45, 100]), ("mười lăm lít", [15]),
    ("thứ bảy tuần sau", []), ("năm nay em làm", []), ("tháng mười hai", []),
])
def test_numbers(text, want):
    assert numbers(text) == want


def test_clock_times_and_durations():
    assert clock_times("từ chín giờ tới mười giờ kém mười lăm") == ["09:00", "21:00", "09:45", "21:45"]
    assert clock_times("tám giờ tối nay") == ["20:00"]
    assert clock_times("lúc sáu giờ rưỡi") == ["06:30", "18:30"]
    assert durations("đón sớm nửa tiếng, nhắc trước mười lăm phút, trước một ngày") == [30, 15, 1440]
    assert durations("hai tiếng rưỡi") == [150]


def test_weekday_dates():
    dates = ["2026-10-26", "2026-10-27", "2026-10-25"] + [f"2026-10-{d}" for d in range(28, 32)] + ["2026-11-01", "2026-11-02"]
    assert weekday_dates("Thứ Năm này con bay", dates) == ["2026-10-29"]
    assert weekday_dates("chủ nhật", dates) == ["2026-11-01"]
