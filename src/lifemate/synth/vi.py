"""Lời nói tiếng Việt cho mẫu tổng hợp: số, tiền, giờ, ngày, và giá trị gold tương ứng.

Ngày chỉ lấy trong cửa sổ hôm qua đến 13 ngày tới, đúng các ngày bảng thời gian `two_weeks` của prompt liệt kê, và gọi
theo đúng nhãn của bảng (`context.timetable`): "mai", "ngày mốt", "thứ Bảy tuần này", "thứ Hai tuần sau"...
"""
from __future__ import annotations

import random
import re
from datetime import datetime, timedelta

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
WEEKDAYS = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ Nhật"]


def words(n: int) -> str:
    """Số 0-99 bằng chữ: 15 → "mười lăm", 21 → "hai mươi mốt"."""
    if n < 10:
        return DIGITS[n]
    t, u = divmod(n, 10)
    if n < 20:
        return "mười" + ("" if u == 0 else " " + ("lăm" if u == 5 else DIGITS[u]))
    if n < 100:
        return f"{DIGITS[t]} mươi" + ("" if u == 0 else " " + {1: "mốt", 5: "lăm"}.get(u, DIGITS[u]))
    return str(n)


def num(n: int, rng: random.Random, p_words: float = 0.3) -> str:
    return words(n) if n < 100 and rng.random() < p_words else str(n)


def money(v: int, rng: random.Random) -> str:
    """Số tiền VND như người ta nói: "350 nghìn", "2 triệu", "1 triệu rưỡi", "1 triệu 2"."""
    m, r = divmod(v, 1_000_000)
    if m and not r:
        return f"{num(m, rng)} triệu"
    if m and r == 500_000:
        return f"{num(m, rng)} triệu rưỡi"
    if m and r % 100_000 == 0:
        return rng.choice([f"{m} triệu {r // 100_000}", f"{m} triệu {r // 100_000} trăm nghìn"])
    return f"{v // 1000} {rng.choice(['nghìn', 'ngàn'])}"


def money_value(rng: random.Random, lo: int, hi: int) -> int:
    """Một số tiền nói tròn được: dưới 1 triệu là bội 10 nghìn, từ 1 triệu là bội 100 nghìn."""
    v = rng.randint(lo, hi)
    return round(v, -4) if v < 1_000_000 else round(v, -5)


def part_of_day(h: int) -> str:
    return "sáng" if h < 11 else "trưa" if h < 13 else "chiều" if h < 18 else "tối"


def clock(h: int, m: int, rng: random.Random) -> str:
    """Giờ nói kèm buổi: "9 giờ sáng", "3 rưỡi chiều", "8 giờ kém 15 tối"."""
    h12 = h if h <= 12 else h - 12
    if m == 45 and rng.random() < 0.5:
        nh = h + 1
        return f"{num(nh if nh <= 12 else nh - 12, rng)} giờ kém 15 {part_of_day(nh)}"
    if m == 0:
        core = f"{num(h12, rng)} giờ"
    elif m == 30:
        core = rng.choice([f"{num(h12, rng)} rưỡi", f"{num(h12, rng)} giờ rưỡi", f"{h12} giờ 30"])
    else:
        core = f"{num(h12, rng)} giờ {num(m, rng)}"
    return f"{core} {part_of_day(h)}"


def day(now: datetime, k: int, rng: random.Random) -> str:
    """Ngày hôm nay + k (−1 ≤ k ≤ 13) theo nhãn của bảng thời gian."""
    d0 = now.date()
    d = d0 + timedelta(days=k)
    monday = d0 - timedelta(days=d0.weekday())
    opts = {0: ["hôm nay"], 1: ["mai", "ngày mai"], 2: ["ngày mốt", "ngày kia"], -1: ["hôm qua"]}.get(k, [])
    if k >= 2:
        week = "tuần này" if d < monday + timedelta(days=7) else "tuần sau" if d < monday + timedelta(days=14) else None
        if week:
            opts.append(f"{WEEKDAYS[d.weekday()]} {week}")
    if k >= 1:
        opts.append(f"mùng {d.day}" if d.day <= 10 else f"ngày {d.day}")
    return rng.choice(opts)


def when(now: datetime, k: int, h: int, m: int, rng: random.Random) -> str:
    """Ngày giờ nói: "9 giờ sáng mai", "7 giờ tối nay", "thứ Bảy tuần này lúc 3 rưỡi chiều"."""
    c = clock(h, m, rng)
    if k == 0 and rng.random() < 0.6:
        return f"{c} nay"
    if k == 1 and rng.random() < 0.6:
        return f"{c} mai"
    d = day(now, k, rng)
    return rng.choice([f"{c} {d}", f"{d} lúc {c}"])


def iso_date(now: datetime, k: int) -> str:
    return (now.date() + timedelta(days=k)).isoformat()


def iso_dt(now: datetime, k: int, h: int, m: int) -> str:
    return f"{iso_date(now, k)}T{h:02d}:{m:02d}:00+07:00"


def pick_dt(now: datetime, rng: random.Random, lo: int = 0, hi: int = 13) -> tuple[int, int, int]:
    """(k, giờ, phút) ở tương lai: hôm nay thì cách bây giờ ít nhất 2 tiếng."""
    for _ in range(20):
        k = rng.randint(lo, hi)
        h = rng.randint(7, 21)
        m = rng.choice([0, 0, 0, 30, 30, 15, 45])
        if k > 0 or h >= now.hour + 2:
            return k, h, m
    return max(lo, 1), 9, 0


def pronoun(profile_name: str, rng: random.Random) -> str:
    """Người dùng tự xưng theo tên hồ sơ ("Chị Hoa" → "chị"); "Bé" xưng "con"."""
    head = (profile_name or "").split(" ")[0].lower()
    p = {"bé": "con"}.get(head, head)
    if p not in {"anh", "chị", "em", "bác", "cô", "chú", "ông", "bà", "con"} or rng.random() < 0.15:
        p = rng.choice(["mình", "tôi", "mình"])
    return p


PREFIX = ["", "", "", "", "Em ơi, ", "Ờ, ", "Ừ, ", "Nè em, ", "À, ", "Ừm, ", "Ê em, ", "Bạn ơi, "]
SUFFIX = ["", "", " nha", " nhé", " giúp {p}", " giùm {p} nha", " với", " đi em", " nghen", " nhen", " luôn nha",
          " giúp {p} với", " cho {p} nhé", " dùm {p}"]
JOIN = [" rồi ", ", xong ", " với ", ", sau đó ", ", rồi ", " và ", ". Rồi "]
FIX = [", à nhầm, ", ", à không, ", ", à không phải, ", ", ờ không, "]
CANCEL = [", à thôi khỏi đi", ", mà thôi, để sau tính", ", à thôi, không cần nữa", ". Mà thôi bỏ đi, không cần đâu",
          ", à thôi khoan, để {p} tính lại đã"]
THANKS = ["Cảm ơn em nha.", "Ừ được rồi, cảm ơn em.", "Ok vậy được rồi.", "Hay quá, cảm ơn nhé.", "Vậy là xong rồi ha, cảm ơn em.",
          "Ừ, vậy thôi nhé.", "Được rồi, không cần gì thêm đâu."]


def utter(core: str, p: str, rng: random.Random, suffix: bool = True) -> str:
    """Lời nói hoàn chỉnh: tiền tố, phần chính, tiểu từ cuối, viết hoa chữ đầu và chấm câu."""
    tail = ""
    if suffix and not re.search(r"\b(nha|nhé|nghen|nhen|đi|với|luôn|thôi)$", core.strip()):  # không lặp tiểu từ
        opts = [x for x in SUFFIX if not (re.search(r"giúp|giùm|dùm|\{p\}", core) and re.search(r"giúp|giùm|dùm|\{p\}", x))]
        opts = [x for x in opts if not (re.search(r"\bđi\b|nữa", core) and "đi" in x)]
        tail = rng.choice(opts)
    s = rng.choice(PREFIX) + core + tail
    s = s.replace("{p}", p).strip()
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".?!" else s + ("?" if s.endswith("được không") else ".")


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def low(s: str) -> str:
    """Chữ đầu viết thường, trừ khi cả từ đầu viết hoa (tên riêng kiểu "SH", "VF 5")."""
    head = s.split(" ")[0]
    return s if len(head) > 1 and head.isupper() else s[:1].lower() + s[1:]
