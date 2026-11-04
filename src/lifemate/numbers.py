"""Bộ giải số tiếng Việt và máy tính ẩn `sys.calc` của agent (nấc 2, docs/PROPOSAL.md mục 3.B).

LLM 4B tính nhẩm hay sai: 10 giờ trừ 7 phút đi đường, 15 lít nhân giá xăng, mục tiêu cũ cộng ngàn rưỡi bước. Agent có
`calc: true` được gọi `sys.calc` với một biểu thức; agent tự tính, không gửi call này cho môi trường nên không ảnh hưởng
điểm. `scripts/build_sft.py` với config có `calc: true` chèn bước `sys.calc` vào quỹ đạo gold ở chỗ giá trị gold tính
được từ các số đã có (`derive`).

    calc("8000+1500")                          → 9500
    calc("2026-10-08T10:00:00+07:00 - 7m")     → "2026-10-08T09:53:00+07:00"
    calc("2026-10-28 - 1d")                    → "2026-10-27"
    calc("21:45 - 21:00")                      → 45   (phút)
"""
from __future__ import annotations

import ast
import operator
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone


# ---------------------------------------------------------------- máy tính

_DT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2})?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?!T)")
_HM = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?![\d:])")
_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(m|h|d|w)\b")
_UNIT_MIN = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_VN = timezone(timedelta(hours=7))  # mọi mẫu đều ở Asia/Ho_Chi_Minh


class _HMTime:
    """Giờ trong ngày không kèm ngày: hiệu hai giờ ra số phút, giờ cộng trừ thời lượng ra giờ."""

    def __init__(self, s: str):
        h, m = map(int, s.split(":"))
        self.minutes = h * 60 + m

    def __sub__(self, other):
        if isinstance(other, _HMTime):
            return timedelta(minutes=self.minutes - other.minutes)
        return _HMTime.of((self.minutes - other.total_seconds() // 60) % 1440)

    def __add__(self, other):
        return _HMTime.of((self.minutes + other.total_seconds() // 60) % 1440)

    __radd__ = __add__

    @staticmethod
    def of(minutes) -> "_HMTime":
        t = _HMTime("0:00")
        t.minutes = int(minutes)
        return t

    def __str__(self):
        return f"{self.minutes // 60:02d}:{self.minutes % 60:02d}"


def _aware(t: datetime) -> datetime:
    return t.replace(tzinfo=_VN) if t.tzinfo is None else t.astimezone(_VN)


def calc(expr: str):
    """Tính biểu thức số, ngày giờ ISO, giờ HH:MM và thời lượng 7m, 2h, 1d, 1w. Lỗi thì trả chuỗi bắt đầu bằng "lỗi"."""
    lits: list = []

    def keep(v) -> str:
        lits.append(v)
        return f"_v{len(lits) - 1}"

    s = str(expr).replace("×", "*").replace("÷", "/").replace(",", "")
    s = _DT.sub(lambda m: keep(_aware(datetime.fromisoformat(m.group(0)))), s)
    s = _DATE.sub(lambda m: keep(date.fromisoformat(m.group(0))), s)
    s = _HM.sub(lambda m: keep(_HMTime(m.group(0))), s)
    s = _DUR.sub(lambda m: keep(timedelta(minutes=float(m.group(1)) * _UNIT_MIN[m.group(2)])), s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return f"lỗi: không đọc được biểu thức {expr!r}"

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name) and re.fullmatch(r"_v\d+", n.id):
            return lits[int(n.id[2:])]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(a, (int, float)) and isinstance(b, timedelta):  # 2*30m
                a, b = b, a
            return _OPS[type(n.op)](a, b)
        raise ValueError(ast.dump(n))

    try:
        v = ev(tree)
    except Exception as e:  # noqa: BLE001 — biểu thức do LLM viết, lỗi nào cũng trả về cho nó đọc
        return f"lỗi: {type(e).__name__} trong {expr!r}"
    if isinstance(v, timedelta):
        return int(v.total_seconds() // 60)
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (date, _HMTime)):
        return v.isoformat() if isinstance(v, date) else str(v)
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 4)
    return v


# ---------------------------------------------------------------- số trong lời nói
# Đọc trên chữ thường còn dấu: bỏ dấu thì "làm" trùng "lăm" (5), "bà" trùng "ba" (3).

_UNIT = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5,
         "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
_SCALE = {"nghìn": 1000, "ngàn": 1000, "ngìn": 1000, "k": 1000, "triệu": 10**6, "củ": 10**6, "tỷ": 10**9, "tỉ": 10**9}
_NUMWORD = set(_UNIT) | set(_SCALE) | {"mười", "mươi", "chục", "trăm", "linh", "lẻ", "rưỡi"}
_NOT_AFTER = {"thứ", "tháng", "mùng", "lần"}  # "thứ bảy", "tháng mười": tên, không phải lượng
_NOT_BEFORE = {"nay", "ngoái", "sau", "trước", "rồi", "mới", "học"}  # "năm nay", "năm học"


def _words(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", str(text).lower())
    return re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+", t)


def _digits(tok: str) -> float | None:
    if not tok[:1].isdigit():
        return None
    t = tok.replace(".", "") if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok) else tok.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _isnum(w: str) -> bool:
    return w in _NUMWORD or _digits(w) is not None


def _value(ws: list[str]) -> float | None:
    """Giá trị một cụm từ số liền nhau: "hai triệu tám" 2.800.000, "ngàn rưỡi" 1.500, "ba trăm hai mươi lăm" 325."""
    total, chunk, cur, prev = 0.0, 0.0, None, ""
    for w in ws:
        d = _digits(w)
        if d is not None:
            cur = d
        elif w in _UNIT:
            if prev in ("mười", "mươi"):
                chunk += _UNIT[w]
            elif w in ("lăm", "mốt", "tư", "nhăm") and prev in _UNIT and cur is not None and cur < 10:
                cur = cur * 10 + _UNIT[w]  # "bốn lăm" = 45, "ba mốt" = 31, "hai tư" = 24
            else:
                cur = _UNIT[w]
        elif w == "mười":
            chunk += 10
        elif w in ("mươi", "chục"):
            chunk += (cur if cur is not None else 1) * 10
            cur = None
        elif w == "trăm":
            chunk += (cur if cur is not None else 1) * 100
            cur = None
        elif w in _SCALE:
            total += ((chunk + (cur or 0)) or 1) * _SCALE[w]
            chunk, cur = 0.0, None
        elif w == "rưỡi":
            if prev in _SCALE:
                total += _SCALE[prev] / 2
            elif prev == "trăm":
                chunk += 50
            elif cur is not None:
                cur += 0.5
        prev = w
    if cur is not None:
        after_scale = len(ws) >= 2 and (ws[-2] in _SCALE or ws[-2] == "trăm") and cur < 10
        if after_scale:  # "hai triệu tám" = 2.800.000, "ba trăm hai" = 320
            total += cur * (_SCALE.get(ws[-2], 100) / 10)
        else:
            chunk += cur
    v = total + chunk
    return v if any(_digits(w) is not None or w in _UNIT or w in ("mười", "trăm") or w in _SCALE for w in ws) else None


def _spans(ws: list[str]):
    """Các cụm từ số liền nhau (i, j), bỏ cụm là tên ("thứ bảy") hoặc "năm" chỉ năm ("năm nay")."""
    i = 0
    while i < len(ws):
        j = i
        while j < len(ws) and _isnum(ws[j]):
            j += 1
        if j == i:
            i += 1
            continue
        named = i and ws[i - 1] in _NOT_AFTER
        year = ws[i:j] == ["năm"] and (j >= len(ws) or ws[j] in _NOT_BEFORE)
        if not named and not year and ws[i:j] != ["rưỡi"]:
            yield i, j
        i = j


def numbers(text: str) -> list[float]:
    """Mọi lượng số trong lời nói, cả chữ số lẫn số đọc bằng chữ."""
    ws = _words(text)
    return [v for i, j in _spans(ws) if (v := _value(ws[i:j])) is not None]


_PM = {"chiều", "tối", "đêm"}
_DAYPART = {"sáng", "trưa"} | _PM


def clock_times(text: str) -> list[str]:
    """Các giờ trong ngày nói ra, dạng "HH:MM". Không rõ sáng hay chiều thì cho cả hai: "bảy rưỡi" ra 07:30 và 19:30."""
    ws = _words(text)
    out = []
    for i, j in _spans(ws):
        if j >= len(ws) and ws[j - 1] != "rưỡi":
            continue
        nxt = ws[j] if j < len(ws) else ""
        if nxt not in ("giờ", "h") and ws[j - 1] != "rưỡi":
            continue
        h = _value(ws[i:j] if ws[j - 1] != "rưỡi" else ws[i:j - 1])
        if h is None or not 0 <= h <= 24 or h != int(h):
            continue
        m, k = (30, j) if ws[j - 1] == "rưỡi" else (0, j + 1)
        rest = ws[k:k + 4]
        if m == 0 and rest[:1] == ["rưỡi"]:
            m = 30
        elif m == 0 and rest[:1] == ["kém"]:
            n = numbers(" ".join(rest[1:4]))
            if n and 0 < n[0] < 60:
                h, m = h - 1, 60 - n[0]
        elif m == 0 and rest:
            q = 0
            while q < len(rest) and _isnum(rest[q]) and rest[q] != "rưỡi":
                q += 1
            n = _value(rest[:q]) if q else None
            if n is not None and 0 < n < 60 and not (q < len(rest) and rest[q] in ("tiếng", "ngày", "giờ")):
                m = n
        part = next((x for x in ws[j:j + 6] if x in _DAYPART), None)
        h = int(h)
        hs = [h + 12] if part in _PM and h < 12 else [h] if part or h == 0 or h >= 12 else [h, h + 12]
        out += [f"{x % 24:02d}:{int(m):02d}" for x in hs]
    for hh, mm in re.findall(r"\b(\d{1,2})[:h](\d{2})\b", str(text)):
        out.append(f"{int(hh):02d}:{mm}")
    return list(dict.fromkeys(out))


def durations(text: str) -> list[int]:
    """Thời lượng nói ra, bằng phút: "nửa tiếng" 30, "hai tiếng rưỡi" 150, "mười lăm phút" 15, "một ngày" 1440."""
    ws = _words(text)
    unit = {"phút": 1, "tiếng": 60, "ngày": 1440, "tuần": 10080}
    out = []
    for i, w in enumerate(ws):
        if w not in unit:
            continue
        k = i
        while k > 0 and _isnum(ws[k - 1]) and ws[k - 1] != "rưỡi":
            k -= 1
        n = _value(ws[k:i]) if k < i else (0.5 if i and ws[i - 1] == "nửa" else None)
        if n is None:
            continue
        if i + 1 < len(ws) and ws[i + 1] == "rưỡi":
            n += 0.5
        out.append(int(n * unit[w]))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- tìm phép tính cho giá trị gold

_ISO_DT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2})?$")
_ISO_D = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HHMM = re.compile(r"^\d{1,2}:\d{2}$")


def _leaves(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v, key)
    else:
        yield key, obj


def _num(x) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _dur(m: int) -> str:
    return f"{m // 1440}d" if m and m % 1440 == 0 else f"{m}m"


_WEEKDAY = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2, "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5}


def weekday_dates(text: str, dates: list[str]) -> list[str]:
    """Ngày ứng với "thứ Năm này", "chủ nhật", "thứ 7 tuần sau", lấy trong các ngày của bảng thời gian (hôm nay trước)."""
    ws = _words(text)
    days = sorted(dates)
    today = dates[0]
    out = []
    for i, w in enumerate(ws):
        wd = _WEEKDAY.get(ws[i + 1]) if w == "thứ" and i + 1 < len(ws) else 6 if w == "chủ" and ws[i + 1:i + 2] == ["nhật"] else None
        if wd is None:
            continue
        nxt = ws[i + 2:i + 5]
        cands = [d for d in days if d >= today and date.fromisoformat(d).weekday() == wd]
        if cands:
            out.append(cands[1] if "sau" in nxt and len(cands) > 1 and nxt[:2] == ["tuần", "sau"] else cands[0])
    return list(dict.fromkeys(out))


def derive(value, said: str, sources: list, now: str, dates: list[str]) -> str | None:
    """Biểu thức `sys.calc` một bước cho ra `value` từ các số đã có, hoặc None nếu giá trị có sẵn hay không tìm được.

    said: lời người dùng; sources: ngữ cảnh, lịch sử, rồi kết quả tool và tham số các call ghi trước đó theo thứ tự, nguồn
    gần nhất được thử trước; dates: các ngày của bảng thời gian, hôm nay trước.
    Chỉ nhận phép tính có ít nhất một toán hạng từ lời nói, để không dạy những trùng hợp giữa các số trong kết quả tool."""
    leaves = [(k, v) for src in reversed(sources) for k, v in _leaves(src)]  # nguồn gần nhất trước
    r_nums = list(dict.fromkeys(float(v) for _, v in leaves if isinstance(v, (int, float)) and not isinstance(v, bool)))
    t_nums = list(dict.fromkeys(numbers(said)))
    texts = [v for _, v in leaves if isinstance(v, str) and len(v) > 12 and not _ISO_DT.match(v)]
    clocks = clock_times(said)
    r_clocks = [c for t in texts for c in clock_times(t) if c not in clocks]  # giờ nhắc trong lịch sử, ghi chú, tin nhắn
    r_days = [d for t in [said] + texts for d in weekday_dates(t, dates)]
    durs = durations(said)
    r_dts = list(dict.fromkeys(v for _, v in leaves if isinstance(v, str) and (_ISO_DT.match(v) or _ISO_D.match(v))))
    r_durs = [int(v) for k, v in leaves if isinstance(v, (int, float)) and not isinstance(v, bool)
              and re.search(r"min|duration|eta|phut", k) and 0 < v < 1440]

    def first(cands):
        return next((e for e in cands if calc(e) == value), None)

    ws = set(_words(said))
    early, late = bool(ws & {"trước", "sớm"}), bool(ws & {"sau", "muộn", "trễ", "lùi", "thêm"})
    signs = "-" if early and not late else "+" if late and not early else "-+"

    def shifted(bases, steps):
        return (f"{b} {sign} {_dur(d)}" for b in bases for d in steps for sign in signs)

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v in t_nums or v in r_nums or abs(v) < 2:
            return None
        e = first(f"{b} - {a}" for a in clocks for b in clocks if a != b)  # số phút giữa hai giờ nói ra
        if e:
            return e
        pool = [(x, True) for x in t_nums] + [(x, False) for x in r_nums]
        pairs = [(p, q) for p in pool for q in pool if p[1] or q[1]]
        pairs.sort(key=lambda pq: pq[0][1] == pq[1][1])  # một số nói ra với một số trong kết quả trước, hai số nói ra sau
        for (a, ta), (b, tb) in pairs:
            for op, f in (("+", a + b), ("-", a - b), ("*", a * b), ("/", a / b if b else None)):
                if f is not None and abs(f - v) < 1e-6 and (op in "+*" and a <= b or op in "-/"):
                    return f"{_num(a)}{op}{_num(b)}"
        targets = [x for x in r_dts if _ISO_DT.match(x)] + [f"{d}T{t}:00+07:00" for d in dates[:3] for t in clocks + r_clocks]
        return first(f"{t} - {now}" for t in targets)  # hẹn sau bao nhiêu phút kể từ bây giờ
    if not isinstance(value, str):
        return None
    if _ISO_DT.match(value):
        if value in r_dts:
            return None
        r_bases = [x for x in r_dts if _ISO_DT.match(x)] + [f"{x}T{t}:00+07:00" for x in r_dts if _ISO_D.match(x) for t in clocks]
        e = first(shifted(r_bases, durs))  # mốc trong kết quả tool, dời theo thời lượng nói ra
        if e or any(value.startswith(f"{d}T{t}") for d in dates for t in clocks):  # ngày trong bảng, giờ nói ra
            return e
        return first(shifted(r_bases, r_durs)) or first(shifted([f"{d}T{t}:00+07:00" for d in dates for t in clocks], durs + r_durs))
    if _ISO_D.match(value):
        if value in r_dts:
            return None
        return first(shifted(list(dict.fromkeys([x[:10] for x in r_dts] + r_days)), [d for d in durs if d % 1440 == 0]))
    if _HHMM.match(value):
        if value in clocks or any(x[11:16] == value for x in r_dts if _ISO_DT.match(x)):
            return None
        return first(shifted([x[11:16] for x in r_dts if _ISO_DT.match(x)] + clocks, durs + r_durs))
    return None
