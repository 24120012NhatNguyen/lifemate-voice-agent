"""Sửa tất định tham số call ghi trước khi chạy (tầng D): những chỗ LLM 4B hay sai mà luật đọc được từ lời nói.

    spoken_dates("thứ ba tuần sau", now)   ngày nói ra, theo bảng thời gian (tuần tính từ thứ Hai)
    spoken_clocks("9 giờ kém 15 tối mai")  giờ nói ra, "HH:MM"; buổi đứng trước hay sau đều được
    fix_args(name, args, ...)              các luật dưới đây, mỗi luật bật/tắt trong config `guards:`

    clock_fix   ngày giờ ISO lệch giờ nói ra: sai sáng/chiều, hoặc lời nói chỉ có một giờ, một ngày mà call ghi khác
    money_k     số tiền VND dưới 1.000 là nói tắt hoặc ASR mất chữ "nghìn": "hai trăm" là 200.000
    span_min    tham số phút khi lời nói có "từ 7 rưỡi tới 8 giờ 10": hiệu hai giờ
    missing     ask_missing: số tiền khi lời chỉ nói "chút tiền", giờ hẹn khi không ai nói giờ: hỏi lại thay vì bịa
    pick_one    "hủy bớt một gói thôi" khi có nhiều gói: hỏi lại mục nào

Ba luật dưới đây đọc giá trị bằng tầng N (normalize.py):

    phone_fix       số điện thoại đọc từng chữ số ("không chín không một, hai ba bốn…"): tham số số điện thoại
                    lệch số nói ra duy nhất thì lấy số nói ra
    rel_day_fix     "sáu giờ chiều hôm trước" khi mốc là "thứ sáu này": ngày ghép với giờ trong cùng mệnh đề.
                    clock_fix tắt khi lời có "hôm trước / hôm đó"; luật này lo đúng chỗ đó
    now_offset_fix  "bốn lăm phút nữa": bây giờ + thời lượng, khi lời không nói giờ nào khác

Giá trị đã có trong lịch sử hay kết quả tool, hoặc tính được một bước từ đó (numbers.derive), không bao giờ bị sửa.
Bật mọi luật rồi chạy call gold của cả 1.545 mẫu train: không call nào đổi (tests/test_fixes.py).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

from . import numbers as N

_PM = {"chiều", "tối", "đêm"}
_AM = {"sáng", "trưa"}
_TODAY = {"nay", "ni"}
_WD = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2, "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5}


def _ws(text: str) -> list[str]:
    return N._words(text)


def _is_name(text: str, k: int) -> bool:
    """Từ thứ k ("mai") viết hoa giữa câu là tên riêng: "chị Mai", "Mai Anh". Đầu câu thì vẫn là "ngày mai"."""
    t = unicodedata.normalize("NFC", str(text))
    toks = list(re.finditer(r"\d+(?:[.,]\d+)*|[^\W\d_]+", t))
    if k >= len(toks) or not toks[k].group()[:1].isupper():
        return False
    return bool(t[:toks[k].start()].strip()) and t[:toks[k].start()].rstrip()[-1] not in ".!?…:,"


def _day_month(ws: list[str], j: int, d0: date, loose: bool) -> date | None:
    """Ngày "N (tháng M)" bắt đầu ở ws[j]. loose: sau "ngày" / "hôm", nơi "một", "hai", "ba" đứng một mình dễ là số lượng."""
    k = j
    while k < len(ws) and N._isnum(ws[k]) and ws[k] != "rưỡi":
        k += 1
    n = N._value(ws[j:k]) if k > j else None
    if k == j + 2 and ws[j] in ("hai", "ba") and ws[j + 1] in N._UNIT and ws[j + 1] not in ("mười",):
        n = N._UNIT[ws[j]] * 10 + N._UNIT[ws[j + 1]]  # "hôm hai chín" là 29, "ba mốt" là 31
    if k == j + 3 and ws[j] in ("hai", "ba") and ws[j + 1] == "mười" and ws[j + 2] in N._UNIT:
        n = N._UNIT[ws[j]] * 10 + N._UNIT[ws[j + 2]]  # ASR "hai mười ba" cho "hai mươi ba"
    if n is None or n != int(n) or not 1 <= n <= 31 or ws[k:k + 1] in (["giờ"], ["tiếng"], ["ngày"], ["phút"], ["người"]):
        return None
    month = None
    if ws[k:k + 1] == ["tháng"]:
        q = k + 1
        while q < len(ws) and N._isnum(ws[q]):
            q += 1
        m = N._value(ws[k + 1:q]) if q > k + 1 else None
        month = int(m) if m and 1 <= m <= 12 and m == int(m) else None
    if loose and month is None and k == j + 1 and ws[j] in ("một", "hai", "ba"):
        return None
    for y, mo in ([(d0.year, month), (d0.year + 1, month)] if month else
                  [(d0.year, d0.month), (d0.year + (d0.month == 12), d0.month % 12 + 1)]):
        try:
            cand = date(y, mo, int(n))
        except ValueError:
            continue
        if cand >= d0 - timedelta(days=1):
            return cand
    return None


def spoken_dates(text: str, now: str) -> list[str]:
    """Các ngày nói rõ trong lời: hôm nay / tối ni, mai, ngày kia / mốt, hôm qua, thứ X (này / tuần sau / nữa),
    chủ nhật, ngày N (tháng M), mùng N, hôm N. Không gồm ngày tương đối với một mốc khác ("hôm trước", "hôm đó")."""
    ws = _ws(text)
    d0 = datetime.fromisoformat(now).date()
    monday = d0 - timedelta(days=d0.weekday())
    out: list[date] = []
    i = 0
    while i < len(ws):
        w = ws[i]
        prev, nxt = ws[i - 1] if i else "", ws[i + 1] if i + 1 < len(ws) else ""
        if w in _TODAY and prev in _AM | _PM | {"hôm", "bữa", "trưa"}:
            out.append(d0)
        elif w == "mai" and nxt != "mốt" and not _is_name(text, i):
            out.append(d0 + timedelta(days=1))
        elif (w == "kia" and prev == "ngày") or (w == "mốt" and prev in {"ngày", "bữa"} | _AM | _PM):
            out.append(d0 + timedelta(days=2))
        elif w == "qua" and prev in {"hôm", "bữa"} | _AM | _PM:
            out.append(d0 - timedelta(days=1))
        elif (w == "thứ" and nxt in _WD) or (w == "chủ" and nxt == "nhật"):
            wd = 6 if w == "chủ" else _WD[nxt]
            q = ws[i + 2:i + 5]
            j = i + 2 + (q[:1] in (["ngày"], ["mùng"], ["mồng"], ["hôm"]))
            exact = _day_month(ws, j, d0, loose=False) if j < len(ws) else None
            if exact:  # "chủ nhật mười một tháng mười", "thứ ba, hôm hai chín": ngày cụ thể thắng
                out.append(exact)
                i = j + 1
                continue
            if q[:3] == ["tuần", "sau", "nữa"]:
                week = 2
            elif q[:2] in (["tuần", "sau"], ["tuần", "tới"]):
                week = 1
            elif q[:2] == ["tuần", "trước"]:
                week = -1
            elif q[:1] == ["này"] or q[:2] == ["tuần", "này"]:
                week = 0
            else:  # không nói tuần: lần tới gần nhất, kể cả hôm nay
                week = 0 if wd >= d0.weekday() else 1
            out.append(monday + timedelta(days=7 * week + wd))
        elif w in ("ngày", "mùng", "mồng", "hôm") and i + 1 < len(ws):
            exact = _day_month(ws, i + 1, d0, loose=w in ("ngày", "hôm"))
            if exact:
                out.append(exact)
        i += 1
    return list(dict.fromkeys(d.isoformat() for d in out))


def _clock_spans(ws: list[str]):
    """Như numbers._spans, nhưng "thứ năm ba giờ", "thứ bảy ba rưỡi": bỏ từ tên ngày, giữ phần giờ đi sau."""
    i = 0
    while i < len(ws):
        j = i
        while j < len(ws) and N._isnum(ws[j]):
            j += 1
        if j == i:
            i += 1
            continue
        if i and ws[i - 1] in ("thứ", "tháng", "mùng", "mồng"):
            i += 1
            if i == j:
                continue
        if ws[i:j] != ["rưỡi"]:
            yield i, j
        i = j


def spoken_clocks(text: str) -> list[str]:
    """Giờ trong ngày nói ra, "HH:MM": "9 giờ kém 15 tối" 20:45, "tối mai bảy giờ" 19:00, "bảy rưỡi" 07:30 và 19:30.
    Buổi lấy ở ngay sau giờ, hoặc trong 3 từ phía trước, không vượt qua một số khác (khác numbers.clock_times)."""
    ws = _ws(text)
    out = []
    for i, j in _clock_spans(ws):
        rui = ws[j - 1] == "rưỡi"
        nxt = ws[j] if j < len(ws) else ""
        if nxt not in ("giờ", "h") and not rui:
            continue
        h = N._value(ws[i:j - 1] if rui else ws[i:j])
        if h is None or not 0 <= h <= 24 or h != int(h):
            continue
        m, k = (30, j) if rui else (0, j + 1)
        rest = ws[k:k + 4]
        if m == 0 and rest[:1] == ["rưỡi"]:
            m, k = 30, k + 1
        elif m == 0 and rest[:1] == ["kém"]:
            q = 1
            while q < len(rest) and N._isnum(rest[q]):
                q += 1
            n = N._value(rest[1:q]) if q > 1 else None
            if n and 0 < n < 60:
                h, m, k = h - 1, 60 - n, k + q + (rest[q:q + 1] == ["phút"])
        elif m == 0 and rest:
            q = 0
            while q < len(rest) and N._isnum(rest[q]) and rest[q] != "rưỡi":
                q += 1
            n = N._value(rest[:q]) if q else None
            if n is not None and 0 < n < 60 and not (q < len(rest) and rest[q] in ("tiếng", "ngày", "giờ")):
                m, k = n, k + q + (rest[q:q + 1] == ["phút"])
        part = None
        for x in ws[k:k + 3]:  # buổi ngay sau: "tám giờ tối", "sáu giờ chiều mai"
            if N._isnum(x):
                break
            if x in _AM | _PM:
                part = x
                break
        if part is None:
            for x in reversed(ws[max(0, i - 3):i]):  # buổi ngay trước: "tối mai bảy giờ"
                if N._isnum(x):
                    break
                if x in _AM | _PM:
                    part = x
                    break
        h = int(h)
        if part in _PM and h < 12 and not (part == "đêm" and h < 5):
            hs = [h + 12]
        elif part == "trưa" and h <= 2:
            hs = [h + 12]
        elif part or h == 0 or h >= 12:
            hs = [h]
        else:
            hs = [h, h + 12]
        out += [f"{x % 24:02d}:{int(m):02d}" for x in hs]
    for hh, mm in re.findall(r"\b(\d{1,2})[:h](\d{2})\b", str(text)):
        out.append(f"{int(hh):02d}:{mm}")
    return list(dict.fromkeys(out))


_OFFSET = re.compile(r"\b(trước|sau|sớm|muộn|trễ|lùi|thêm|cộng|trừ)\b(\s+\S+){0,6}?\s+(phút|tiếng|ngày|tuần)\b"
                     r"|(?<!thứ )\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|lăm|rưỡi|nửa|vài|mấy)\s+(phút|tiếng|ngày|tuần)\s+"
                     r"(trước|sau|nữa)\b|\b(trừ|cộng|tính)\b")  # "hai tiếng sau"; "thứ ba tuần sau" là ngày, không phải dời
_REL_DAY = re.compile(r"\b(hôm|ngày|bữa)\s+(trước|sau|đó|ấy|đấy|nớ)\b|\b(trước|sau)\s+(\S+\s+)?ngày\b"
                      r"|\b(cùng|đúng)\s+(ngày|hôm|bữa)\b")
_REL_TIME = re.compile(r"\bkịp\b|\btheo\s+giờ\b|\b(trước|sau)\s+(\S+\s+){0,3}?(giờ|rưỡi)\b|\b(cùng|đúng)\s+(giờ|lúc)\b"
                       r"|\bgiờ\s+(đó|đấy|ấy|cũ)\b|\bgiữ\s+nguyên\b|\blúc\s+(đó|đấy|ấy)\b")
_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2})(:\d{2})?([+-]\d{2}:\d{2})?)?$")


def _norm(text: str) -> str:
    return " ".join(_ws(text))


def _end_like(key: str) -> bool:
    return "end" in key or "until" in key or key.endswith("_to")


def _shift(value: str, d: str | None, t: str | None) -> str:
    m = _ISO.match(value)
    out = (d or m.group(1))
    if m.group(2):
        out += "T" + (t or m.group(2)) + (m.group(3) or ":00") + (m.group(4) or "")
    return out


def spoken_span(text: str) -> int | None:
    """Số phút của "từ 7 rưỡi tới 8 giờ 10", "từ chín giờ đến mười giờ kém mười lăm"; None nếu không có."""
    m = re.search(r"\btừ\s+(.{2,30}?)\s+(?:tới|đến)\s+(.{2,30})", _norm(text))
    if not m:
        return None
    a, b = spoken_clocks(m.group(1))[-2:], spoken_clocks(m.group(2))[:2]  # câu tự sửa: giờ sau cùng bên trái
    if not a or not b:
        return None
    best = None
    for x in a:
        for y in b:
            d = (int(y[:2]) * 60 + int(y[3:]) - int(x[:2]) * 60 - int(x[3:]))
            if 0 < d <= 600 and (best is None or d < best):
                best = d
    return best


def fix_args(name: str, args: dict, spec: dict, said: str, now: str, seen: set, rules: dict,
             past: str = "") -> tuple[dict, list[str]]:
    """Trả (tham số đã sửa, ghi chú từng chỗ sửa). seen: mọi chuỗi và số trong lịch sử và kết quả tool lượt này;
    giá trị đã có ở đó coi là có nguồn, không sửa. past: lời các lượt user trước (mốc "hôm đó" của rel_day_fix)."""
    props = (spec.get("parameters") or {}).get("properties", {})
    out, notes = dict(args), []
    norm = _norm(said)
    if rules.get("clock_fix"):
        from .context import timetable_dates
        dates = timetable_dates(now)
        seen_days = {x[:10] for x in seen if isinstance(x, str) and _ISO.match(x)}
        seen_times = {x[11:16] if _ISO.match(x) else x for x in seen
                      if isinstance(x, str) and (_ISO.match(x) or re.fullmatch(r"\d{2}:\d{2}", x))}
        ds, cs = spoken_dates(said, now), spoken_clocks(said)
        shifts = {}
        for k, v in args.items():
            if not _temporal_prop(props.get(k, {})):
                continue
            m = _ISO.match(v) if isinstance(v, str) else None
            if not m or _end_like(k) or v in seen or N.derive(v, said, [list(seen)], now, dates) is not None:
                continue
            d, t = m.group(1), m.group(2)
            nd = ds[0] if len(ds) == 1 and d != ds[0] and d not in seen_days and not _REL_DAY.search(norm) \
                and not re.search(r"\b(ngày|tuần)\b", " ".join(x.group(0) for x in _OFFSET.finditer(norm))) else None
            nt = None
            if t in seen_times:  # giờ tàu tới, giờ trong lịch: có nguồn
                t = None
            if t and not _OFFSET.search(norm) and not _REL_TIME.search(norm):
                if len(cs) == 1 and t != cs[0]:
                    nt = cs[0]
            if t and nt is None and t not in cs:  # sai sáng / chiều: cùng giờ trên mặt đồng hồ với đúng một giờ nói ra
                same = [c for c in cs if int(c[:2]) % 12 == int(t[:2]) % 12 and c[3:] == t[3:]]
                if len(same) == 1:
                    nt = same[0]
            if nd or nt:
                out[k] = _shift(v, nd, nt)
                shifts[k] = (v, out[k])
                notes.append(f"clock_fix {k}: {v} → {out[k]}")
        if shifts:  # dời giờ bắt đầu thì dời giờ kết thúc theo, giữ thời lượng LLM đã chọn
            (old, new), = list(shifts.values())[:1]
            delta = datetime.fromisoformat(new) - datetime.fromisoformat(old) if "T" in old else None
            for k, v in args.items():
                if not _temporal_prop(props.get(k, {})):
                    continue
                if delta and _end_like(k) and isinstance(v, str) and _ISO.match(v) and "T" in v and v not in seen:
                    out[k] = (datetime.fromisoformat(v) + delta).isoformat()
                    notes.append(f"clock_fix {k}: {v} → {out[k]}")
    for k, v in args.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        desc = props.get(k, {}).get("description", "")
        if rules.get("money_k") and "1200000" in desc and not name.startswith("exchange_rate") and 0 < v < 1000:
            out[k] = int(v * 1000)
            notes.append(f"money_k {k}: {v} → {out[k]}")
        if rules.get("span_min") and ("minutes" in k and "delay" not in k):
            span = spoken_span(said)
            if span and span != v:
                out[k] = span
                notes.append(f"span_min {k}: {v} → {span}")
    if rules.get("phone_fix"):
        out, more = _phone_fix(out, props, said, seen)
        notes += more
    if rules.get("rel_day_fix"):
        out, more = _rel_day_fix(out, said, now, seen, past, props=props)
        notes += more
    if rules.get("now_offset_fix"):
        out, more = _now_offset_fix(out, said, now, seen, props=props)
        notes += more
    if rules.get("clock_face"):
        out, more = _clock_face(out, props, said, now, seen)
        notes += more
    return out, notes


def _clock_face(args: dict, props: dict, said: str, now: str, seen: set) -> tuple[dict, list[str]]:
    """Lời nói đúng MỘT giờ trên mặt đồng hồ ("mười giờ mười lăm", "chín giờ năm mươi"), không kèm dời giờ hay giờ theo
    mốc, mà tham số ngày giờ cùng giờ trên mặt đồng hồ đó lại lệch:

    - lệch phút (LLM viết 09:30 khi lời nói "chín giờ năm mươi"): lấy phút nói ra, giữ giờ LLM chọn;
    - lời không nói buổi (hai cách hiểu 10:15 và 22:15) và ngày của tham số là hôm nay: lấy lúc gần nhất còn ở phía
      trước giờ hiện tại.

    clock_fix không đụng hai ca này: nó chỉ sửa khi lời có đúng một cách hiểu, hoặc khi giờ LLM viết không có trong lời.
    Giá trị có nguồn (lịch sử, kết quả tool) hay tính được một bước (numbers.derive) không sửa."""
    norm = _norm(said)
    if _OFFSET.search(norm) or _REL_TIME.search(norm):
        return args, []
    cs = spoken_clocks(said)
    faces = {(int(c[:2]) % 12, c[3:]) for c in cs}
    if len(faces) != 1:
        return args, []
    (face_h, face_m), = faces
    from .context import timetable_dates
    dates = timetable_dates(now)
    n = datetime.fromisoformat(now)
    seen_times = {x[11:16] if _ISO.match(x) else x for x in seen
                  if isinstance(x, str) and (_ISO.match(x) or re.fullmatch(r"\d{2}:\d{2}", x))}
    out, notes = dict(args), []
    for k, v in args.items():
        if not _temporal_prop(props.get(k, {})) or _end_like(k):
            continue
        m = _ISO.match(v) if isinstance(v, str) else None
        if not m or not m.group(2) or v in seen or m.group(2) in seen_times:
            continue
        if N.derive(v, said, [list(seen)], now, dates) is not None:
            continue
        h = int(m.group(2)[:2])
        if h % 12 != face_h:
            continue  # giờ khác hẳn: việc của clock_fix
        new_h = h
        if len(cs) == 2 and m.group(1) == n.date().isoformat():
            ahead = [int(c[:2]) for c in sorted(cs) if n.replace(hour=int(c[:2]), minute=int(c[3:]), second=0) > n]
            if ahead:
                new_h = ahead[0]
        t = f"{new_h:02d}:{face_m}"
        if t != m.group(2):
            out[k] = _shift(v, None, t)
            notes.append(f"clock_face {k}: {v} → {out[k]}")
    if notes:  # dời giờ bắt đầu thì dời giờ kết thúc theo, giữ thời lượng LLM đã chọn
        k0 = next(k for k in out if out[k] != args.get(k))
        delta = datetime.fromisoformat(out[k0]) - datetime.fromisoformat(args[k0])
        for k, v in args.items():
            if _temporal_prop(props.get(k, {})) and _end_like(k) and isinstance(v, str) and _ISO.match(v) and "T" in v \
                    and v not in seen:
                out[k] = (datetime.fromisoformat(v) + delta).isoformat()
                notes.append(f"clock_face {k}: {v} → {out[k]}")
    return out, notes


def _is_phone_param(key: str, prop: dict) -> bool:
    desc = str(prop.get("description", "")).lower()
    return "phone" in key.lower() or "điện thoại" in desc or "số máy" in desc


def _phone_fix(args: dict, props: dict, said: str, seen: set) -> tuple[dict, list[str]]:
    """Tham số số điện thoại lệch số nói ra duy nhất (normalize.phones) thì lấy số nói ra; cùng chữ số mà khác cách
    viết ("0901 234 567", "+84901…") thì viết lại dạng liền. Số đã có trong lịch sử hay kết quả tool không sửa."""
    from . import normalize as Z
    spoken = Z.phones(said)
    out, notes = dict(args), []
    # A unique mentioned number can be the explicitly rejected target.
    if len(spoken) != 1 or re.search(r"\b(?:đừng|chớ)\b|\bkhông\s+(?:gọi|nạp|đổi)\b", _norm(said)):
        return out, notes
    for k, v in args.items():
        if not isinstance(v, str) or v in seen or not _is_phone_param(k, props.get(k, {})):
            continue
        if v != spoken[0]:
            out[k] = spoken[0]
            notes.append(f"phone_fix {k}: {v} → {spoken[0]}")
    return out, notes


def _temporal_prop(prop: dict) -> bool:
    """Free text may itself be an ISO string; only rewrite schema date/time parameters."""
    return prop.get("format") in ("date", "date-time") or bool(
        re.search(r"ISO|YYYY-MM-DD", prop.get("description", ""), re.I))


def _rel_day_fix(args: dict, said: str, now: str, seen: set, past: str = "", *, props: dict | None = None) -> tuple[dict, list[str]]:
    """Lời có ngày tính theo mốc ("hôm trước", "hôm đó", "trước … một ngày") với mốc là ngày nói rõ duy nhất:

    - Tham số ngày giờ: nếu giờ của nó được lời ghép với đúng một ngày (normalize.day_for_clock), mà ngày của tham số
      khác ngày đó, thì sửa ngày. Giờ lệch sáng / chiều mà khớp đúng một giờ ghép thì sửa cả giờ.
    - Tham số chỉ có ngày: lấy ngày tính theo mốc khi mệnh đề đó không nói giờ; nếu mọi ngày tính theo mốc đã đi với
      giờ (tức thuộc tham số ngày giờ), thì tham số chỉ ngày lấy ngày mốc.
    - Dời giờ bắt đầu thì dời giờ kết thúc theo, giữ thời lượng.

    Không đụng giá trị có nguồn (seen) hay giá trị tính được một bước (numbers.derive). Khác clock_fix, ngày
    chỉ trùng một ngày trong dữ liệu vẫn được sửa: lời đã neo ngày rõ ("hôm trước" của ngày nói ra), còn ngày trong
    dữ liệu thường là hôm nay hay ngày trống bất kỳ."""
    from . import normalize as Z
    from .context import timetable_dates
    # Mốc nằm ngay trong cụm dời ("trước thứ sáu tuần này một ngày") chỉ dùng cho gợi ý: qua ASR, cụm này hay bị nghe
    # nhầm ("trước bữa Tí bay một ngày" thành "trước bữa thứ bảy một ngày"), nên luật sửa không dựa vào nó.
    anchors = [a for a in Z.anchored(said, now, past) if not a.inner]
    out, notes = dict(args), []
    if not anchors:
        return out, notes
    dates = timetable_dates(now)
    by_clock = Z.day_for_clock(said, now, past)
    free = [a for a in anchors if not a.clocks]
    base = anchors[0].base
    shifts = {}
    for k, v in args.items():
        if props is not None and not _temporal_prop(props.get(k, {})):
            continue
        m = _ISO.match(v) if isinstance(v, str) else None
        if not m or _end_like(k) or v in seen or N.derive(v, said, [list(seen)], now, dates) is not None:
            continue
        d, t = m.group(1), m.group(2)
        new = None
        if t is None:
            if d in spoken_dates(said, now):
                continue  # explicit base date may belong to this action; do not steal another action's relative day
            want = free[0].day if len(free) == 1 else (base if not free else None)
            if want and d != want:
                new = want
        else:
            days = by_clock.get(t)
            nt = None
            if days is None:  # sai sáng / chiều: cùng giờ trên mặt đồng hồ với đúng một giờ được ghép
                same = [c for c in by_clock if int(c[:2]) % 12 == int(t[:2]) % 12 and c[3:] == t[3:]]
                if len(same) == 1:
                    nt, days = same[0], by_clock[same[0]]
            if days and len(days) == 1:
                (want,) = days
                if d != want or nt:
                    new = _shift(v, want, nt)
        if new and new != v:
            out[k] = new
            shifts[k] = (v, new)
            notes.append(f"rel_day_fix {k}: {v} → {new}")
    if shifts:
        (old, new), = list(shifts.values())[:1]
        delta = datetime.fromisoformat(new) - datetime.fromisoformat(old) if "T" in old else None
        for k, v in args.items():
            if props is not None and not _temporal_prop(props.get(k, {})):
                continue
            if delta and _end_like(k) and isinstance(v, str) and _ISO.match(v) and "T" in v and v not in seen:
                out[k] = (datetime.fromisoformat(v) + delta).isoformat()
                notes.append(f"rel_day_fix {k}: {v} → {out[k]}")
    return out, notes


def _now_offset_fix(args: dict, said: str, now: str, seen: set, *, props: dict | None = None) -> tuple[dict, list[str]]:
    """"Bốn lăm phút nữa nhắc chị": tham số ngày giờ (không phải giờ kết thúc) lệch bây giờ + thời lượng thì sửa, khi
    lời có đúng một cụm "… nữa", không nói giờ nào trong ngày, và giá trị không có nguồn."""
    from . import normalize as Z
    from .context import timetable_dates
    cands = [x for x in Z.now_offsets(said, now) if "T" in x]
    out, notes = dict(args), []
    if len(cands) != 1 or spoken_clocks(said):
        return out, notes
    want = cands[0]
    dates = timetable_dates(now)
    shifts = []
    for k, v in args.items():
        if props is not None and not _temporal_prop(props.get(k, {})):
            continue
        m = _ISO.match(v) if isinstance(v, str) else None
        if not m or not m.group(2) or _end_like(k) or v in seen:
            continue
        if N.derive(v, said, [list(seen)], now, dates) is not None:
            continue
        if datetime.fromisoformat(v) != datetime.fromisoformat(want):
            out[k] = want
            shifts.append(datetime.fromisoformat(want) - datetime.fromisoformat(v))
            notes.append(f"now_offset_fix {k}: {v} → {want}")
    if len(shifts) == 1:
        for k, v in args.items():
            if props is not None and not _temporal_prop(props.get(k, {})):
                continue
            if _end_like(k) and isinstance(v, str) and _ISO.match(v) and "T" in v and v not in seen:
                out[k] = (datetime.fromisoformat(v) + shifts[0]).isoformat()
                notes.append(f"now_offset_fix {k}: {v} → {out[k]}")
    return out, notes


def strings(obj, into: set) -> set:
    """Mọi chuỗi và số lá trong obj (kết quả tool, tham số call, lịch sử)."""
    if isinstance(obj, dict):
        for v in obj.values():
            strings(v, into)
    elif isinstance(obj, list):
        for v in obj:
            strings(v, into)
    elif isinstance(obj, (str, int, float)) and not isinstance(obj, bool):
        into.add(obj)
    return into


_VAGUE_MONEY = re.compile(r"\b(chút|ít|tí)\s+tiền\b|\bmấy đồng\b")
_WHEN_REF = re.compile(r"\b(xong|khi|lúc)\b")  # "giặt xong thì nhắc", "lúc thợ làm xong": giờ lấy từ một mốc khác


def grounded(value, said: str, now: str, seen: set) -> bool:
    """Giá trị số / ngày giờ có nguồn: nói ra (kể cả "hai trăm" cho 200.000), có trong lịch sử hay kết quả tool
    (cả giờ viết trong văn bản: tin nhắn, ghi chú), hoặc tính được một bước từ các số đó (numbers.derive).
    said: lời lượt này và các lượt user trước."""
    from .context import timetable_dates
    if value in seen:
        return True
    texts = [x for x in seen if isinstance(x, str) and len(x) > 3 and not _ISO.match(x)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        nums = set(N.numbers(said)) | {float(x) for x in seen if isinstance(x, (int, float)) and not isinstance(x, bool)}
        nums |= {n for x in texts for n in N.numbers(x)}
        v = float(value)
        if v in nums or v / 1000 in nums or v / 1000000 in nums:
            return True
    elif isinstance(value, str) and _ISO.match(value):
        t = _ISO.match(value).group(2)
        if t is None:
            return value in spoken_dates(said, now)
        if t in spoken_clocks(said) or any(x[11:16] == t or x == t for x in seen if isinstance(x, str)):
            return True
        if any(t in spoken_clocks(x) for x in texts):
            return True
    else:
        return True
    return N.derive(value, said, [list(seen)], now, timetable_dates(now)) is not None


def missing(name: str, args: dict, spec: dict, said: str, now: str, seen: set, final: str | None = None) -> list[str]:
    """Tham số bắt buộc LLM tự bịa khi người dùng không nói, chỗ gold hỏi lại (sys.ask_missing_info):
    số tiền VND khi lời chỉ nói "chút tiền", "ít tiền" (ewallet.transfer amount, 19 mẫu train), giờ hẹn khi lời và
    lịch sử không có giờ hay thời lượng nào (reminders.create time 7, video_meeting.create start 6).
    said: lời lượt này và các lượt user trước; final: riêng lời lượt này (cách nói "chút tiền" chỉ tính ở đây)."""
    props = (spec.get("parameters") or {}).get("properties", {})
    req = set((spec.get("parameters") or {}).get("required", []))
    norm = _norm(said)
    vague = _VAGUE_MONEY.search(_norm(final if final is not None else said))
    out = []
    for k, v in args.items():
        if k not in req or isinstance(v, bool):
            continue
        desc = props.get(k, {}).get("description", "")
        money = isinstance(v, (int, float)) and "1200000" in desc and not name.startswith("exchange_rate")
        when = isinstance(v, str) and bool(_ISO.match(v)) and _ISO.match(v).group(2) is not None and not _end_like(k)
        if money and not vague:
            continue
        if when and (spoken_clocks(said) or N.durations(said) or _WHEN_REF.search(norm)):
            continue
        if (money or when) and not grounded(v, said, now, seen):
            out.append(k)
    return out


_CLF = r"(cái|gói|đơn|việc|chiếc|tờ|cuốn|kiện|món|lịch|máy)"
_PICK_ONE = re.compile(rf"\bbớt\s+(một|1)\s+{_CLF}\b|\bmột trong (hai|mấy|các|những)\b"
                       rf"|\b(hủy|huỷ|xóa|xoá|bỏ|tắt|dừng|ngừng|cắt)(\s+\S+){{0,2}}\s+(một|1)\s+{_CLF}(\s+\S+)?\s+thôi\b")
_MANY = re.compile(rf"\b(mấy|hai|các|những|cả hai)\s+{_CLF}\b|\bbớt\b|\bmột trong\b")


def pick_one(name: str, args: dict, spec: dict, said: str) -> list[str]:
    """"Mấy gói ... hủy bớt một gói thôi", "hai cái máy ... tắt bớt một cái": người dùng muốn một trong nhiều mục mà
    không nói mục nào; cả 5 mẫu train có cách nói này gold đều hỏi lại. Trả tham số ID của call ghi cần hỏi."""
    norm = _norm(said)
    if not (_PICK_ONE.search(norm) and _MANY.search(norm)) or re.search(r"\bmỗi\b", norm):
        return []  # "mỗi cái một vé thôi", "giữ cho chị một chiếc", "mở một cái thôi" là số lượng, không phải chọn
    props = (spec.get("parameters") or {}).get("properties", {})
    return [k for k in args if k.endswith("_id") and props.get(k, {}).get("x-ref")][:1]
