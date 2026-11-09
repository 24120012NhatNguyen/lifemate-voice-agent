"""Tầng N: chuẩn hoá tất định từ lời nói, trước khi LLM đọc.

LLM 4B hay sai ở những chỗ phải đổi lời nói ra giá trị:

- số điện thoại đọc từng chữ số: "không chín không một, hai ba bốn, năm sáu bảy bảy" ra 09012345677;
- ngày tính theo một ngày khác: "sáu giờ chiều hôm trước" khi mốc là "thứ sáu này";
- giờ tính từ bây giờ: "bốn lăm phút nữa".

Đếm trên 1.545 mẫu train:

- 18 số điện thoại đọc từng chữ số; `numbers.numbers` không đọc được các số này;
- 22 giờ hẹn có giờ nói ra nhưng ngày là "hôm đó", "hôm trước";
- 122 ngày giờ phải cộng trừ từ một mốc.

Tầng này đọc các giá trị đó bằng luật, dùng lại bộ đọc ngày giờ và số của fixes.py và numbers.py. Kết quả dùng ở hai chỗ:

- `agent.normalize_hint`: khối "giá trị máy đọc" trong prompt, để LLM chép giá trị thay vì tự tính;
- luật sửa tham số trong fixes.py (`phone_fix`, `rel_day_fix`, `now_offset_fix`): dùng làm nguồn đối chiếu.

Các hàm:

    facts(text, now)           -> list[Fact]   giá trị đọc được từ một bản chép
    merge(hyps, now)           -> list[Fact]   gộp N-best: bản đầu trước; bản khác chỉ thêm giá trị bản đầu không có
    render(facts)              -> str          khối chữ cho prompt ("" nếu không có gì)
    phones(text)               -> list[str]    số điện thoại đọc từng chữ số hoặc viết liền
    phone_suffixes(text)       -> list[str]    "số đuôi tám sáu tám" -> "868"
    anchored(text, now)        -> list[Anchored]   "hôm trước", "hôm đó", "trước … một ngày" tính từ ngày nói rõ duy nhất
    now_offsets(text, now)     -> list[str]    "bốn lăm phút nữa" -> ISO của bây giờ + 45 phút
    offsets(text)              -> list[Offset] "trước hạn khiếu nại một ngày" -> −1440 phút so với một mốc
    amounts(text)              -> list[tuple]  "năm mươi nghìn" -> (50000, ""), "hai nghìn bước" -> (2000, "bước")
    percents(text)             -> list[int]    "30 phần trăm" -> 30, "một nửa" -> 50

Mọi hàm chỉ đọc lời nói và giờ hiện tại, không đọc nhãn hay state. Giá trị cần mốc lấy từ dữ liệu ("trước hạn khiếu
nại một ngày") chỉ được tả dạng phép dời; LLM vẫn phải đọc tool để biết mốc.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from . import fixes as F
from . import numbers as N

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


@dataclass(frozen=True)
class Fact:
    kind: str  # phone | phone_suffix | date | anchored | datetime | now_offset | offset | amount | percent | duration
    value: str | int | float
    label: str  # cách nói, hoặc giải thích ngắn
    source: str = "top"  # top: bản chép đầu; alt: chỉ có ở bản nghe khác


@dataclass(frozen=True)
class Anchored:
    day: str  # YYYY-MM-DD
    clocks: tuple[str, ...]  # giờ nói trong cùng mệnh đề, "HH:MM"
    label: str  # "hôm trước của 2026-10-09"
    base: str = ""  # ngày mốc nói rõ, YYYY-MM-DD
    inner: bool = False  # mốc nằm ngay trong cụm dời của cùng mệnh đề


@dataclass(frozen=True)
class Offset:
    minutes: int  # có dấu: trước / sớm là âm
    label: str  # đoạn lời nói, vd. "trước hạn khiếu nại một ngày"


# ------------------------------------------------------------------ token

def _toks(text: str) -> list[str]:
    """Như numbers._words nhưng giữ dấu câu ngắt mệnh đề (",", ".", "?", "!", ";", "…") thành token riêng."""
    t = unicodedata.normalize("NFC", str(text).lower())
    return re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+|[.,;!?…]", t)


_PUNCT = {".", ",", ";", "!", "?", "…"}
_BREAK_WORDS = {"rồi", "xong", "còn"}  # đầu một việc mới: "…, rồi đặt…", "xong nhắc…"


def clauses(text: str, *, preserve_case: bool = False) -> list[str]:
    """Tách lời thành các mệnh đề theo dấu câu và các từ nối "rồi", "xong", "còn", "với cả", "với lại"."""
    out, cur = [], []
    toks = _toks(text)
    original = re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+|[.,;!?…]",
                          unicodedata.normalize("NFC", str(text))) if preserve_case else toks
    for i, w in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if w in _PUNCT or (w in _BREAK_WORDS and cur) or (w == "với" and nxt in ("cả", "lại") and cur):
            if cur:
                out.append(" ".join(cur))
            cur = [] if w in _PUNCT else [original[i]]
            continue
        cur.append(original[i])
    if cur:
        out.append(" ".join(cur))
    return out


# ------------------------------------------------------------------ cách viết chuẩn

_HAM = re.compile(r"\b([Hh])ăm(?=\s+(?:mốt|một|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín|mươi)\b)")
_G_CLOCK = re.compile(r"\b(\d{1,2})\s?[gG](\d{2})?\b")


def _canon(text: str) -> str:
    """Đưa cách viết, cách nói vùng miền về dạng bộ đọc hiểu:
    - "hăm mốt" ra "hai mươi mốt";
    - "hăm nhăm" ra "hai mươi nhăm";
    - "6g15" (miền Nam viết "g" thay "h") ra "6 giờ 15".
    Giữ hoa / thường để bộ đọc ngày còn nhận ra tên riêng ("cô Mai")."""
    t = unicodedata.normalize("NFC", str(text))
    t = _HAM.sub(lambda m: m.group(1).replace("H", "Hai").replace("h", "hai") + " mươi", t)
    return _G_CLOCK.sub(lambda m: f"{m.group(1)} giờ" + (f" {m.group(2)}" if m.group(2) else ""), t)


def spoken_dates(text: str, now: str) -> list[str]:
    """fixes.spoken_dates trên lời đã chuẩn cách viết, cộng thêm:
    - "N tháng M" không có "ngày" / "mùng" đứng trước: "hai mốt tháng mười", "hăm mốt tháng mười";
    - "hôm kia" / "bữa kia": hai ngày trước. Khác "ngày kia" là hai ngày sau; fixes đọc "ngày kia" rồi.
    Dùng trong tầng N; clock_fix của config 98 vẫn dùng fixes.spoken_dates như cũ."""
    t = _canon(text)
    out = list(F.spoken_dates(t, now))
    ws = N._words(t)
    d0 = datetime.fromisoformat(now).date()
    for i, j in N._spans(ws):
        if j + 1 < len(ws) and ws[j] == "tháng" and N._isnum(ws[j + 1]) and not (i and ws[i - 1] in ("ngày", "mùng", "mồng", "hôm", "thứ", "trong", "mấy")):
            d = F._day_month(ws, i, d0, loose=False)
            if d:
                out.append(d.isoformat())
    if re.search(r"\b(?:hôm|bữa)\s+kia\b", " ".join(ws)):
        out.append((d0 - timedelta(days=2)).isoformat())
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ số điện thoại

_DIGIT = {"không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4", "năm": "5", "lăm": "5",
          "nhăm": "5", "sáu": "6", "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9", "linh": "0", "lẻ": "0"}
_NOT_FIRST = {"linh", "lẻ", "lăm", "nhăm", "mốt"}  # không mở đầu một dãy: "em Linh 50 ngàn" không phải số
_ONE_TO_NINE = set(_DIGIT) - {"không", "linh", "lẻ"}
# đọc theo cụm: "ba mươi tư" 34, "một linh tám" 108, "một trăm hai mươi ba" 123, "kép tám" 88, "tứ quý chín" 9999
_GROUP = {"mười", "mươi", "trăm", "kép", "tam", "hoa", "tứ", "quý"}
_PHONE_CUE = {"số", "sđt", "thoại", "gọi", "nạp", "đổi", "thành", "là", "đuôi", "máy", "hotline", "cộng"}


def _run_digits(seq: list[str]) -> str:
    """Chuỗi chữ số của một dãy token số điện thoại, kể cả cách đọc theo cụm."""
    out, rep = "", 0
    for k, w in enumerate(seq):
        nxt = seq[k + 1] if k + 1 < len(seq) else ""
        if w in _DIGIT:
            out += _DIGIT[w] * (rep or 1)
            rep = 0
        elif w.isdigit():
            out += w
        elif w == "mười":
            out += "1" + ("" if nxt in _ONE_TO_NINE else "0")
        elif w == "mươi":
            out += "" if nxt in _ONE_TO_NINE else "0"
        elif w == "trăm":
            out += "" if (nxt in _ONE_TO_NINE or nxt in ("linh", "lẻ", "mười")) else "00"
        elif w == "kép":
            rep = 2
        elif w == "hoa" and k and seq[k - 1] == "tam":
            rep = 3
        elif w == "quý" and k and seq[k - 1] == "tứ":
            rep = 4
    return out


def _digit_runs(text: str):
    """Các dãy token liền nhau mà mỗi token là một chữ số đọc bằng chữ hoặc một cụm chữ số viết liền; dấu phẩy, chấm
    ngắt nhịp đọc ("không chín không một, hai ba bốn") không cắt dãy. Trả (vị trí token đầu, chuỗi chữ số, các token)."""
    toks = _toks(_canon(text))
    i = 0
    while i < len(toks):
        if not ((toks[i] in _DIGIT and toks[i] not in _NOT_FIRST) or toks[i].isdigit()):
            i += 1
            continue
        j, used = i, []
        while j < len(toks):
            w = toks[j]
            if w in _DIGIT or w.isdigit() or (w in _GROUP and used):
                used.append(w)
                j += 1
            elif w in (",", ".") and j + 1 < len(toks) and (toks[j + 1] in _DIGIT or toks[j + 1].isdigit()):
                j += 1  # dấu ngắt nhịp chỉ nối sang chữ số thường: "…bảy, mười phút nữa" không nuốt "mười"
            else:
                break
        yield i, _run_digits(used), used, toks
        i = max(j, i + 1)


def phones(text: str) -> list[str]:
    """Số điện thoại nói ra.
    - "không chín năm không năm ba sáu không ba sáu bảy" ra "09505360367".
    - "0988 999 0002" ra "09889990002".
    - "+84 912…" và "84 912…" ra "0912…".

    Tổng đài 1800/1900: 8–10 chữ số, cần từ gợi ý. Dấu chấm nhóm chữ số được bỏ.
    Điều kiện số thường: ít nhất 9 chữ số, bắt đầu bằng 0 hoặc 84. Riêng dãy đọc bằng chữ thì phải có từ gợi ý ("số",
    "gọi", "thành"…) trong 3 token trước, để "hai ba bốn năm" (đếm) không thành số điện thoại.

    Dãy không được chứa từ chỉ lượng ("mười", "trăm", "nghìn"…): những từ đó ngắt dãy. Hai số không ở đầu
    ("à không, không chín…") được bỏ bớt một."""
    out = []
    text = re.sub(r"\b\d+(?:\.\d{2,4})+\b", lambda m: m.group().replace(".", ""), text)
    for i, digits, used, toks in _digit_runs(text):
        cue = bool(_PHONE_CUE & set(toks[max(0, i - 3):i]))
        if digits.startswith(("1800", "1900")) and 8 <= len(digits) <= 10 and cue:
            out.append(digits)
            continue
        if len(digits) < 9:
            continue
        spoken = any(not w.isdigit() for w in used)
        if spoken and not (_PHONE_CUE & set(toks[max(0, i - 3):i])):
            continue
        if digits.startswith("84") and len(digits) >= 11:
            digits = "0" + digits[2:]
        while digits.startswith("00"):
            digits = digits[1:]
        if digits.startswith("0") and 9 <= len(digits) <= 12:
            out.append(digits)
    return list(dict.fromkeys(out))


def phone_suffixes(text: str) -> list[str]:
    """Đuôi số nói ra: "số đuôi tám sáu tám" -> "868", "đuôi là sáu tám" -> "68" (ít nhất 2 chữ số)."""
    toks = _toks(text)
    out = []
    for k, w in enumerate(toks):
        if w != "đuôi":
            continue
        digits = ""
        start = k + 1 + (toks[k + 1:k + 2] == ["là"])
        for x in toks[start:start + 8]:
            if x in _DIGIT:
                digits += _DIGIT[x]
            elif x.isdigit():
                digits += x
            elif x in (",", "."):
                continue
            else:
                break
        if len(digits) >= 2:
            out.append(digits)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ ngày giờ

def _tz(now: str) -> str:
    z = datetime.fromisoformat(now).strftime("%z") or "+0700"
    return f"{z[:3]}:{z[3:]}"


def _iso(day: str, clock: str, now: str) -> str:
    if clock == "24:00":  # nửa đêm của ngày đó là 00:00 ngày hôm sau
        day, clock = (date.fromisoformat(day) + timedelta(days=1)).isoformat(), "00:00"
    return f"{day}T{clock}:00{_tz(now)}"


_MIDNIGHT = re.compile(r"\bnửa đêm\b|\b(?:mười hai|12)\s+giờ\s+(?:đêm|khuya)\b")


# "hôm trước ngày bay", "hôm sau buổi họp": mốc là một thứ trong dữ liệu, không phải ngày nói ra
_ENTITY = r"(?:ngày|buổi|hạn|lúc|giờ|chuyến|khi|lịch|hôm|cái|vụ|đợt|kỳ|lần)"
_REL_BEFORE = re.compile(rf"\b(?:hôm|bữa|ngày)\s+trước\b(?!\s+{_ENTITY}\b)|\btrước\s+(?:đó|hôm\s+đó)\s+(?:một|1)\s+ngày\b")
_REL_AFTER = re.compile(rf"\b(?:hôm|bữa|ngày)\s+sau\b(?!\s+{_ENTITY}\b)|\bsau\s+(?:đó|hôm\s+đó)\s+(?:một|1)\s+ngày\b")
_REL_SAME = re.compile(r"\b(?:hôm|bữa|ngày)\s+(?:đó|ấy|đấy|nớ)\b|\b(?:cùng|đúng)\s+(?:hôm|ngày|bữa)\b")
# "trước … một ngày", "sau … hai ngày": dời theo số ngày từ mốc (mốc là ngày nói rõ duy nhất trong lời)
_REL_N = re.compile(r"\b(trước|sau)\s+(?:\S+\s+){0,6}?(một|hai|ba|1|2|3)\s+ngày\b")


def anchored(text: str, now: str, past: str = "") -> list[Anchored]:
    """Ngày nói theo một ngày khác: "hôm trước", "hôm sau", "hôm đó", "cùng hôm", "trước … một ngày".

    Mốc là ngày nói rõ duy nhất trong cả lời (fixes.spoken_dates). Không có mốc, hoặc có từ hai ngày trở lên, thì trả
    rỗng: mốc lúc đó nằm trong dữ liệu hoặc lịch sử, và luật không đoán.

    past: lời các lượt user trước. Lời này không có ngày nào thì lấy ngày nói rõ duy nhất của các lượt trước làm mốc.

    Mỗi kết quả kèm các giờ nói trong cùng mệnh đề: "sáu giờ chiều hôm trước" ra (ngày trước mốc, ("18:00",)).
    Mệnh đề có ngày nói rõ riêng thì bỏ qua: ngày đó thắng."""
    dates = spoken_dates(text, now)
    if not dates and past and _DAYISH.search(" ".join(_toks(text))):
        dates = spoken_dates(past, now)  # "hôm đó" khi ngày nói ở lượt trước: "thứ sáu tuần sau…" rồi "…hôm đó"
    if len(dates) != 1:
        return []
    base = date.fromisoformat(dates[0])
    out = []
    for cl in clauses(text, preserve_case=True):
        if spoken_dates(cl, now):
            # "trước thứ sáu tuần này một ngày": mốc nằm ngay trong cụm dời của mệnh đề này
            m = _REL_N.search(cl.lower())
            inner = spoken_dates(m.group(0), now) if m else []
            if len(inner) == 1 and spoken_dates(cl, now) == inner:
                n = {"một": 1, "hai": 2, "ba": 3}.get(m.group(2)) or int(m.group(2))
                b = date.fromisoformat(inner[0])
                day = b + timedelta(days=(-1 if m.group(1) == "trước" else 1) * n)
                out.append(Anchored(day.isoformat(), (), f'"{m.group(0)}" tính từ {b.isoformat()}', b.isoformat(), True))
            continue
        cl = cl.lower()
        day, label = None, ""
        m = _REL_N.search(cl)
        if m and " ".join(m.group(0).split()[1:-2]) not in ("", "đó", "hôm đó", "ngày đó"):
            m = None  # "trước hạn khiếu nại một ngày": mốc là hạn trong dữ liệu, không phải ngày nói ra
        if m:
            n = {"một": 1, "hai": 2, "ba": 3}.get(m.group(2)) or int(m.group(2))
            sign = -1 if m.group(1) == "trước" else 1
            day, label = base + timedelta(days=sign * n), f'"{m.group(0)}" tính từ {base.isoformat()}'
        elif _REL_BEFORE.search(cl):
            day, label = base - timedelta(days=1), f"hôm trước của {base.isoformat()}"
        elif _REL_AFTER.search(cl):
            day, label = base + timedelta(days=1), f"hôm sau của {base.isoformat()}"
        elif _REL_SAME.search(cl):
            day, label = base, f"cùng ngày {base.isoformat()}"
        if day is not None:
            out.append(Anchored(day.isoformat(), tuple(F.spoken_clocks(_canon(cl))), label, base.isoformat()))
    return list(dict.fromkeys(out))


def datetimes(text: str, now: str, past: str = "") -> list[tuple[str, str]]:
    """Ngày giờ ghép trong cùng một mệnh đề: (ISO, cách đọc). "tám giờ sáng thứ hai tuần sau" ra ngày của thứ hai đó
    lúc 08:00. Mệnh đề chỉ có giờ không ghép, vì ngày có thể nằm ở mệnh đề khác hoặc trong dữ liệu."""
    out = []
    whole = spoken_dates(text, now)
    anchors = anchored(text, now, past)
    rel_clauses = {i for i, cl in enumerate(clauses(text))
                   if _REL_N.search(cl) or _REL_BEFORE.search(cl) or _REL_AFTER.search(cl) or _REL_SAME.search(cl)}
    today = datetime.fromisoformat(now).date().isoformat()
    cls = clauses(text, preserve_case=True)
    dropped = superseded([cl.lower() for cl in cls])
    for i, cl in enumerate(cls):
        ds, cs = spoken_dates(cl, now), F.spoken_clocks(_canon(cl))
        if _MIDNIGHT.search(cl.lower()):  # "nửa đêm", "mười hai giờ đêm": 00:00 của ngày hôm sau, không phải 12:00
            cs = [c for c in cs if c not in ("12:00", "00:00")] + ["24:00"]
        if not cs or i in dropped:
            continue
        cs = _prune_ampm(cs, now) if not ds and not whole else cs
        if len(ds) == 1:
            day, how = ds[0], f"{WEEKDAYS[date.fromisoformat(ds[0]).weekday()]} {ds[0]}"
        elif ds or i in rel_clauses or _DATA_TIME.search(cl.lower()):
            continue  # nhiều ngày, ngày theo mốc (anchored lo), hoặc giờ lấy từ dữ liệu
        elif len(whole) == 1:  # giờ ở mệnh đề này, ngày nói ở mệnh đề khác
            day, how = whole[0], f"{WEEKDAYS[date.fromisoformat(whole[0]).weekday()]} {whole[0]} (ngày nói ở vế khác)"
        elif not whole and not _DAYISH.search(" ".join(_toks(text))) and not _DATA_TIME.search(" ".join(_toks(text))):
            # chỉ nói giờ, không có chữ nào chỉ ngày: hôm nay (quy ước nhãn train: 8/8 mẫu chỉ nói giờ là hôm nay)
            day, how = today, f"hôm nay {today}"
        else:
            continue
        for c in cs:
            out.append((_iso(day, c, now), f"nửa đêm sau {how}" if c == "24:00" else f"{c} {how}"))
    for a in anchors:
        for c in a.clocks:
            out.append((_iso(a.day, c, now), f"{c} {a.label}"))
    return list(dict.fromkeys(out))


# chữ chỉ ngày mà spoken_dates có thể không đọc ra ("hai mốt tháng mười", "sáng hôm đó"): có thì không mặc định hôm nay
_DAYISH = re.compile(r"\b(hôm|bữa|ngày|tháng|thứ|tuần|mai|mốt|kia|chủ nhật)\b")
_CORRECTION = re.compile(r"^(?:(?:à|á|ờ|ơ|ừ|ừm)\s+(?:không|nhầm|thôi)|nhầm|thôi|không phải)\b")


def superseded(cls: list[str]) -> set[int]:
    """Mệnh đề bị câu tự sửa thay thế: "5 giờ nhé, à không, 5 rưỡi" bỏ "5 giờ nhé". Mệnh đề sửa là mệnh đề bắt đầu bằng
    "à không", "à nhầm", "à thôi"; mệnh đề bị bỏ là mệnh đề có nội dung ngay trước nó (bỏ qua chính cụm "à không")."""
    out = set()
    for i, cl in enumerate(cls):
        if not _CORRECTION.match(cl):
            continue
        k = i - 1
        while k >= 0 and _CORRECTION.fullmatch(cls[k].strip()):
            k -= 1
        if k >= 0:
            out.add(k)
    return out


def _prune_ampm(clocks: list[str], now: str) -> list[str]:
    """Giờ không nói buổi cho hai khả năng (07:30 và 19:30). Nếu hôm nay là ngày ghép, giữ khả năng sớm nhất còn ở
    phía trước bây giờ: lúc 08:00 nói "mười giờ" là 10:00, lúc 13:00 nói "ba giờ" là 15:00."""
    n0 = datetime.fromisoformat(now)
    now_hm = n0.strftime("%H:%M")
    out, i = [], 0
    while i < len(clocks):
        a = clocks[i]
        b = clocks[i + 1] if i + 1 < len(clocks) else None
        if b and int(b[:2]) == int(a[:2]) + 12 and a[3:] == b[3:]:
            ahead = [x for x in (a, b) if x > now_hm]
            out.append(ahead[0] if ahead else a)
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# giờ tính từ một mốc trong dữ liệu, không phải giờ đặt: "trước giờ họp một tiếng", "theo giờ đến", "giờ báo thức"
_DATA_TIME = re.compile(r"\b(trước|sau)\s+(giờ|lúc)\b|\btheo\s+giờ\b|\bgiờ\s+(báo thức|họp|đến|tới|bay|chạy)\b"
                        r"|\bkém\s+\S+\s+(giờ|tiếng)\s+(trước|sau)\b")


_UNIT_MIN = {"phút": 1, "tiếng": 60, "giờ": 60, "ngày": 1440, "tuần": 10080}


_SUFFIX = {"lăm", "mốt", "tư", "nhăm"}


def _garbled(span: list[str]) -> bool:
    """Cụm số đọc không theo luật, thường do ASR nhân đôi từ ("hai lăm lăm", "ba ba mươi"): cùng một từ số đứng liền
    nhau hai lần, hay hai từ đuôi ("lăm", "mốt", "tư") liền nhau. Cụm như vậy không đọc, để LLM tự quyết."""
    words = [w for w in span if not w[:1].isdigit()]
    return any(a == b for a, b in zip(words, words[1:])) or any(a in _SUFFIX and b in _SUFFIX for a, b in zip(words, words[1:]))


def _amount_before(toks: list[str], i: int) -> tuple[int, float] | None:
    """Cụm số đứng ngay trước token i: (vị trí đầu cụm, giá trị). "nửa tiếng" cho 0,5; "tiếng rưỡi" xử lý ở chỗ gọi."""
    k = i
    while k > 0 and N._isnum(toks[k - 1]) and toks[k - 1] != "rưỡi":
        k -= 1
    if k < i:
        span = toks[k:i]
        if _garbled(span):
            return None
        v = 10.0 if span == ["chục"] else N._value(span)  # "chục phút" là 10
        return (k, v) if v is not None else None
    if i and toks[i - 1] == "nửa":
        return i - 1, 0.5
    return None


def _duration_spans(toks: list[str]):
    """(start, end, minutes, unit), joining adjacent hours + minutes as one duration."""
    consumed = 0
    for i, w in enumerate(toks):
        if i < consumed or w not in _UNIT_MIN:
            continue
        rui = toks[i + 1:i + 2] == ["rưỡi"]
        got = _amount_before(toks, i)
        if got is None and w == "tiếng" and rui:
            got = (i, 1.0)
        if got is None or (got[0] and toks[got[0] - 1] == "thứ"):
            continue
        k, v = got
        end = i + 1 + rui
        minutes = (v + 0.5 * rui) * _UNIT_MIN[w]
        if w in ("giờ", "tiếng") and not rui:
            q = end + (toks[end:end + 1] == ["và"])
            j = q
            while j < len(toks) and N._isnum(toks[j]) and toks[j] != "rưỡi":
                j += 1
            if j > q and toks[j:j + 1] == ["phút"]:
                extra = N._value(toks[q:j])
                if extra is not None and 0 <= extra < 60:
                    minutes += extra
                    end = j + 1
        consumed = end
        yield k, end, minutes, w


def now_offsets(text: str, now: str) -> list[str]:
    """Elapsed duration from now: also handles compound hours + minutes and half an hour."""
    text = _canon(text)
    toks = _toks(text)
    n0 = datetime.fromisoformat(now)
    out = []
    for _, end, minutes, unit in _duration_spans(toks):
        if toks[end:end + 1] != ["nữa"] or not 0 < minutes <= 60 * 24 * 30:
            continue
        t = n0 + timedelta(minutes=minutes)
        out.append(t.date().isoformat() if unit in ("ngày", "tuần") else t.replace(microsecond=0).isoformat())
    return list(dict.fromkeys(out))


_SIGN = {"trước": -1, "sớm": -1, "sau": 1, "muộn": 1, "trễ": 1, "lùi": 1, "chậm": 1, "qua": 1}
# "sớm nhất", "muộn nhất" là phần tên mốc ("hạn sớm nhất"), không phải hướng dời


def offsets(text: str) -> list[Offset]:
    """Phép dời theo một mốc chưa nói ra: "trước hạn khiếu nại một ngày" (−1 ngày), "đón sớm bốn mươi lăm phút"
    (−45), "trước lúc xe tới mười lăm phút" (−15), "một tiếng sau" (+60).

    Bỏ qua "… nữa" (tính từ bây giờ, xem now_offsets) và "thứ ba tuần sau" (tên ngày)."""
    text = _canon(text)
    toks = _toks(text)
    out = []
    for i, w in enumerate(toks):
        if w not in _UNIT_MIN or w == "giờ":
            continue
        rui = i + 1 < len(toks) and toks[i + 1] == "rưỡi"
        after = toks[i + 1 + rui] if i + 1 + rui < len(toks) else ""
        if after == "nữa":
            continue
        got = _amount_before(toks, i)
        if got is None:
            continue
        k, v = got
        if k and toks[k - 1] in ("thứ", "tuần"):
            continue
        minutes = int((v + 0.5 * rui) * _UNIT_MIN[w])
        sign, start = None, k
        if after in ("trước", "sau"):
            sign, end = _SIGN[after], i + 2 + rui
        else:
            end = i + 1 + rui
            for q in range(k - 1, max(-1, k - 8), -1):
                if toks[q] in _PUNCT:
                    break
                if toks[q] in _SIGN and toks[q + 1:q + 2] != ["nhất"]:
                    sign, start = _SIGN[toks[q]], q
                    break
        if sign is None or minutes <= 0:
            continue
        out.append(Offset(sign * minutes, " ".join(toks[start:end])))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ số tiền, số lượng, phần trăm

_UNIT_WORDS = {"bước", "km", "cây", "lít", "người", "phần", "vé", "chiếc", "cái", "kwh", "calo", "kg", "gram", "ml"}
_CURRENCY_WORDS = {"đồng", "vnd", "vnđ", "đô", "usd", "eur", "yên", "jpy"}


def amounts(text: str) -> list[tuple[int | float, str]]:
    """Lượng lớn nói bằng chữ có hàng nghìn trở lên: (giá trị, đơn vị đi sau hoặc "").
    "năm mươi nghìn" -> (50000, ""), "một triệu rưỡi" -> (1500000, ""), "hai nghìn bước" -> (2000, "bước"),
    "50k" -> (50000, ""). Tiền có đơn vị rõ ("ba trăm đô") giữ cả lượng dưới 1.000.
    Lượng dưới 1.000 không có đơn vị tiền bỏ qua: "hai trăm" có thể là 200 hay 200.000,
    luật money_k lo chỗ đó."""
    text = _canon(text)
    ws = N._words(text)
    out = []
    for i, j in N._spans(ws):
        span = ws[i:j]
        if j < len(ws) and ws[j] in ("giờ", "phút", "tiếng", "ngày", "tháng", "tuần", "rưỡi"):
            continue
        scaled = any(w in N._SCALE for w in span) or (j < len(ws) and ws[j] in ("k", "nghìn", "ngàn"))
        v = N._value(span + ([ws[j]] if j < len(ws) and ws[j] == "k" and "k" not in span else []))
        currency = ws[j] if j < len(ws) and ws[j] in _CURRENCY_WORDS else ""
        if v is None or (not scaled and not currency) or v <= 0 or (v < 1000 and not currency) or len(span) > 8:
            continue
        unit = ws[j] if j < len(ws) and ws[j] in _UNIT_WORDS else currency
        if all(w in _DIGIT or w.isdigit() for w in span) and len(span) > 3:
            continue  # dãy đọc từng chữ số: số điện thoại, mã
        out.append((int(v) if v == int(v) else v, unit))
    return list(dict.fromkeys(out))


def percents(text: str) -> list[int | float]:
    """"30 phần trăm" -> 30, "ba mươi phần trăm" -> 30, "hai phẩy năm phần trăm" / "2,5 phần trăm" -> 2.5,
    "một nửa" / "phân nửa" -> 50."""
    ws = N._words(_canon(text))
    out = []
    for i in range(len(ws) - 1):
        if ws[i] == "phần" and ws[i + 1] == "trăm":
            got = _amount_before(ws, i)
            if not got:
                continue
            k, v = got
            if k >= 2 and ws[k - 1] in ("phẩy", "chấm"):  # phần thập phân đọc bằng chữ: "hai phẩy năm"
                a = _amount_before(ws, k - 1)
                if a is None or v != int(v):
                    continue
                v = a[1] + v / 10 ** len(str(int(v)))
            if 0 <= v <= 100:
                out.append(int(v) if v == int(v) else round(v, 4))
    if re.search(r"\b(một nửa|phân nửa)\b", " ".join(ws)):
        out.append(50)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ gộp và hiển thị

def facts(text: str, now: str, source: str = "top", past: str = "") -> list[Fact]:
    """Mọi giá trị đọc được từ một bản chép, theo thứ tự hiển thị."""
    if not text:
        return []
    text = _canon(text)
    out: list[Fact] = []
    for p in phones(text):
        out.append(Fact("phone", p, "số điện thoại", source))
    for s in phone_suffixes(text):
        out.append(Fact("phone_suffix", s, "số có đuôi", source))
    for d in spoken_dates(text, now):
        out.append(Fact("date", d, WEEKDAYS[date.fromisoformat(d).weekday()], source))
    for a in anchored(text, now, past):
        out.append(Fact("anchored", a.day, a.label, source))
    for iso, label in datetimes(text, now, past):
        out.append(Fact("datetime", iso, label, source))
    for iso in now_offsets(text, now):
        out.append(Fact("now_offset", iso, "tính từ bây giờ", source))
    for o in offsets(text):
        out.append(Fact("offset", o.minutes, o.label, source))
    for v, unit in amounts(text):
        out.append(Fact("amount", v, unit, source))
    for p in percents(text):
        out.append(Fact("percent", p, "phần trăm", source))
    used = {abs(o.minutes) for o in offsets(text)}
    for m in durations(text):
        if m not in used:
            out.append(Fact("duration", m, "phút", source))
    return out


def durations(text: str) -> list[int]:
    """Thời lượng phút / tiếng nói ra, bằng phút: "bốn lăm phút" 45, "tiếng rưỡi" 90, "nửa tiếng" 30.
    Khác numbers.durations: bỏ "thứ hai tuần sau" (tên ngày, không phải hai tuần), bỏ "… nữa" (tính từ bây giờ),
    và chỉ lấy phút, tiếng."""
    text = _canon(text)
    toks = _toks(text)
    out = []
    span = F.spoken_span(text)
    if span is not None:
        out.append(span)
    for _, end, minutes, unit in _duration_spans(toks):
        if unit not in ("phút", "tiếng") or toks[end:end + 1] == ["nữa"]:
            continue
        if 0 < minutes <= 24 * 60 and minutes == int(minutes):
            out.append(int(minutes))
    return list(dict.fromkeys(out))


def day_for_clock(text: str, now: str, past: str = "") -> dict[str, set[str]]:
    """Giờ nói ra -> các ngày mà lời ghép với giờ đó (cùng mệnh đề, kể cả ngày tính theo mốc như "hôm trước")."""
    out: dict[str, set[str]] = {}
    for iso, _ in datetimes(text, now, past):
        out.setdefault(iso[11:16], set()).add(iso[:10])
    return out


_ALT_KINDS = {"phone", "date", "anchored", "datetime", "now_offset", "amount"}


def merge(hyps: list[str], now: str, max_alt: int = 3, past: str = "") -> list[Fact]:
    """Gộp N-best: giá trị của bản đầu, cộng giá trị loại chắc chắn (số điện thoại, ngày giờ, số tiền) chỉ có ở bản
    nghe khác, đánh dấu "alt". Mỗi (loại, giá trị) chỉ một lần."""
    out = facts(hyps[0], now, past=past) if hyps else []
    have = {(f.kind, f.value) for f in out}
    for h in hyps[1:1 + max_alt]:
        for f in facts(h, now, "alt", past=past):
            if f.kind in _ALT_KINDS and (f.kind, f.value) not in have:
                out.append(f)
                have.add((f.kind, f.value))
    return out


def _dur_text(minutes: int) -> str:
    m = abs(minutes)
    if m % 10080 == 0:
        return f"{m // 10080} tuần"
    if m % 1440 == 0:
        return f"{m // 1440} ngày"
    if m % 60 == 0:
        return f"{m // 60} tiếng"
    if m > 60:
        return f"{m // 60} tiếng {m % 60} phút"
    return f"{m} phút"


def render(fs: list[Fact], limit: int = 14) -> str:
    """Khối chữ cho prompt. Rỗng nếu không có giá trị nào."""
    if not fs:
        return ""
    lines = []
    for f in fs[:limit]:
        alt = " (chỉ có ở bản nghe khác)" if f.source == "alt" else ""
        if f.kind == "phone":
            lines.append(f"- số điện thoại: {f.value}{alt}")
        elif f.kind == "phone_suffix":
            lines.append(f"- số có đuôi {f.value}{alt}")
        elif f.kind == "date":
            lines.append(f"- ngày nói ra: {f.value} ({f.label}){alt}")
        elif f.kind == "anchored":
            lines.append(f"- {f.label} = {f.value}{alt}")
        elif f.kind == "datetime":
            lines.append(f"- {f.label} = {f.value}{alt}")
        elif f.kind == "now_offset":
            lines.append(f"- tính từ bây giờ: {f.value}{alt}")
        elif f.kind == "offset":
            sign = "−" if int(f.value) < 0 else "+"
            lines.append(f'- dời theo mốc: "{f.label}" = mốc {sign} {_dur_text(int(f.value))} (mốc lấy từ dữ liệu){alt}')
        elif f.kind == "amount":
            lines.append(f"- số: {f.value}{(' ' + f.label) if f.label else ''}{alt}")
        elif f.kind == "percent":
            lines.append(f"- phần trăm: {f.value}{alt}")
        elif f.kind == "duration":
            lines.append(f"- thời lượng: {f.value} phút{alt}")
    head = ("GIÁ TRỊ MÁY ĐỌC TỪ LỜI NÓI (tính bằng luật từ bản chép và giờ hiện tại; chỉ dùng khi đúng ý người dùng, "
            "mốc trong dữ liệu vẫn phải đọc bằng tool):")
    return head + "\n" + "\n".join(lines)


def hint(transcript: dict, now: str) -> str:
    """Khối gợi ý cho một bản chép của tầng ASR ({"final", "alts": {"final": [...]}})."""
    hyps = [transcript.get("final", "")] + list((transcript.get("alts") or {}).get("final") or [])
    past = " ".join(str(v) for v in (transcript.get("turns") or {}).values())
    return render(merge([h for h in hyps if h], now, past=past))
