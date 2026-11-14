"""Chọn đúng một mục trong kết quả đọc theo lời nói, cho luật `ask_resolve` của GuardedEnv (LM-47).

Lỗi lớn nhất của 195 trên app chưa thấy là "làm → hỏi lại": model đã đọc đúng danh sách nguồn của một tham số ID rồi lại
hỏi người dùng ID đó, dù lời nói gọi tên đúng một mục. `pick(items, said)` trả mục đó khi chắc chắn, None khi không:

- chữ của một mục: mọi giá trị chuỗi của nó trừ ID, ngày giờ, địa chỉ email, đường dẫn; tách từ, bỏ dấu, bỏ hoa thường;
- chữ phân biệt: chữ của mục đó mà không phải mọi mục đều có, không thuộc nhóm từ chức năng; khớp khi có trong lời;
- điểm một mục = số chữ phân biệt khớp, cộng số cặp chữ liền nhau của mục có nguyên văn trong lời;
- chọn khi mục điểm cao nhất đạt `min_score`, hơn mục thứ nhì ít nhất `margin`, và không mục nào khác có cùng tên.

Ngưỡng chọn bằng `scripts/resolve_check.py` trên call gold của train_rest (cần đúng ≥ 95% khi chọn): ID gold lấy từ kết
quả đọc thì phải chọn đúng ID đó hoặc không chọn; mẫu gold hỏi lại ID sau khi đọc thì không được chọn.
"""
from __future__ import annotations

import re

from .scoring import tokens

STOP = set("""
anh chị em tôi tớ mình bạn cô chú bác ông bà con cháu mẹ bố ba má nó họ ta tui ổng bả
giúp giùm dùm hộ cho với và rồi thì là mà nhé nha nghen hen nhá ạ ơi đi luôn nữa đó đấy ấy này kia nọ nớ ni
cái chiếc con cuốn quyển gói đơn lịch việc buổi lần cuộc phòng mục món tờ bản bộ những các mấy một hai ba
của trong ngoài trên dưới tại ở lúc khi nào gì sao vậy thế đã đang sẽ vừa mới còn không chưa có được bị
xem coi kiểm tra đặt tạo thêm xoá xóa huỷ hủy đổi sửa gọi nhắn gửi mở tắt bật ghi lấy giữ vào ra lên xuống
giờ phút ngày hôm tuần tháng năm sáng trưa chiều tối đêm mai nay qua sau trước
""".split())
_SKIP = re.compile(r"^\d{4}-\d{2}-\d{2}|@|^https?://|^[a-z]{1,12}_[a-z0-9_]*\d[a-z0-9_]*$|^[A-Z]{2,6}\d{5,}$")


NAME_KEYS = ("name", "title", "label", "subject")


def item_words(item: dict, keys: tuple | None = None) -> list[str]:
    """Chữ của một mục; keys: chỉ lấy các trường có tên chứa một trong các chuỗi đó (vd. NAME_KEYS)."""
    out = []
    for k, v in item.items():
        if k == "id" or k.endswith("_id") or not isinstance(v, str) or _SKIP.search(v.strip()):
            continue
        if keys and not any(x in k for x in keys):
            continue
        out += tokens(v)
    return out


def _bigrams(ws: list[str]) -> set[str]:
    return {f"{a} {b}" for a, b in zip(ws, ws[1:])}


def scores(items: list[dict], said: str, keys: tuple | None = None) -> list[float]:
    words = [item_words(x, keys) for x in items]
    sets = [set(w) for w in words]
    common = set.intersection(*sets) if sets else set()
    said_ws = tokens(said)
    said_set, said_bi = set(said_ws), _bigrams(said_ws)
    out = []
    for ws, s in zip(words, sets):
        distinct = {w for w in s - common - STOP if len(w) >= 2 or w.isdigit()}
        hit = distinct & said_set
        bi = {b for b in _bigrams(ws) if b in said_bi and not set(b.split()) <= STOP | common}
        out.append(len(hit) + len(bi))
    return out


def pick(items: list[dict], said: str, min_score: int = 2, margin: int = 2, exclusive: bool = True,
         keys: tuple | None = None) -> dict | None:
    """Mục duy nhất mà lời nói gọi tới, hoặc None. exclusive: không mục nào khác được khớp chữ nào (lời nói nhắc đúng
    một mục); keys: chỉ chấm trên các trường tên (NAME_KEYS) thay vì mọi trường chuỗi."""
    items = [x for x in items if isinstance(x, dict) and isinstance(x.get("id"), str)]
    if not items:
        return None
    sc = scores(items, said, keys)
    best = max(range(len(items)), key=lambda i: sc[i])
    rest = [s for i, s in enumerate(sc) if i != best]
    if sc[best] < min_score or (rest and sc[best] - max(rest) < margin) or (exclusive and rest and max(rest) > 0):
        return None
    name = sorted(item_words(items[best], keys))
    if any(sorted(item_words(x, keys)) == name for i, x in enumerate(items) if i != best):
        return None
    return items[best]


def lists_with_ids(obj) -> list[list[dict]]:
    """Mọi danh sách mục có ID trong một kết quả tool."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) and isinstance(x.get("id"), str) for x in v):
                out.append(v)
            elif isinstance(v, (dict, list)):
                out += lists_with_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            out += lists_with_ids(v)
    return out
