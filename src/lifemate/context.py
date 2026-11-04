"""Tầng B: dựng ngữ cảnh bằng mã tất định. Mỗi khối có nhiều chế độ, chọn trong `agent.context`.

    timetable   none | three_days (HTML) | two_weeks (proposal)
    schema      json (schema thô) | compact (mỗi tool một dòng)
    history     full (lời user + call + kết quả) | brief (chỉ lời và tên call)
    ledger      true | false   bảng ID đã biết từ lịch sử
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import data

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def timetable(current_time: str, mode: str = "two_weeks") -> str:
    if mode == "none":
        return ""
    now = datetime.fromisoformat(current_time)
    d0 = now.date()
    line = lambda label, d: f"- {label}: {WEEKDAYS[d.weekday()]} {d.isoformat()}"
    rows = [f"Bây giờ: {now.isoformat()} ({WEEKDAYS[d0.weekday()]})"]
    if mode == "three_days":
        for label, k in [("Hôm qua", -1), ("Hôm nay", 0), ("Ngày mai", 1), ("Ngày kia", 2)]:
            rows.append(line(label, d0 + timedelta(days=k)))
        return "\n".join(rows)
    names = {-1: "Hôm qua", 0: "Hôm nay", 1: "Ngày mai", 2: "Ngày mốt / ngày kia"}
    monday = d0 - timedelta(days=d0.weekday())
    for k in range(-1, 14):
        d = d0 + timedelta(days=k)
        week = "tuần này" if d < monday + timedelta(days=7) else "tuần sau" if d < monday + timedelta(days=14) else "tuần sau nữa"
        label = names.get(k, f"{WEEKDAYS[d.weekday()]} {week}")
        rows.append(line(label, d) + (f" ({WEEKDAYS[d.weekday()]} {week})" if k in names else ""))
    sat = monday + timedelta(days=5)
    rows.append(f"- Cuối tuần này: {sat.isoformat()} đến {(sat + timedelta(days=1)).isoformat()}")
    rows.append("Định dạng: ngày YYYY-MM-DD; giờ YYYY-MM-DDTHH:MM:00+07:00")
    return "\n".join(rows)


def timetable_dates(current_time: str) -> list[str]:
    """Các ngày bảng thời gian two_weeks liệt kê, hôm nay, mai, hôm qua trước."""
    d0 = datetime.fromisoformat(current_time).date()
    return [(d0 + timedelta(days=k)).isoformat() for k in [0, 1, -1] + list(range(2, 14))]


def _ptype(s: dict) -> str:
    t = s.get("type", "any")
    if "enum" in s:
        t = "|".join(map(str, s["enum"]))
    if t == "array" and isinstance(s.get("items"), dict):
        t = f"[{s['items'].get('type', 'object')}]"
    return t


def schema(split: str, sample: dict, mode: str = "compact", specs: dict | None = None) -> str:
    """specs: catalog thay thế (vd. bản đã đổi tên của scripts/build_sft.py); mặc định là catalog của split."""
    specs = specs or data.tool_specs(split)
    tools = [specs[n] for n in data.installed_tools(sample) if n in specs]
    if mode == "json":
        keep = ("name", "kind", "description", "parameters")
        return "\n".join(json.dumps({k: t[k] for k in keep if k in t}, ensure_ascii=False) for t in tools)
    lines = []
    for t in tools:
        params = t.get("parameters", {})
        req = set(params.get("required", []))
        parts = []
        for p, s in params.get("properties", {}).items():
            mark = "" if p in req else "?"
            extra = []
            if s.get("x-ref"):
                extra.append(f"id từ {s['x-ref']}")
            if s.get("x-match") == "fuzzy":
                extra.append("văn bản tự do")
            desc = s.get("description", "")
            parts.append(f"{p}{mark}: {_ptype(s)}" + (f" ({'; '.join(extra)})" if extra else "") + (f" — {desc}" if desc else ""))
        oneof = params.get("x-one-of")
        head = f"{t['name']} [{t['kind']}] {t.get('description', '')}"
        if oneof:
            head += f" | bắt buộc một trong: {oneof}"
        lines.append(head + ("\n    " + "\n    ".join(parts) if parts else ""))
    return "\n".join(lines)


def history(sample: dict, user_texts: dict[str, str], mode: str = "full") -> str:
    """user_texts: {"u1": ..., "u2": ...} lấy từ tầng ASR (hoặc audio_text gold)."""
    out, k = [], 0
    for turn in sample.get("history", []):
        if turn.get("role") == "user":
            k += 1
            out.append(f"USER: {user_texts.get(f'u{k}', '(không có bản chép)')}")
            continue
        out.append(f"ASSISTANT: {turn.get('text', '')}")
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        for i, c in enumerate(calls):
            if mode == "brief":
                out.append(f"  call {c['name']}")
                continue
            out.append(f"  call {c['name']} {json.dumps(c.get('arguments', {}), ensure_ascii=False)}")
            if i < len(results):
                out.append(f"  -> {json.dumps(results[i], ensure_ascii=False)[:1200]}")
    return "\n".join(out)


def ledger(sample: dict) -> str:
    """Các thực thể có ID xuất hiện trong lịch sử (proposal 4.B): dùng lại thay vì đọc lại."""
    seen: dict[str, str] = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("id"), str):
                label = o.get("name") or o.get("title") or o.get("label") or o.get("content") or ""
                seen.setdefault(o["id"], str(label)[:60])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for turn in sample.get("history", []):
        walk(turn.get("tool_results") or [])
    return "\n".join(f"- {i}: {lab}" for i, lab in seen.items())


def profile(sample: dict) -> str:
    p = sample["context"].get("user_profile", {})
    return f"Tên: {p.get('name')} | Thành phố: {p.get('city')} | Nhà: {p.get('home_address')} | Công ty: {p.get('work_address')}"
