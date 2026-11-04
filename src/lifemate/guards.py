"""Tầng D: thực thi có bảo vệ quanh ToolEnv. Mỗi bảo vệ bật/tắt hoặc đổi chế độ trong config `guards:`.

    validate           true | false     env.validate trước env.call; call sai schema không vào bài nộp
    normalize          true | false     ép int, bổ sung giây và +07:00 cho ISO-8601
    calendar_duration  true | false     dời `start` mà thiếu `end` thì giữ nguyên thời lượng (gold 36/36)
    id_guard           off | warn | block   ID chưa thấy ở nguồn nào trong `id_sources`
    id_sources         [turn, history, utterance]   HTML gốc chỉ có [turn]
    id_harvest         html | all       html: chỉ khoá id, merchant_id; all: mọi khoá id / *_id
    dup_guard          off | warn | block   gọi lại y hệt một call đã chạy

Chế độ `warn` vẫn chạy call và chèn cảnh báo vào kết quả cho agent; `block` không chạy.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from . import data

DEFAULTS = {
    "validate": True,
    "normalize": True,
    "calendar_duration": True,
    "id_guard": "warn",
    "id_sources": ["turn", "history", "utterance"],
    "id_harvest": "all",
    "dup_guard": "warn",
}


class GuardedEnv:
    def __init__(self, env, split: str, sample: dict, transcript: str = "", **cfg):
        self.env = env
        self.split = split
        self.sample = sample
        self.cfg = {**DEFAULTS, **cfg}
        self.specs = data.tool_specs(split)
        self.allowed = set(data.installed_tools(sample))
        self.ids_turn: set[str] = set()
        self.ids_history: set[str] = set()
        self.events: dict[str, tuple[str, str]] = {}
        self.seen_calls: set[str] = set()
        self.log: list[dict] = []  # mỗi lần agent đề nghị một call: đề nghị, sau chuẩn hoá, quyết định, kết quả
        self.utterance = transcript
        for turn in sample.get("history", []):
            for r in turn.get("tool_results") or []:
                self._harvest(r, self.ids_history)
            for c in turn.get("tool_calls") or []:
                self._harvest(c.get("arguments"), self.ids_history)

    # -- thu ID và sự kiện đã thấy ------------------------------------------------------------
    def _harvest(self, obj, into: set):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and (k == "id" or k == "merchant_id" or (self.cfg["id_harvest"] == "all" and k.endswith("_id"))):
                    into.add(v)
                self._harvest(v, into)
            if isinstance(obj.get("id"), str) and "start" in obj and "end" in obj:
                self.events[obj["id"]] = (obj["start"], obj["end"])
        elif isinstance(obj, list):
            for x in obj:
                self._harvest(x, into)

    def known_ids(self) -> set[str]:
        src = self.cfg["id_sources"]
        out = set(self.ids_turn) if "turn" in src else set()
        if "history" in src:
            out |= self.ids_history
        return out

    # -- chuẩn hoá -----------------------------------------------------------------------------
    def normalize(self, name: str, args: dict) -> dict:
        clean = dict(args or {})
        props = self.specs.get(name, {}).get("parameters", {}).get("properties", {})
        if self.cfg["normalize"]:
            for p, s in props.items():
                if p not in clean:
                    continue
                v = clean[p]
                if s.get("type") == "integer" and isinstance(v, str) and re.fullmatch(r"-?[\d.,\s]+", v.strip()):
                    clean[p] = int(re.sub(r"[^\d-]", "", v))
                if isinstance(v, str) and "ISO" in s.get("description", ""):
                    v = v.strip()
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", v):
                        v += ":00+07:00"
                    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", v):
                        v += "+07:00"
                    clean[p] = v
        if self.cfg["calendar_duration"] and name == "calendar.update_event":
            ev = clean.get("event_id")
            if ev in self.events and "start" in clean and "end" not in clean:
                try:
                    s0, e0 = (datetime.fromisoformat(x) for x in self.events[ev])
                    clean["end"] = (datetime.fromisoformat(clean["start"]) + (e0 - s0)).isoformat()
                except ValueError:
                    pass
        return clean

    # -- một bước gọi tool ---------------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        rec = {"name": name, "proposed": args}
        clean = self.normalize(name, args)
        rec["arguments"] = clean
        warnings = []

        if self.cfg["validate"]:
            v = self.env.validate(name, clean)
            if not v.get("ok", False):
                rec.update(decision="chặn: sai schema", result={"ok": False, "error": v.get("error")})
                self.log.append(rec)
                return {"ok": False, "error": v.get("error"), "note": "call chưa chạy, không tính vào bài nộp"}

        sig = json.dumps([name, clean], sort_keys=True, ensure_ascii=False)
        if self.cfg["dup_guard"] != "off" and sig in self.seen_calls:
            msg = "call này với cùng tham số đã chạy rồi; dùng lại kết quả trước nếu không cố ý gọi lần nữa"
            if self.cfg["dup_guard"] == "block":
                rec.update(decision="chặn: gọi trùng", result={"ok": False, "error": msg})
                self.log.append(rec)
                return {"ok": False, "error": msg, "note": "call chưa chạy"}
            warnings.append(msg)

        if self.cfg["id_guard"] != "off" and name != "sys.ask_missing_info":
            known = self.known_ids()
            spoken = data_fold(self.utterance) if "utterance" in self.cfg["id_sources"] else ""
            for k, v in clean.items():
                if not (k.endswith("_id") and isinstance(v, str)):
                    continue
                if v in known or (spoken and data_fold(v) in spoken):
                    continue
                if self.cfg["id_sources"] == ["turn"] and not known:
                    continue  # hành vi HTML gốc: chưa đọc gì thì cho qua
                msg = f"{k}='{v}' chưa xuất hiện trong kết quả đọc hay lịch sử; kiểm tra lại nguồn của ID"
                if self.cfg["id_guard"] == "block":
                    rec.update(decision="chặn: ID chưa thấy", result={"ok": False, "error": msg})
                    self.log.append(rec)
                    return {"ok": False, "error": msg, "note": "call chưa chạy"}
                warnings.append(msg)

        obs = self.env.call(name, clean)
        self.seen_calls.add(sig)
        self._harvest(obs, self.ids_turn)
        rec.update(decision="chạy" + (" (cảnh báo)" if warnings else ""), result=obs, warnings=warnings)
        self.log.append(rec)
        if warnings:
            obs = {**obs, "warnings": warnings}
        return obs

    def validate(self, name, args):
        return self.env.validate(name, self.normalize(name, args))


def data_fold(s) -> str:
    from .scoring import fold
    return fold(s or "")
