"""Tầng D: thực thi có bảo vệ quanh ToolEnv. Mỗi bảo vệ bật/tắt hoặc đổi chế độ trong config `guards:`.

    validate           true | false     env.validate trước env.call; call sai schema không vào bài nộp
    normalize          true | false     ép int, bổ sung giây và +07:00 cho ISO-8601
    calendar_duration  true | false     dời `start` mà thiếu `end` thì giữ nguyên thời lượng (gold 36/36)
    id_guard           off | warn | block   ID chưa thấy ở nguồn nào trong `id_sources`
    id_sources         [turn, history, utterance]   HTML gốc chỉ có [turn]
    id_harvest         html | all       html: chỉ khoá id, merchant_id; all: mọi khoá id / *_id
    dup_guard          off | warn | block   gọi lại y hệt một call đã chạy
    clock_fix          true | false     ngày giờ ISO lệch ngày / giờ nói ra (sai sáng chiều, "9 giờ kém 15 tối") (fixes.py)
    money_k            true | false     số tiền VND dưới 1.000 nhân 1.000 ("hai trăm", ASR mất "nghìn")
    span_min           true | false     tham số phút = hiệu hai giờ trong "từ 7 rưỡi tới 8 giờ 10"
    phone_fix          true | false     số điện thoại đọc từng chữ số: lấy số nói ra (normalize.phones)
    rel_day_fix        true | false     "sáu giờ chiều hôm trước" khi mốc là "thứ sáu này": ngày ghép với giờ cùng mệnh đề
    now_offset_fix     true | false     "bốn lăm phút nữa": bây giờ + thời lượng khi lời không nói giờ nào
    ask_missing        true | false     số tiền "chút tiền" hay giờ hẹn không có nguồn: hỏi lại thay vì chạy
    pick_one           true | false     "hủy bớt một gói thôi": hỏi lại mục nào thay vì chọn bừa

Câu hỏi lại của ask_missing, pick_one được hoãn tới `finish()` (runner gọi khi agent xong), để agent làm nốt các phần
đủ thông tin như luật 9 của prompts/rules/verified.md; gửi ngay thì môi trường trả `waiting_for_user` và agent dừng luôn.
Call bị hoãn trả `ok: true` kèm ghi chú: trả `ok: false` thì agent cũng dừng (train_0352, public_0033).

Chế độ `warn` vẫn chạy call và chèn cảnh báo vào kết quả cho agent; `block` không chạy.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from . import data, fixes

DEFAULTS = {
    "validate": True,
    "normalize": True,
    "calendar_duration": True,
    "id_guard": "warn",
    "id_sources": ["turn", "history", "utterance"],
    "id_harvest": "all",
    "dup_guard": "warn",
    "clock_fix": False,
    "money_k": False,
    "span_min": False,
    "phone_fix": False,
    "rel_day_fix": False,
    "now_offset_fix": False,
    "ask_missing": False,
    "pick_one": False,
}
FIX_RULES = ("clock_fix", "money_k", "span_min", "phone_fix", "rel_day_fix", "now_offset_fix")


class GuardedEnv:
    def __init__(self, env, split: str, sample: dict, transcript: str = "", turns: dict | None = None, **cfg):
        """turns: bản chép các lượt user trước (ASR); không có thì lấy lời gold trong history."""
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
        self.now = sample["context"]["current_time"]
        past = list(turns.values()) if turns is not None else [t.get("audio_text", "") for t in sample.get("history", []) if t.get("role") == "user"]
        self.said_all = " ".join([transcript, *past])
        self.past_text = " ".join(past)
        self.seen: set = set()  # mọi chuỗi, số đã thấy: giá trị có nguồn thì luật fixes không đụng
        self.asked: set[str] = set()  # tool đã hỏi lại (do luật hay do LLM tự gọi sys.ask_missing_info)
        self.llm_asked: set[str] = set()
        self.pending: list[dict] = []  # câu hỏi lại của luật, gửi ở finish()
        self.picked = False
        for turn in sample.get("history", []):
            for r in turn.get("tool_results") or []:
                self._harvest(r, self.ids_history)
            for c in turn.get("tool_calls") or []:
                self._harvest(c.get("arguments"), self.ids_history)
            fixes.strings([turn.get("tool_results"), turn.get("tool_calls"), turn.get("text")], self.seen)

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
        self.fix_notes = []
        if any(self.cfg[r] for r in FIX_RULES) and self.specs.get(name, {}).get("kind") == "write":
            clean, self.fix_notes = fixes.fix_args(name, clean, self.specs[name], self.utterance, self.now, self.seen,
                                                   {r: self.cfg[r] for r in FIX_RULES}, past=self.past_text)
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
        if self.fix_notes:
            rec["fixes"] = self.fix_notes
        warnings = []

        if self.cfg["validate"]:
            v = self.env.validate(name, clean)
            if not v.get("ok", False):
                rec.update(decision="chặn: sai schema", result={"ok": False, "error": v.get("error")})
                self.log.append(rec)
                return {"ok": False, "error": v.get("error"), "note": "call chưa chạy, không tính vào bài nộp"}

        ask = self._ask_instead(name, clean)
        if ask:  # trả ok: true, vì gặp ok: false agent SFT dừng luôn mà bỏ dở các phần còn lại
            if name in self.asked:  # LLM gọi lại call đã được đổi thành câu hỏi: không chạy, không hỏi lần hai
                obs = {"ok": True, "note": f"đã ghi câu hỏi lại về {', '.join(ask)} của {name}; không gọi lại tool này"}
                rec.update(decision="chặn: đã hỏi lại", result=obs)
                self.log.append(rec)
                return obs
            self.asked.add(name)
            self.pending.append({"tool_name": name, "params": ask})
            obs = {"ok": True, "note": f"người dùng chưa nói {', '.join(ask)} cho {name}: call chưa chạy, câu hỏi lại sẽ "
                                       "được gửi ở cuối; làm tiếp các phần còn lại của yêu cầu"}
            rec.update(decision=f"hoãn: hỏi lại {', '.join(ask)} ở cuối", result=obs)
            self.log.append(rec)
            return obs
        if name == "sys.ask_missing_info" and isinstance(clean.get("tool_name"), str):
            self.asked.add(clean["tool_name"])
            self.llm_asked.add(clean["tool_name"])

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
        fixes.strings([obs, clean], self.seen)
        rec.update(decision="chạy" + (" (cảnh báo)" if warnings else ""), result=obs, warnings=warnings)
        self.log.append(rec)
        if warnings:
            obs = {**obs, "warnings": warnings}
        return obs

    def finish(self):
        """Gửi các câu hỏi lại đã hoãn, trừ khi LLM đã tự hỏi về cùng tool. Gọi một lần khi agent xong."""
        for q in self.pending:
            if q["tool_name"] in self.llm_asked:
                continue
            obs = self.env.call("sys.ask_missing_info", q)
            self.log.append({"name": "sys.ask_missing_info", "proposed": q, "arguments": q,
                             "decision": "chạy (hỏi lại đã hoãn)", "result": obs, "warnings": []})
        self.pending = []

    def _ask_instead(self, name: str, clean: dict) -> list[str]:
        spec = self.specs.get(name, {})
        if name == "sys.ask_missing_info" or spec.get("kind") != "write":
            return []
        if self.cfg["pick_one"] and not self.picked:
            ask = fixes.pick_one(name, clean, spec, self.utterance)
            if ask:
                self.picked = True
                return ask
        if self.cfg["ask_missing"]:
            return fixes.missing(name, clean, spec, self.said_all, self.now, self.seen, final=self.utterance)
        return []

    def validate(self, name, args):
        return self.env.validate(name, self.normalize(name, args))


def data_fold(s) -> str:
    from .scoring import fold
    return fold(s or "")
