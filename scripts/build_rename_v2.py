"""RenamerV2: renamed COPIES of production SFT rows (Hammer-style function masking, arXiv 2410.04587), additive only.

Why a v2: the old `Renamer` (fable-eval/lm51-asr/scripts/build_sft.py:61-195, flag --rename) measured -5.2/-2.6 on
Qwen3 because its rows contradicted themselves (argon-lm85-code-audit.md, Renamer section; DISTIL_1115 appendix):
tool results kept the old keys, rules kept the old param names, apps became nonsense syllables, `_respell` glued
synonyms piecewise (start_time -> from_at), a regex rewrote ordinary words in descriptions, and it replaced clean rows
in place. This builder fixes each of those:

* one rename map per copy (app prefix, tool short name, field name), drawn from curated semantic synonym tables
  (English, some Vietnamese ASCII), whole-name, never piecewise;
* a field rename is global inside the copy: schema param, `required`, `x-one-of`, descriptions and rules that mention
  it as an identifier, call arguments (top level), result keys at every depth (history and current turn), and the
  `params` list of sys.ask_missing_info, so a read result key still matches the write param it feeds;
* text is rewritten by one identifier-token pass (dotted names first, then bare field names); fields that also occur
  as plain words in the catalogue text (email, merchant, video), fields used only as nested array-item keys (not shown
  by the compact schema), and sys fields are never renamed; enum values, ID values, timetable, profile and every user
  or assistant utterance stay byte-identical; the ledger is rendered from the original sample (it shows only IDs and
  values);
* rows are rendered with the same lifemate code as production (`ReactJSON.system_prompt` pieces, `context.schema`,
  `context.history`, `agent._obs`), and every source row must first survive an identity round-trip (empty map ->
  byte-identical row) or it is skipped;
* copies are added next to the originals (--ratio x input rows by default), originals are never touched;
  only gold-derived rows (train_*, *_valasr) and D1 (teach_p_*) are renamed, synth_* never.

    PYTHONPATH=/path/to/workspace/fable-work/pro-rules/src \
    LIFEMATE_DATA=/path/to/workspace/data/lifemate \
    /path/to/workspace/fable-work/pro-rules/.venv/bin/python build_rename_v2.py \
        --inp ../pro-threshold/sft/train_para_v1_asrfold.jsonl --out ../pro-threshold/sft/train_para_v1_asrfold_ren2.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PRO = Path(__file__).resolve().parents[1]
if str(PRO / "src") not in sys.path:
    sys.path.insert(0, str(PRO / "src"))

from lifemate import config, context, data  # noqa: E402
from lifemate.agent.agents import CALC, _render  # noqa: E402
from lifemate.registry import build  # noqa: E402

SPLIT = "train"
ASK = "sys.ask_missing_info"
KEEP_APPS = {"sys"}
# never renamed: sys fields (fixed at test time); fields that are also plain words in catalogue descriptions
NEVER_FIELD = {"tool_name", "params", "expression", "email", "merchant", "video"}
TEXT_VALUE_KEYS = {"message", "error"}  # result string values that may name a field or tool ("không tìm thấy chat_id")

APP_SYN = {
    "smart_home": "home_control|smarthome|nha_thong_minh", "home_security": "security_cam|home_guard|an_ninh",
    "door_lock": "smart_lock|khoa_cua|lock_manager", "robot_vacuum": "vacuum_bot|cleaning_robot|robot_hut_bui",
    "appliances": "home_appliances|household_devices|thiet_bi_gia_dung", "home_energy": "energy_monitor|power_usage|dien_nang",
    "alarm": "alarm_clock|bao_thuc|wake_up", "reminders": "reminder|nhac_viec|todo_reminders",
    "calendar": "agenda|lich|planner", "notes": "notebook|ghi_chu|memo", "tasks": "todo|cong_viec|todo_list",
    "contacts": "phonebook|danh_ba|address_book", "phone": "dialer|dien_thoai|calls", "sms": "text_messages|tin_nhan|messaging",
    "chat_ott": "messenger|chat_app|nhan_tin_ott", "email": "mail|hop_thu|mailbox",
    "video_meeting": "video_call|hop_truc_tuyen|meetings", "banking": "bank|ngan_hang|mobile_banking",
    "ewallet": "wallet|vi_dien_tu|e_wallet", "bill_payment": "bills|thanh_toan_hoa_don|utility_pay",
    "loyalty": "rewards|tich_diem|membership", "weather": "forecast|thoi_tiet|weather_info",
    "gold_price": "gold_rates|gia_vang", "fuel_price": "petrol_price|gia_xang", "lottery": "lotto|xo_so",
    "exchange_rate": "forex|ty_gia|currency", "news": "newsfeed|tin_tuc|press", "calculator": "math_tools|may_tinh",
    "maps": "navigation|ban_do|map", "food_delivery": "food_order|giao_do_an|meal_delivery",
    "ride_hailing": "taxi|goi_xe|ride_booking", "flights": "air_travel|ve_may_bay|airline",
    "shopping_list": "grocery_list|danh_sach_mua_sam", "package_tracking": "parcels|theo_doi_don|delivery_tracking",
    "music": "music_player|nghe_nhac|songs", "tv": "television|smart_tv|ti_vi",
    "podcast_radio": "audio_streams|radio_podcast|phat_thanh", "medication": "medicine|thuoc|pill_reminder",
    "fitness": "workout|the_duc|health_tracker", "learning": "courses|hoc_tap|e_learning",
    "bike_rental": "bike_share|thue_xe_dap", "home_automation": "automation|tu_dong_hoa|smart_rules",
    "household_chores": "chores|viec_nha", "library": "books|thu_vien", "subscriptions": "subscription_manager|goi_dang_ky",
}
TOOL_SYN = {
    "smart_home.get_status": "get_devices|device_status|read_state", "smart_home.light_set": "set_light|control_light",
    "smart_home.fan_set": "set_fan|control_fan", "smart_home.climate_set": "set_ac|set_climate|control_aircon",
    "smart_home.switch_set": "set_switch|toggle_switch", "smart_home.scene_run": "run_scene|activate_scene",
    "home_security.list_alerts": "get_alerts|show_alerts", "home_security.get_camera": "view_camera|camera_feed",
    "home_security.arm": "arm_system|set_arming", "home_security.acknowledge_alert": "dismiss_alert|ack_alert",
    "door_lock.list_locks": "get_locks|show_locks", "door_lock.set_lock": "lock_control|set_lock_state",
    "door_lock.grant_guest_code": "create_guest_code|issue_guest_pin",
    "robot_vacuum.get_status": "get_state|check_status", "robot_vacuum.start": "start_cleaning|begin_clean",
    "robot_vacuum.stop": "stop_cleaning|halt",
    "appliances.list": "list_appliances|get_appliances", "appliances.start_cycle": "run_cycle|begin_cycle",
    "appliances.stop": "stop_cycle|halt",
    "home_energy.get_usage": "get_consumption|energy_usage", "home_energy.set_alert": "set_usage_alert|set_threshold",
    "alarm.list": "list_alarms|get_alarms", "alarm.create": "add|set_alarm", "alarm.update": "edit|change_alarm",
    "alarm.delete": "remove|delete_alarm",
    "reminders.list": "list_reminders|get_all", "reminders.create": "add|add_reminder", "reminders.complete": "mark_done|finish",
    "reminders.delete": "remove|erase",
    "calendar.list_events": "get_events|show_events", "calendar.create_event": "add_event|new_event",
    "calendar.update_event": "edit_event|modify_event", "calendar.cancel_event": "delete_event|remove_event",
    "notes.list": "list_notes|get_notes", "notes.create": "add_note|new_note", "notes.append": "append_text|add_to_note",
    "tasks.list": "list_tasks|get_tasks", "tasks.add": "create|new_task", "tasks.complete": "mark_done|finish_task",
    "contacts.search": "find|lookup", "contacts.get": "get_contact|details", "contacts.update": "edit|update_contact",
    "phone.call": "dial|make_call", "phone.recent_calls": "call_log|call_history",
    "sms.send": "send_sms|send_text", "sms.read_recent": "recent_messages|read_messages",
    "chat_ott.list_chats": "get_chats|conversations", "chat_ott.read_chat": "open_chat|view_chat",
    "chat_ott.send_message": "post_message|send_text", "chat_ott.send_location": "share_location|send_position",
    "chat_ott.call": "voice_call|start_call",
    "email.list": "inbox|list_emails", "email.read": "open|read_email", "email.send": "send_email|deliver",
    "email.create_draft": "save_draft|new_draft", "email.reply": "respond|reply_email", "email.archive": "archive_email|move_to_archive",
    "video_meeting.list_meetings": "get_meetings|upcoming", "video_meeting.create": "schedule|new_meeting",
    "video_meeting.join": "join_meeting|enter",
    "banking.get_balance": "balance|check_balance", "banking.list_transactions": "transactions|get_transactions",
    "banking.list_beneficiaries": "payees|get_beneficiaries", "banking.transfer": "send_money|make_transfer",
    "banking.open_savings": "create_savings|open_deposit",
    "ewallet.get_balance": "balance|check_balance", "ewallet.transfer": "send_money|pay_contact",
    "ewallet.topup_mobile": "mobile_topup|recharge_phone", "ewallet.pay_qr": "qr_payment|scan_pay",
    "bill_payment.list_bills": "get_bills|unpaid_bills", "bill_payment.pay": "pay_bill|settle",
    "bill_payment.schedule_autopay": "set_autopay|enable_autopay",
    "loyalty.list_vouchers": "get_vouchers|show_coupons", "loyalty.get_points": "check_points|point_balance",
    "loyalty.redeem": "use_points|claim_reward",
    "weather.get_current": "current|current_conditions", "weather.get_forecast": "forecast|daily_forecast",
    "gold_price.get_price": "price|get_rates", "fuel_price.get_price": "price|get_rates",
    "lottery.get_results": "results|draw_results", "lottery.check_ticket": "verify_ticket|ticket_check",
    "exchange_rate.get_rate": "rate|current_rate", "exchange_rate.convert": "convert_amount|exchange",
    "news.headlines": "top_stories|latest", "news.search": "find_articles|search_news", "news.save": "bookmark|save_article",
    "calculator.calculate": "compute|evaluate", "calculator.convert_unit": "unit_convert|change_unit",
    "maps.search_place": "find_place|lookup_place", "maps.get_route": "directions|route", "maps.save_place": "bookmark_place|add_favorite",
    "food_delivery.list_restaurants": "restaurants|find_restaurants", "food_delivery.get_menu": "menu|show_menu",
    "food_delivery.order_history": "past_orders|my_orders", "food_delivery.place_order": "create_order|checkout",
    "food_delivery.cancel_order": "void_order|revoke_order",
    "ride_hailing.get_quotes": "quotes|estimate_fare", "ride_hailing.book": "book_ride|request_ride",
    "ride_hailing.cancel": "cancel_ride|abort_ride", "ride_hailing.get_ride": "ride_status|current_ride",
    "flights.search": "find_flights|search_flights", "flights.book": "book_flight|reserve", "flights.my_bookings": "bookings|my_trips",
    "flights.cancel": "cancel_booking|cancel_flight", "flights.check_in": "checkin|online_checkin",
    "shopping_list.get": "show|view_list", "shopping_list.add": "add_item|insert", "shopping_list.remove": "remove_item|drop",
    "shopping_list.check": "tick|mark_bought",
    "package_tracking.track": "track_parcel|where_is", "package_tracking.reschedule_delivery": "change_delivery|move_delivery",
    "package_tracking.schedule_pickup": "book_pickup|request_pickup",
    "music.list_tracks": "library|get_tracks", "music.play": "play_music|stream", "music.control": "playback|player_control",
    "music.add_to_playlist": "add_track|save_to_playlist",
    "tv.guide": "program_guide|tv_schedule", "tv.tune": "change_channel|set_channel", "tv.control": "remote|control_tv",
    "tv.record": "record_show|schedule_recording",
    "podcast_radio.search": "find|lookup", "podcast_radio.play": "listen|play_audio", "podcast_radio.subscribe": "follow|subscribe_show",
    "podcast_radio.queue": "add_to_queue|enqueue",
    "medication.list": "list_meds|get_schedule", "medication.add": "add_medicine|new_schedule",
    "medication.log_taken": "mark_taken|record_dose", "medication.remove": "remove_schedule|delete",
    "fitness.get_activity": "activity_summary|daily_stats", "fitness.log_workout": "record_workout|add_workout",
    "fitness.set_goal": "set_target|update_goal",
    "learning.list_courses": "my_courses|get_courses", "learning.enroll": "join_course|register",
    "learning.schedule_lesson": "book_lesson|plan_lesson", "learning.log_progress": "record_progress|update_progress",
    "bike_rental.find_bikes": "nearby_bikes|search_bikes", "bike_rental.reserve": "book_bike|rent",
    "home_automation.list_rules": "get_rules|show_rules", "home_automation.create_rule": "add_rule|new_rule",
    "home_automation.delete_rule": "remove_rule|delete_automation",
    "household_chores.list_chores": "get_chores|show_chores", "household_chores.assign": "assign_chore|delegate",
    "library.search": "find_books|search_catalog", "library.reserve": "reserve_book|hold",
    "subscriptions.list": "list_subscriptions|my_plans", "subscriptions.cancel": "unsubscribe|cancel_plan",
}
FIELD_SYN = {
    "account_id": "acct_id|bank_account_id", "action": "command|thao_tac", "address": "addr|dia_chi|street_address",
    "alarm_id": "wake_alarm_id|bao_thuc_id", "alert_id": "warning_id|canh_bao_id", "amount": "so_tien|money_amount|sum",
    "appliance_id": "machine_id|thiet_bi_id", "article_id": "story_id|bai_bao_id", "assignee": "assigned_to|nguoi_lam",
    "attendees": "participants|guests|nguoi_tham_du", "beneficiary_id": "payee_id|recipient_id", "bike_id": "bicycle_id|xe_dap_id",
    "bill_id": "invoice_id|hoa_don_id", "body": "message_body|text", "book_id": "sach_id|publication_id",
    "booking_id": "reservation_id|dat_cho_id", "brightness": "brightness_level|do_sang", "cc": "copy_to|cc_list",
    "channel": "tv_channel|kenh|channel_name", "chat_id": "conversation_id|thread_id", "chore_id": "housework_id|viec_nha_id",
    "color": "colour|mau|light_color", "contact_id": "person_id|lien_he_id", "content": "text|noi_dung|message",
    "course_id": "class_id|khoa_hoc_id", "cycle": "cycle_type|chu_trinh", "date": "day|ngay|on_date",
    "date_from": "start_date|tu_ngay|from_day", "date_to": "end_date|den_ngay|to_day", "days": "frequency|schedule_days",
    "delay_minutes": "delay_mins|postpone_minutes", "destination": "dest|diem_den|arrival", "device": "output_device|thiet_bi",
    "device_id": "thiet_bi_id|gadget_id|dev_id", "dropoff": "drop_location|diem_tra|drop_off_point", "due": "due_date|deadline|han",
    "email_id": "mail_id|thu_id", "enabled": "active|is_on", "end": "end_time|ends_at|ket_thuc",
    "episode_id": "ep_id|tap_id", "event_id": "appointment_id|su_kien_id", "flight_id": "chuyen_bay_id|flight_ref_id",
    "floor": "level|tang|storey", "from_account_id": "source_account_id|tu_tai_khoan_id",
    "from_currency": "source_currency|base_currency", "from_unit": "source_unit|unit_from", "label": "tag|nhan|caption",
    "limit": "max_results|top_n", "location": "place|dia_diem|where", "lock_id": "door_id|khoa_id",
    "locked": "is_locked|lock_state", "meeting_id": "session_id|cuoc_hop_id", "merchant_id": "shop_id|store_id|cua_hang_id",
    "minutes": "duration_minutes|so_phut|mins", "missed_only": "only_missed|missed", "mode": "setting|che_do|operating_mode",
    "name": "display_name|ten", "near": "nearby|gan|around", "note": "memo|ghi_chu|remark", "note_id": "memo_id|ghi_chu_id",
    "number": "ticket_number|ticket_no", "order_id": "purchase_id|don_hang_id", "origin": "from_place|start_point|diem_di",
    "passengers": "pax|so_khach|travellers", "payment": "payment_method|pay_with", "period": "time_range|ky",
    "phone": "phone_no|so_dien_thoai|mobile", "phone_number": "tel_number|so_dt|msisdn", "pickup": "pickup_point|diem_don",
    "place_id": "location_id|dia_diem_id|poi_id", "playlist_id": "list_id|mix_id", "power": "power_state|on_off|nguon",
    "priority": "importance|uu_tien|urgency", "program": "programme|chuong_trinh", "program_id": "broadcast_id|chuong_trinh_id",
    "province": "region|tinh_thanh", "quantity": "qty|so_luong", "query": "keyword|search_text|tu_khoa",
    "reminder_id": "nhac_viec_id|notice_id", "repeat": "recurrence|lap_lai|repeat_mode", "restaurant_id": "eatery_id|quan_an_id|venue_id",
    "reward_id": "gift_id|perk_id", "ride_id": "journey_id|chuyen_xe_id", "room": "room_name|phong|area",
    "rooms": "areas|room_list", "rule_id": "automation_id|quy_tac_id", "scene_id": "preset_id|routine_id",
    "schedule_id": "regimen_id|lich_thuoc_id", "schedule_time": "pickup_time|scheduled_at|hen_gio",
    "seat_class": "cabin_class|hang_ghe|fare_class", "show_id": "series_id|podcast_id", "since": "from_time|after",
    "source": "pay_from|nguon_tien|funding_source", "speed": "fan_speed|toc_do", "start": "start_time|begins_at|bat_dau",
    "station": "radio_station|station_name", "status": "state|trang_thai", "steps": "step_count|so_buoc",
    "subject": "email_subject|tieu_de_thu", "subscription_id": "membership_id|goi_id", "task_id": "todo_id|viec_id",
    "temperature": "temp|nhiet_do|target_temp", "term_months": "duration_months|ky_han_thang", "threshold_kwh": "limit_kwh|nguong_kwh",
    "time": "when|thoi_gian|at_time", "times": "dose_times|time_list", "title": "heading|tieu_de",
    "to": "recipients|nguoi_nhan|send_to", "to_currency": "target_currency|quote_currency", "to_unit": "target_unit|unit_to",
    "topic": "category|chu_de|theme", "track_id": "song_id|bai_hat_id", "tracking_id": "tracking_number|ma_van_don|parcel_id",
    "type": "kind|loai", "unread_only": "only_unread|chua_doc", "valid_until": "expires|expiry", "value": "val|gia_tri|input_value",
    "vehicle": "vehicle_type|loai_xe|ride_type", "volume": "volume_level|am_luong|loudness",
}
TOKEN = re.compile(r"(?<![\w.])([a-z][a-z0-9_]*)(?:\.([a-z][a-z0-9_]*))?(?!\w)")


def opts(s: str) -> list[str]:
    return s.split("|")


# ------------------------------------------------------------------------------------------------------------ the map
class RenameMap:
    """app / tool (full name) / field maps for one copy. Empty map = identity."""

    def __init__(self, app=None, tool=None, field=None):
        self.app, self.tool, self.field = dict(app or {}), dict(tool or {}), dict(field or {})

    def inverse(self) -> "RenameMap":
        return RenameMap({v: k for k, v in self.app.items()}, {v: k for k, v in self.tool.items()},
                         {v: k for k, v in self.field.items()})

    def to_json(self) -> dict:
        return {"app": self.app, "tool": self.tool, "field": self.field}

    # -- text: one identifier pass; dotted names first (full tool, else app prefix), then bare field names
    def text(self, s: str) -> str:
        def sub(m):
            a, b = m.group(1), m.group(2)
            if b is not None:
                full = f"{a}.{b}"
                if full in self.tool:
                    return self.tool[full]
                return f"{self.app[a]}.{b}" if a in self.app else full
            return self.field.get(a, a)
        return TOKEN.sub(sub, s)

    def tool_name(self, n: str) -> str:
        if n in self.tool:
            return self.tool[n]
        a, _, b = n.partition(".")
        return f"{self.app[a]}.{b}" if a in self.app and b else n

    # -- values
    def keys(self, o, key=None):
        """Result objects: rename keys at every depth; strings under message/error get the text pass."""
        if isinstance(o, dict):
            return {self.field.get(k, k): self.keys(v, k) for k, v in o.items()}
        if isinstance(o, list):
            return [self.keys(v, key) for v in o]
        if isinstance(o, str) and key in TEXT_VALUE_KEYS:
            return self.text(o)
        return o

    def call(self, name: str, args: dict) -> tuple[str, dict]:
        """Call arguments: only top-level keys (nested array-item keys are not shown by the compact schema)."""
        if name == CALC:
            return name, args
        args = {self.field.get(k, k): v for k, v in (args or {}).items()}
        if name == ASK:
            if isinstance(args.get("tool_name"), str):
                args["tool_name"] = self.tool_name(args["tool_name"])
            if isinstance(args.get("params"), list):
                args["params"] = [self.field.get(p, p) if isinstance(p, str) else p for p in args["params"]]
        return self.tool_name(name), args

    def spec(self, spec: dict) -> dict:
        out = copy.deepcopy(spec)
        out["name"] = self.tool_name(spec["name"])
        if "app" in out:
            out["app"] = self.app.get(out["app"], out["app"])
        if "description" in out:
            out["description"] = self.text(out["description"])
        params = out.get("parameters") or {}
        props = params.get("properties")
        if props:
            new = {}
            for p, s in props.items():
                s = dict(s)
                if "description" in s:
                    s["description"] = self.text(s["description"])
                if "x-ref" in s:  # "<app>.<collection>": app prefix only (collection = result list key, unchanged)
                    a, _, c = s["x-ref"].partition(".")
                    s["x-ref"] = f"{self.app.get(a, a)}.{c}" if c else s["x-ref"]
                new[self.field.get(p, p)] = s
            params["properties"] = new
        if "required" in params:
            params["required"] = [self.field.get(p, p) for p in params["required"]]
        if "x-one-of" in params:
            params["x-one-of"] = [[self.field.get(p, p) for p in g] for g in params["x-one-of"]]
        return out

    def sample(self, s: dict) -> dict:
        s = copy.deepcopy(s)
        ctx = s["context"]
        ctx["installed_apps"] = {self.app.get(a, a): {**v, "tools": [self.tool_name(t) for t in v["tools"]]}
                                 for a, v in ctx["installed_apps"].items()}
        for turn in s.get("history", []):
            if turn.get("tool_calls"):
                turn["tool_calls"] = [dict(c, **dict(zip(("name", "arguments"), self.call(c["name"], c.get("arguments", {})))))
                                      for c in turn["tool_calls"]]
            if turn.get("tool_results"):
                turn["tool_results"] = self.keys(turn["tool_results"])
        lab = s["label"]
        lab["tool_calls"] = [dict(c, **dict(zip(("name", "arguments"), self.call(c["name"], c.get("arguments", {})))))
                             for c in lab["tool_calls"]]
        lab["tool_results"] = self.keys(lab["tool_results"])
        return s


# ---------------------------------------------------------------------------------------------------- catalogue facts
class Catalogue:
    def __init__(self):
        self.specs = data.tool_specs(SPLIT)
        self.apps = sorted({n.split(".", 1)[0] for n in self.specs})
        self.fields = {p for t in self.specs.values() for p in (t.get("parameters") or {}).get("properties", {})}
        nested = set()
        for t in self.specs.values():
            for s in (t.get("parameters") or {}).get("properties", {}).values():
                it = s.get("items")
                if isinstance(it, dict):
                    nested |= set(it.get("properties", {}))
        self.nested = nested  # item_id, quantity, note: shown nowhere in compact schema
        self.renamable = {f for f in self.fields if f in FIELD_SYN and f not in NEVER_FIELD and f not in nested}
        self.originals = set(self.apps) | self.fields | {n.split(".", 1)[1] for n in self.specs}
        # x-ref collections per app ("food_delivery.restaurants"): a new tool short name must not equal one
        self.collections: dict[str, set[str]] = {}
        texts = []
        for t in self.specs.values():
            texts.append(t.get("description", ""))
            for s in (t.get("parameters") or {}).get("properties", {}).values():
                texts.append(s.get("description", ""))
                a, _, c = s.get("x-ref", "").partition(".")
                if c:
                    self.collections.setdefault(a, set()).add(c)
        # every ASCII identifier token of catalogue text, rules and prompt: a new field name must not be an ordinary
        # word there ("ngay" in "dừng ngay khi"), or the field and the word become indistinguishable
        agent_text = [(PRO / "prompts/rules/verified.md").read_text(encoding="utf-8"),
                      (PRO / "prompts/react_json.md").read_text(encoding="utf-8")]
        self.text_tokens = {m.group(1) for x in texts + agent_text for m in TOKEN.finditer(x)}
        missing = [n for n in self.specs if n.split(".")[0] not in KEEP_APPS and n not in TOOL_SYN]
        assert not missing, missing
        assert not [a for a in self.apps if a not in KEEP_APPS and a not in APP_SYN]


def sample_keys(s: dict) -> set[str]:
    """Every dict key in results and call arguments of the sample: a new field name must not collide with one."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            out.update(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for t in s.get("history", []):
        walk(t.get("tool_results") or [])
        walk([c.get("arguments") for c in t.get("tool_calls") or []])
    walk(s["label"]["tool_results"])
    walk([c.get("arguments") for c in s["label"]["tool_calls"]])
    return out


def draw_map(cat: Catalogue, s: dict, rng: random.Random, p_app: float, p_tool: float, p_field: float) -> RenameMap:
    taken_app = set(cat.apps)
    app = {}
    for a in cat.apps:
        if a in KEEP_APPS or rng.random() >= p_app:
            continue
        c = [x for x in opts(APP_SYN[a]) if x not in taken_app]
        if c:
            app[a] = rng.choice(c)
            taken_app.add(app[a])
    tool = {}
    by_app: dict[str, list[str]] = {}
    for n in cat.specs:
        by_app.setdefault(n.split(".", 1)[0], []).append(n)
    for a, names in by_app.items():
        if a in KEEP_APPS:
            continue
        shorts = {n.split(".", 1)[1] for n in names}
        used = set(shorts) | cat.collections.get(a, set())
        for n in names:
            short = n.split(".", 1)[1]
            new_short = short
            if rng.random() < p_tool:
                c = [x for x in opts(TOOL_SYN[n]) if x not in used]
                if c:
                    new_short = rng.choice(c)
                    used.add(new_short)
            new = f"{app.get(a, a)}.{new_short}"
            if new != n:
                tool[n] = new
    keys = sample_keys(s)
    taken_f = set(cat.fields) | keys | cat.text_tokens
    field = {}
    for f in sorted(cat.renamable):
        if rng.random() >= p_field:
            continue
        c = [x for x in opts(FIELD_SYN[f]) if x not in taken_f]
        if c:
            field[f] = rng.choice(c)
            taken_f.add(field[f])
    return RenameMap(app, tool, field)


def relevant(m: RenameMap, s: dict) -> int:
    """How many renamed names the copy actually exercises: gold tools/apps and gold argument fields."""
    k = 0
    for c in s["label"]["tool_calls"]:
        n = c["name"]
        if n == ASK:
            n = (c.get("arguments") or {}).get("tool_name", n)
            k += sum(p in m.field for p in (c.get("arguments") or {}).get("params", []) if isinstance(p, str))
        k += m.tool_name(n) != n
        k += sum(a in m.field for a in (c.get("arguments") or {}))
    if not s["label"]["tool_calls"]:
        k += sum(a in m.app for a in s["context"]["installed_apps"])
    return k


# -------------------------------------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self):
        cfg = config.load(PRO / "configs/base.yaml")
        self.agent = build("agent", cfg["agent"])
        o = self.agent.opts
        assert o["schema"] == "compact" and o["history"] == "full" and o["ledger"] and not self.agent.calc

    def system(self, orig: dict, m: RenameMap, cache: dict | None = None) -> str:
        a = self.agent
        specs = data.tool_specs(SPLIT)
        rs = m.sample(orig)
        rspecs = {m.tool_name(n): m.spec(t) for n, t in specs.items()}
        return _render(a.template, {
            "rules": m.text(a.rules),
            "timetable": context.timetable(orig["context"]["current_time"], a.opts["timetable"]),
            "profile": context.profile(orig),
            "schema": context.schema(SPLIT, rs, a.opts["schema"], rspecs),
            "ledger": context.ledger(orig),  # IDs and values only; walking renamed keys would lose labels
        })

    def history_texts(self, orig: dict, user: str):
        """Recover the history user texts of the row by rendering the history with placeholders."""
        nu = sum(t.get("role") == "user" for t in orig.get("history", []))
        ph = {f"u{k}": f"\x00U{k}\x00" for k in range(1, nu + 1)}
        tmpl = context.history(orig, ph, self.agent.opts["history"])
        pat = re.escape(tmpl)
        for k in range(1, nu + 1):
            pat = pat.replace(re.escape(ph[f"u{k}"]), f"(?P<u{k}>.*?)")
        head = "LỊCH SỬ HỘI THOẠI:\n"
        if not user.startswith(head):
            return None, None
        mm = re.match(re.escape(head) + pat + r"\n\n(?P<rest>YÊU CẦU CUỐI CỦA NGƯỜI DÙNG.*)\Z", user, re.S)
        if not mm:
            return None, None
        return {f"u{k}": mm.group(f"u{k}") for k in range(1, nu + 1)}, mm.group("rest")

    def user(self, orig: dict, m: RenameMap, user: str) -> str | None:
        if not orig.get("history"):
            return user if not user.startswith("LỊCH SỬ") else None
        texts, rest = self.history_texts(orig, user)
        if texts is None:
            return None
        rs = m.sample(orig)
        for t0, t1 in zip(orig["history"], rs["history"]):  # context.history cuts each result at 1200 chars
            for r0, r1 in zip(t0.get("tool_results") or [], t1.get("tool_results") or []):
                if len(json.dumps(r0, ensure_ascii=False)) <= 1200 < len(json.dumps(r1, ensure_ascii=False)):
                    return None
        hist = context.history(rs, texts, self.agent.opts["history"])
        return "LỊCH SỬ HỘI THOẠI:\n" + hist + "\n\n" + rest

    def steps(self, msgs: list[dict], m: RenameMap) -> list[dict] | None:
        out = []
        for msg in msgs:
            c = msg["content"]
            if msg["role"] == "assistant":
                act = json.loads(c)
                if act.get("action") == "call":
                    name, args = m.call(act["name"], act.get("arguments", {}))
                    act = {**act, "name": name, "arguments": args}
                out.append({"role": "assistant", "content": json.dumps(act, ensure_ascii=False)})
            else:
                mt = re.match(r"KẾT QUẢ (\S+): (.*)\Z", c, re.S)
                if not mt:
                    return None
                obs = json.loads(mt.group(2))
                name = mt.group(1)
                new = m.keys(obs)
                if len(json.dumps(new, ensure_ascii=False)) > self.agent.obs_chars:  # longer keys would cut the JSON
                    return None
                out.append({"role": "user", "content": f"KẾT QUẢ {m.call(name, {})[0]}: {self.agent._obs(new)}"})
        return out

    def row(self, orig: dict, m: RenameMap, row: dict) -> list[dict] | None:
        msgs = row["messages"]
        u = self.user(orig, m, msgs[1]["content"])
        st = self.steps(msgs[2:], m)
        if u is None or st is None:
            return None
        return [{"role": "system", "content": self.system(orig, m)}, {"role": "user", "content": u}, *st]


# --------------------------------------------------------------------------------------------------------------- main
def source_id(row_id: str) -> str | None:
    if row_id.startswith("synth"):
        return None
    mm = re.match(r"(?:teach_p_)?(train_\d+)", row_id)
    return mm.group(1) if mm else None


def kind_of(row: dict) -> str:
    i = row["id"]
    return "d1" if i.startswith("teach_p_") else "valasr" if i.endswith("_valasr") else "asr" if row.get("asr") else "clean"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=0.25, help="number of renamed copies = ratio x rows of --inp")
    ap.add_argument("--ratio-of", choices=["input", "output"], default="input",
                    help="output: copies = ratio x rows of the OUTPUT file instead")
    ap.add_argument("--p-app", type=float, default=0.6)
    ap.add_argument("--p-tool", type=float, default=0.5)
    ap.add_argument("--p-field", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2611)
    ap.add_argument("--maps-out", default=None, help="JSONL of {id, map} for every copy (default: <out>.maps.jsonl)")
    a = ap.parse_args(argv)

    inp, out = Path(a.inp).resolve(), Path(a.out).resolve()
    rows = [json.loads(line) for line in inp.open(encoding="utf-8")]
    cat, ren = Catalogue(), Renderer()
    S = data.by_id(SPLIT)
    rng = random.Random(a.seed)
    stats = Counter()
    eligible = []
    for i, r in enumerate(rows):
        sid = source_id(r["id"])
        if sid is None:
            stats["skip_synth"] += 1
            continue
        orig = S[sid]
        if ren.row(orig, RenameMap(), r) != r["messages"]:  # identity round-trip: same code path, empty map
            stats["skip_roundtrip_fail"] += 1
            continue
        eligible.append(i)
    n_copies = round(a.ratio * len(rows) / (1 - a.ratio)) if a.ratio_of == "output" else round(a.ratio * len(rows))
    n_copies = min(n_copies, len(eligible))
    pick = sorted(rng.sample(eligible, n_copies))
    copies, maps = [], []
    for i in pick:
        r = rows[i]
        orig = S[source_id(r["id"])]
        msgs = None
        for _ in range(20):  # redraw until the copy exercises a rename and no observation grows past its cut
            m = draw_map(cat, orig, rng, a.p_app, a.p_tool, a.p_field)
            if not relevant(m, orig):
                stats["redraw_not_relevant"] += 1
                continue
            msgs = ren.row(orig, m, r)
            if msgs is not None:
                break
            stats["redraw_obs_too_long"] += 1
        if msgs is None:
            stats["copy_skipped_no_valid_map"] += 1
            continue
        new = {k: v for k, v in r.items() if k != "messages"}
        # input ids are not unique (clean and ASR copies of train_* share one id): tag the ASR source and keep its line
        tag = "_asr" if kind_of(r) == "asr" else ""
        new.update({"id": f"{r['id']}{tag}_ren2", "src_id": r["id"], "src_line": i, "renamed": True, "ren2": True,
                    "messages": msgs})
        copies.append(new)
        maps.append({"id": new["id"], "map": m.to_json()})
        stats[f"copy_{kind_of(r)}"] += 1
        stats["renamed_apps"] += len(m.app)
        stats["renamed_tools"] += len(m.tool)
        stats["renamed_fields"] += len(m.field)
        stats["relevant_renames"] += relevant(m, orig)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows + copies:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mp = Path(a.maps_out) if a.maps_out else out.with_suffix(".maps.jsonl")
    with mp.open("w", encoding="utf-8") as f:
        for x in maps:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    st = {"input": str(inp), "input_sha256": sha256(inp), "output": str(out), "output_sha256": sha256(out),
          "maps": str(mp), "rows_in": len(rows), "eligible": len(eligible), "copies": len(copies),
          "rows_out": len(rows) + len(copies), "copy_share": round(len(copies) / (len(rows) + len(copies)), 4),
          "args": vars(a), **dict(stats)}
    out.with_suffix(".stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
