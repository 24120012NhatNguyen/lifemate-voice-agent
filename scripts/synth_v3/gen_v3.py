"""Sinh synth v3 (SV3_REPORT.md): mẫu SFT trên app tổng hợp kiểu catalog train, gold chạy thật trong ToolEnv của mẫu chủ,
lời nói kiểu bản chép ASR có khối "Bản nghe khác", render bằng build_sft.conversation + config 10 như dòng production.

    cd fable-eval/lm51-asr && LIFEMATE_DATA=... .venv/bin/python ../tools-pro/synth_v3/gen_v3.py --n 1000

Chỉ đọc split train (train_rest làm mẫu chủ, cache ASR theo fold của config 181 cho bảng nhầm lẫn). Không đọc public_test.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LM51 = HERE.parents[1]
sys.path[:0] = [str(HERE), str(LM51 / "src"), str(LM51 / "scripts")]

from lifemate import config, data, splits  # noqa: E402
from lifemate.config import REPO  # noqa: E402
from lifemate.guards import GuardedEnv  # noqa: E402
from lifemate.registry import build  # noqa: E402
from lifemate.synth import vi  # noqa: E402
import build_sft  # noqa: E402
import apps_v3 as A  # noqa: E402
import speech as S  # noqa: E402

SPLIT = "train"
CFG = "configs/experiments/10_qwen3-4b_gold_react.yaml"
ASR181 = LM51 / "cache/asr/train/hf_whisper_97e4c4320d.jsonl"  # bản chép theo fold của config 181 (train_rest)
GOLD = LM51 / "cache/asr/train/gold_2c6032ed1a.jsonl"
SLOT = re.compile(r"\{(\w+)\}")

# phân bố mục tiêu ~ gold train_rest (measure_real.py): làm / hỏi / dừng
KINDS = {
    "rw_ref": 13,
    "rw_name": 4,
    "rw_contact": 6,
    "rw_resched": 6,
    "two_source": 4,
    "cond": 7,
    "pick_detail": 3,
    "correction": 3,
    "create": 7,
    "hist_ref": 5,
    "multi": 11,
    "read_only": 4,
    "read_ask": 4,
    "act_ask": 9,
    "ask_only": 3,
    "stop": 8,
}
TEXT = {
    ("pet", "note"): ["bé hơi biếng ăn", "bé bị ngứa tai", "nhớ cắt móng luôn", "bé sợ người lạ"],
    ("car", "note"): ["kiểm tra phanh luôn", "xem giúp tiếng kêu ở bánh trước", "thay luôn lọc gió"],
    ("school", "reason"): ["bé bị sốt", "gia đình về quê", "bé đi khám răng", "bé bị ho"],
    ("gift", "message"): ["chúc mừng sinh nhật", "chúc mẹ luôn khoẻ", "mừng tân gia", "chúc mừng thăng chức"],
    ("split", "note"): ["ăn lẩu tối qua", "vé xem phim", "tiền taxi", "quà sinh nhật Lan", "tiền điện tháng trước"],
    ("clinic", "symptom"): ["đau họng", "nổi mẩn ngứa", "đau răng", "ho lâu ngày", "mỏi mắt"],
    ("building", "description"): [
        "bóng đèn hành lang tầng sáu cháy",
        "vòi nước nhà tắm bị rỉ",
        "thang máy kêu to",
        "ổ cắm bếp bị chập",
    ],
}
ASK = {  # (domain, tool): [(tham số còn thiếu, mẫu câu không nói tham số đó)]
    ("pet", "book"): [("start", "đặt lịch {service} cho bé {obj} giúp {p}")],
    ("car", "book"): [("time", "đặt lịch {service} cho {obj}")],
    ("clean", "book"): [("start", "đặt {obj} qua dọn nhà[ {hours}] giúp {p}")],
    ("school", "leave"): [
        ("date", "xin cho {obj} nghỉ học, lý do {reason}"),
        ("reason", "xin cho {obj} nghỉ học {date}"),
    ],
    ("school", "pay"): [("source", "đóng {obj} cho trường giúp {p}")],
    ("salon", "book"): [("start", "đặt lịch {obj} ở salon[ với {stylist_id}]")],
    ("split", "request"): [("amount", "đòi tiền {contact_id} giúp {p}[, tiền {note}]")],
    ("parking", "topup"): [("amount", "nạp tiền vào tài khoản gửi xe {source}")],
    ("clinic", "book"): [("start", "đặt lịch khám {obj}[, triệu chứng {symptom}]")],
    ("heater", "temp"): [("temperature", "chỉnh nhiệt độ bình nóng lạnh giúp {p}")],
    ("building", "report"): [("description", "báo ban quản lý bên {p} bị {category}")],
    ("building", "amenity"): [("start", "đặt {amenity} giúp {p}")],
}
BUILDING_DESC = {
    "electric": ["bóng đèn hành lang tầng sáu cháy", "ổ cắm bếp bị chập"],
    "water": ["vòi nước nhà tắm bị rỉ", "nước yếu từ sáng"],
    "elevator": ["thang máy kêu to", "thang máy số hai đứng giữa tầng"],
    "cleaning": ["rác hành lang chưa ai dọn"],
    "noise": ["nhà tầng trên khoan tường cả tối"],
}
RESCHED_WHAT = {"pet": "lịch khám của bé {obj}", "clean": "lịch dọn nhà của {obj}", "clinic": "lịch khám {obj}"}
UNSUPPORTED_DOMS = ["cinema", "salon", "clinic", "pet", "car", "garden", "heater", "parking", "school", "building"]
REPLIES = ["Dạ, em đã làm xong rồi ạ.", "Dạ xong rồi ạ.", "Đã xong ạ.", "Dạ, em đặt xong rồi ạ."]


class Skip(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
class Utt:
    """Câu nói ghép từ các mảnh; mảnh mang giá trị người dùng nói ra được đánh dấu (entity / value) để nhiễu ASR biết chỗ."""

    def __init__(self):
        self.parts: list[tuple[str, str | None, S.Say | None]] = []

    def add(self, text: str, kind: str | None = None, say: S.Say | None = None):
        self.parts.append((text, kind, say))
        return self

    def extend(self, other: "Utt"):
        self.parts += other.parts
        return self

    @property
    def text(self) -> str:
        return "".join(p[0] for p in self.parts)

    def spans(self):
        out, pos = [], 0
        for t, k, s in self.parts:
            if k:
                out.append((pos, pos + len(t), k, s, t))
            pos += len(t)
        return out


def render(tpl: str, vals: dict, p: str) -> Utt:
    """Mẫu câu → Utt. vals: slot → (Say, kind). Đoạn [...] chỉ giữ khi mọi slot trong đó có giá trị."""

    def keep(m):
        return m.group(1) if all(s == "p" or s in vals for s in SLOT.findall(m.group(1))) else ""

    tpl = re.sub(r"\[([^\]]*)\]", keep, tpl).replace("{p}", p)
    u, pos = Utt(), 0
    for m in SLOT.finditer(tpl):
        if m.start() > pos:
            u.add(tpl[pos : m.start()])
        say, kind = vals[m.group(1)]
        u.add(say.text, kind, say)
        pos = m.end()
    if pos < len(tpl):
        u.add(tpl[pos:])
    return u


def finish(u: Utt, rng: random.Random, p: str, suffix=True) -> Utt:
    """Tiền tố, tiểu từ cuối, viết hoa đầu câu, chấm câu (như vi.utter nhưng giữ mảnh)."""
    core = u.text
    pre = rng.choice(vi.PREFIX)
    tail = ""
    if suffix and not re.search(
        r"\b(nha|nhé|nghen|nhen|đi|với|luôn|thôi|không|chưa|bao nhiêu|gì|nào|vậy|nhỉ)$", core.strip()
    ):
        opts = [
            x
            for x in vi.SUFFIX
            if not (re.search(r"giúp|giùm|dùm|\{p\}|" + re.escape(p), core) and re.search(r"giúp|giùm|dùm|\{p\}", x))
        ]
        tail = rng.choice(opts).replace("{p}", p)
    out = Utt()
    if pre:
        out.add(pre)
    out.extend(u)
    if tail:
        out.add(tail)
    t0, k0, s0 = out.parts[0]
    out.parts[0] = (t0[:1].upper() + t0[1:], k0, s0)
    end = out.text.rstrip()
    if end[-1:] not in ".?!":
        out.add("?" if re.search(r"(không|chưa|bao nhiêu|mấy|gì|nào|sao|vậy|nhỉ|hả)$", end) else ".")
    return out


# ---------------------------------------------------------------------------------------------------------------------
class Gen:
    def __init__(self, ctx, host: dict, rng: random.Random, kind: str):
        self.ctx, self.host, self.rng, self.kind = ctx, host, rng, kind
        self.now = datetime.fromisoformat(host["context"]["current_time"])
        self.p = vi.pronoun(host["context"]["user_profile"].get("name"), rng)
        self.metas: list[dict] = []
        self.doms: list[A.Domain] = []
        self.turn: list = []
        self.cache: dict = {}
        self.known: dict = {}  # id → item đã thấy trong lịch sử tổng hợp
        self.synth_hist: list[dict] = []
        self._cat = None
        self.fuzzy_ents: set[str] = set()  # thực thể tìm qua tool list/status: được phép nghe sai ở mọi phương án
        self.ask_in_alt = True
        host_env = data.tool_env(SPLIT, host["id"])
        self.contacts = (
            (host_env.states.get("contacts") or {}).get("contacts") or []
            if "contacts" in host["context"]["installed_apps"]
            else []
        )

    # -- dựng app, môi trường ---------------------------------------------------------------------------------------
    def setup(self, keys: list[str]):
        taken = set(self.ctx.real_apps) | A.HELDOUT | set(self.host["context"]["installed_apps"])
        for k in keys:
            dom = A.BY_KEY[k]
            names = A.choose_names(dom, self.rng, taken)
            taken.add(names["app"])
            state = A.initial_state(dom, names, self.now, self.rng, self.contacts)
            self.metas.append({"domain": k, "names": names, "state": state})
            self.doms.append(dom)
        hist_apps = {c["name"].split(".")[0] for t in self.host.get("history", []) for c in t.get("tool_calls") or []}
        items = list(self.host["context"]["installed_apps"].items())
        # thay bớt app thật không được gọi (ưu tiên app của val_app) bằng app tổng hợp: số app cài giữ như phân bố thật
        # app của val_app: gỡ hết (như build_sft.without_heldout) để synth không thêm lần lộ schema app giữ lại
        held = {a for a, _ in items if a in A.HELDOUT and a not in hist_apps}
        removable = [a for a, _ in items if a not in hist_apps and a not in ("contacts", "reminders") and a not in held]
        self.rng.shuffle(removable)
        drop = held | set(removable[: max(0, len(keys) - len(held))])
        items = [(a, v) for a, v in items if a not in drop]
        for m in self.metas:
            cat = A.catalog(A.BY_KEY[m["domain"]], m["names"])
            items.insert(self.rng.randint(0, len(items)), (cat["id"], {"tools": [t["name"] for t in cat["tools"]]}))
        self.installed = dict(items)
        self.specs = {t["name"]: t for m in self.metas for t in A.catalog(A.BY_KEY[m["domain"]], m["names"])["tools"]}
        self.env = self.new_env()

    def tools_installed(self):
        return [t for a in self.installed.values() for t in a["tools"]]

    def new_env(self):
        return A.make_env(
            lambda: data.zaic_tools(SPLIT), data.data_root(), self.host["id"], self.metas, self.tools_installed()
        )

    def tname(self, i: int, key: str) -> str:
        return f"{self.metas[i]['names']['app']}.{self.metas[i]['names']['tools'][key]}"

    def tool(self, i: int, key: str) -> A.Tool:
        return next(t for t in self.doms[i].tools if t.key == key)

    def call(self, name: str, args: dict):
        sig = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if sig in self.cache:
            return self.cache[sig]
        r = self.env.call(name, args)
        if r.get("ok") is False or r.get("done") is False:
            raise Skip(f"{name} {args}: {r}")
        self.turn.append(({"name": name, "arguments": args}, r))
        self.cache[sig] = r
        return r

    def ask(self, tool_name: str, params: list[str]):
        args = {"tool_name": tool_name, "params": params}
        r = self.env.call("sys.ask_missing_info", args)
        if r.get("ok") is False:
            raise Skip(f"ask: {r}")
        self.turn.append(({"name": "sys.ask_missing_info", "arguments": args}, r))

    def items(self, i: int, ck: str) -> list[dict]:
        return self.env.states[self.metas[i]["names"]["app"]][self.metas[i]["names"]["colls"][ck]["name"]]

    def status(self, i: int) -> dict:
        return self.env.states[self.metas[i]["names"]["app"]][self.doms[i].status_key]

    # -- giá trị nói ra --------------------------------------------------------------------------------------------
    def value(self, i: int, t: A.Tool, p: A.P):
        rng, dom = self.rng, self.doms[i]
        if p.kind == "datetime":
            d = A.future_slot(self.now, rng, 0, 9)
            return A._iso(d), S.when(self.now, d, rng)
        if p.kind == "date":
            d = self.now.date() + timedelta(days=rng.randint(1, 9))
            return d.isoformat(), S.date_say(self.now, d, rng)
        if p.kind == "int":
            n = rng.randint(p.lo, p.hi)
            if p.canon == "temperature":
                n = rng.choice([40, 45, 50, 55, 60, 65])
            return n, S.integer(n, p.unit, rng)
        if p.kind == "money":
            v = rng.choice(
                [
                    50000,
                    80000,
                    120000,
                    150000,
                    200000,
                    250000,
                    300000,
                    450000,
                    500000,
                    1000000,
                    1200000,
                    1500000,
                    2000000,
                ]
            )
            return v, S.money(v, rng)
        if p.kind == "enum":
            v = rng.choice(p.values)
            return v, S.Say(rng.choice(p.say[v]) if p.say else v)
        if p.kind == "bool":
            v = rng.random() < 0.5
            return v, S.Say("bật" if v else "tắt")
        if p.kind == "text":
            if (dom.key, p.canon) == ("building", "description") and self._cat:
                s = rng.choice(BUILDING_DESC[self._cat])
                return s, S.Say(s)
            s = rng.choice(TEXT[(dom.key, p.canon)])
            return s, S.Say(s)
        raise ValueError(p.kind)

    def key_of(self, i: int, ck: str, item: dict) -> str:
        return item[self.doms[i].colls[ck].key]

    def pick_item(self, i: int, ck: str, ok=lambda x: True) -> dict:
        xs = [x for x in self.items(i, ck) if ok(x)]
        if not xs:
            raise Skip(f"không có mục {ck}")
        x = self.rng.choice(xs)
        same = [y for y in self.items(i, ck) if self.key_of(i, ck, y) == self.key_of(i, ck, x)]
        if len(same) > 1:
            raise Skip("tên trùng")
        return x

    def read_ref(self, i: int, ck: str, spoken: str) -> list[dict]:
        """Đọc để lấy ID: tool list gọi {}, tool search gọi query = đúng tên người dùng nói (luật 5, 8)."""
        f = A.finder_of(self.doms[i], ck)
        name = self.tname(i, f.key)
        if f.op == "list":
            self.fuzzy_ents.add(spoken)
            r = self.call(name, {})
        else:
            r = self.call(name, {self.metas[i]["names"]["params"][f"{f.key}.$q"]: spoken})
        return r[self.metas[i]["names"]["colls"][ck]["name"]]

    def read_status(self, i: int) -> dict:
        t = next(t for t in self.doms[i].tools if t.op == "status")
        return self.call(self.tname(i, t.key), {})

    def contact(self, ok=lambda c: True) -> dict:
        cands = []
        for c in self.contacts:
            hits = [
                x
                for x in self.contacts
                if S.fold(c["name"]) in S.fold(x["name"]) or S.fold(c["name"]) in S.fold(x.get("relation", ""))
            ]
            if len(hits) == 1 and ok(c):
                cands.append(c)
        if not cands:
            raise Skip("không có liên hệ dùng được")
        return self.rng.choice(cands)

    # -- một lệnh ghi đầy đủ ------------------------------------------------------------------------------------------
    def write(
        self,
        i: int,
        key: str,
        *,
        target: dict | None = None,
        tpl: str | None = None,
        drop: str | None = None,
        from_hist: bool = False,
        fix: bool = False,
        anaphor: str | None = None,
        extra: dict | None = None,
    ) -> Utt:
        """Đọc những gì cần (theo thứ tự xuất hiện trong câu) rồi gọi tool ghi. drop: tham số bắt buộc người dùng không nói
        → đọc phần đủ thông tin rồi sys.ask_missing_info. from_hist: ID mục đích đã có trong lịch sử, không đọc lại."""
        rng, t = self.rng, self.tool(i, key)
        names = self.metas[i]["names"]
        tpl = tpl or rng.choice(t.say)
        slots = SLOT.findall(re.sub(r"\[[^\]]*\]", "", tpl))
        opt_slots = set(SLOT.findall(tpl)) - set(slots)
        main = next((p for p in t.params if p.kind == "ref" and p.ref == t.coll), None) or next(
            (p for p in t.params if p.kind in ("ref", "nameref")), None
        )
        vals, gold, reads = {}, {}, []
        for p in t.params:
            slot = "obj" if p is main else p.canon
            if p.canon == drop:
                continue
            said = slot in slots or (slot in opt_slots and (p.req or rng.random() < 0.5))
            if not said and not p.req:
                continue
            if not said and p.req and p.kind not in ("ref", "contact", "nameref"):
                raise Skip(f"mẫu câu thiếu {slot}")
            if p.kind == "ref":
                item = target if (p is main and target is not None) else self.pick_item(i, p.ref, self.ok_for(t, p))
                spoken = self.key_of(i, p.ref, item)
                vals[slot] = (
                    S.Say(anaphor if (anaphor and p is main) else spoken),
                    None if anaphor and p is main else "entity",
                )
                gold[p.canon] = item["id"]
                if not (from_hist and p is main):
                    reads.append((tpl.find("{" + slot + "}"), ("ref", p.ref, spoken)))
                if p.ref == "showtimes":
                    when = S.when(self.now, datetime.fromisoformat(item["start"]), rng)
                    vals["start"] = (when, "value")
            elif p.kind == "contact":
                c = self.contact()
                vals[slot] = (S.Say(c["name"]), "entity")
                gold[p.canon] = c["id"]
                reads.append((tpl.find("{" + slot + "}"), ("contact", c["name"])))
            elif p.kind == "nameref":
                st = self.status(i)
                z = rng.choice(st[p.ref])
                vals[slot] = (S.Say(z), "entity")
                gold[p.canon] = z
                reads.append((tpl.find("{" + slot + "}"), ("status",)))
            else:
                v, say = self.value(i, t, p)
                if p.canon == "category":
                    self._cat = v
                if fix and p.kind in ("datetime", "int", "money") and "fixed" not in vals:
                    v2, say2 = self.value(i, t, p)
                    if v2 == v:
                        raise Skip("sửa trùng giá trị")
                    joint = f"{say.text}{rng.choice(vi.FIX)}{say2.text}"
                    say, v = S.Say(joint), v2
                    vals["fixed"] = True
                vals[slot] = (say, "value")
                gold[p.canon] = v
        if fix and "fixed" not in vals:
            raise Skip("không có gì để sửa")
        vals.pop("fixed", None)
        if extra:
            vals.update(extra)
        for _, r in sorted(reads, key=lambda x: x[0]):
            if r[0] == "ref":
                got = self.read_ref(i, r[1], r[2])
                if not anaphor and sum(self.key_of(i, r[1], x) == r[2] for x in got) != 1:
                    raise Skip("đọc không ra đúng một mục")
            elif r[0] == "contact":
                res = self.call("contacts.search", {"query": r[1]})
                if len(res.get("contacts", [])) != 1:
                    raise Skip("tìm liên hệ không ra đúng một")
            else:
                self.read_status(i)
        u = render(tpl, vals, self.p)
        if drop:
            pn = A.pname(names, t, next(p for p in t.params if p.canon == drop))
            self.ask(self.tname(i, key), [pn])
            return u
        args = {A.pname(names, t, p): gold[p.canon] for p in t.params if p.canon in gold}
        self.call(self.tname(i, key), args)
        return u

    def ok_for(self, t: A.Tool, p: A.P):
        if t.op in ("act", "update") and p.ref == t.coll:

            def ok(x):
                if x.get("status") in ("cancelled", "delivered", "paid") or (
                    t.sets and all(x.get(k) == v for k, v in t.sets.items())
                ):
                    return False
                return t.op != "update" or ("start" in x and "end" in x)

            return ok
        return lambda x: True

    def tools_where(self, i: int, pred) -> list[A.Tool]:
        return [t for t in self.doms[i].tools if pred(t)]

    # -- lịch sử tổng hợp -------------------------------------------------------------------------------------------
    def hist_turn(self, text: str, reply: str, calls_results: list):
        self.synth_hist.append({"role": "user", "audio_text": text})
        self.synth_hist.append(
            {
                "role": "assistant",
                "text": reply,
                "tool_calls": [c for c, _ in calls_results],
                "tool_results": [r for _, r in calls_results],
            }
        )

    def flush_hist(self):
        """Lượt lịch sử vừa chạy xong: state hiện tại thành state ban đầu của lượt cuối; môi trường mới cho lượt cuối."""
        out = list(self.turn)
        for m in self.metas:
            m["state"] = copy.deepcopy(self.env.states[m["names"]["app"]])
        self.turn, self.cache = [], {}
        self.env = self.new_env()
        return out


# ---------------------------------------------------------------------------------------------------------------------
def has(dom: A.Domain, pred) -> bool:
    return any(pred(t) for t in dom.tools)


def direct(t: A.Tool) -> bool:
    """Tool ghi gọi thẳng được, không cần đọc: không có tham số ref/contact/nameref."""
    return (
        t.op in ("create", "set", "topup")
        and t.say
        and not any(p.kind in ("ref", "contact", "nameref") for p in t.params)
        and t.params
    )


def refwrite(t: A.Tool) -> bool:
    return (
        t.op in ("create", "act", "set")
        and t.say
        and any(p.kind == "ref" for p in t.params)
        and not any(p.kind == "contact" for p in t.params)
    )


CAPS = {
    "rw_ref": lambda d: has(d, refwrite),
    "rw_name": lambda d: has(d, lambda t: any(p.kind == "nameref" for p in t.params)),
    "rw_contact": lambda d: has(d, lambda t: any(p.kind == "contact" for p in t.params)),
    "rw_resched": lambda d: d.key in RESCHED_WHAT,
    "two_source": lambda d: d.key in ("salon", "gift"),
    "cond": lambda d: d.key in ("garden", "parking", "heater", "school", "split"),
    "pick_detail": lambda d: d.key in ("pet", "school", "clinic"),
    "correction": lambda d: has(
        d, lambda t: t.say and any(p.kind in ("datetime", "int", "money") and p.req for p in t.params)
    ),
    "create": lambda d: has(d, direct),
    "hist_ref": lambda d: has(d, lambda t: t.op == "act" and t.say),
    "multi": lambda d: has(d, direct) or has(d, refwrite),
    "read_only": lambda d: has(d, lambda t: t.op in ("list", "status") and t.say) or d.key in ("cinema", "clinic"),
    "read_ask": lambda d: (
        d.key in ("pet", "school", "clinic")
        or any(k[0] == d.key and any(p.kind == "ref" for t in d.tools if t.key == k[1] for p in t.params) for k in ASK)
    ),
    "act_ask": lambda d: has(d, direct) or has(d, refwrite),
    "ask_only": lambda d: any(k[0] == d.key and direct(next(t for t in d.tools if t.key == k[1])) for k in ASK),
    "stop": lambda d: True,
}


def scenario(g: Gen, kind: str) -> tuple[Utt, str]:
    """Dựng lượt cuối trên app tổng hợp 0 (app đích). Trả về (câu nói, nhãn con)."""
    rng, dom, p = g.rng, g.doms[0], g.p
    sub = kind
    if kind == "rw_ref":
        t = rng.choice(g.tools_where(0, refwrite))
        return g.write(0, t.key), sub
    if kind == "rw_name":
        t = rng.choice(g.tools_where(0, lambda t: any(x.kind == "nameref" for x in t.params)))
        return g.write(0, t.key), sub
    if kind == "rw_contact":
        if not g.contacts:
            raise Skip("mẫu chủ không có danh bạ")
        t = rng.choice(g.tools_where(0, lambda t: any(x.kind == "contact" for x in t.params)))
        return g.write(0, t.key), sub
    if kind == "two_source":
        if dom.key == "salon":
            t = g.tool(0, "book")
            return g.write(
                0, "book", tpl=next(s for s in t.say if "stylist_id" in s).replace("[", "").replace("]", "")
            ), "salon_service+stylist"
        if not g.contacts:
            raise Skip("không có danh bạ")
        return g.write(0, "send"), "gift+contact"
    if kind == "rw_resched":
        t = next(t for t in dom.tools if t.op == "update")
        ck = t.coll
        item = g.pick_item(0, ck, lambda x: x.get("status") == "booked" and "end" in x)
        spoken = g.key_of(0, ck, item)
        same = [x for x in g.items(0, ck) if g.key_of(0, ck, x) == spoken]
        if len(same) != 1:
            raise Skip("trùng")
        s0, e0 = (
            datetime.fromisoformat(item["start"]).replace(tzinfo=None),
            datetime.fromisoformat(item["end"]).replace(tzinfo=None),
        )
        dur = e0 - s0
        mode = rng.choice(["abs", "abs", "shift", "day"])
        what = RESCHED_WHAT[dom.key]
        if mode == "abs":
            ns = A.future_slot(g.now, rng, 0, 9)
            say = S.when(g.now, ns, rng)
            tpl = rng.choice(
                [f"dời {what} sang {{new}}", f"chuyển {what} qua {{new}} nha", f"đổi {what} thành {{new}}"]
            )
            vals = {"obj": (S.Say(spoken), "entity"), "new": (say, "value")}
        elif mode == "shift":
            m = rng.choice([30, 60, 90, 120]) * rng.choice([1, -1])
            ns = s0 + timedelta(minutes=m)
            say = S.duration(abs(m), rng)
            tpl = (
                (f"lùi {what} {{new}}" if m > 0 else f"đẩy {what} lên sớm {{new}}")
                if rng.random() < 0.6
                else (f"dời {what} muộn hơn {{new}}" if m > 0 else f"dời {what} sớm hơn {{new}}")
            )
            vals = {"obj": (S.Say(spoken), "entity"), "new": (say, "value")}
        else:
            k = rng.choice([1, 2, 3])
            ns = s0 + timedelta(days=k)
            say = S.date_say(g.now, ns.date(), rng)
            tpl = rng.choice([f"chuyển {what} sang {{new}}, giữ giờ cũ", f"dời {what} qua {{new}} cùng giờ"])
            vals = {"obj": (S.Say(spoken), "entity"), "new": (say, "value")}
        if ns <= g.now.replace(tzinfo=None) + timedelta(hours=1) or (ns.date() - g.now.date()).days > 13 or ns == s0:
            raise Skip("giờ mới không hợp lệ")
        g.read_ref(0, ck, spoken)
        names = g.metas[0]["names"]
        args = {A.pname(names, t, x): v for x, v in zip(t.params, [item["id"], A._iso(ns), A._iso(ns + dur)])}
        g.call(g.tname(0, t.key), args)
        return render(tpl, vals, p), f"resched_{mode}"
    if kind == "cond":
        return cond(g)
    if kind == "pick_detail" or (kind == "read_ask" and dom.key in ("pet", "school", "clinic") and rng.random() < 0.6):
        return ambiguous(g, pick=kind == "pick_detail")
    if kind == "correction":
        ts = g.tools_where(
            0,
            lambda t: (
                t.say
                and any(x.kind in ("datetime", "int", "money") and x.req for x in t.params)
                and not any(x.kind == "contact" for x in t.params)
            ),
        )
        t = rng.choice(ts)
        tpl = rng.choice(t.say)
        return g.write(0, t.key, tpl=tpl, fix=True), sub
    if kind in ("create", "multi"):
        pool = g.tools_where(0, direct)
        if kind == "multi" and (not pool or rng.random() < 0.25):
            pool = g.tools_where(0, refwrite)
        t = rng.choice(pool)
        u = g.write(0, t.key)
        if kind == "create":
            return u, f"create_{t.op}"
        # việc thứ hai: nhắc việc trên app thật (nếu cài) hoặc một lệnh tạo trên app tổng hợp thứ hai
        if "reminders.create" in g.tools_installed() and rng.random() < 0.7:
            d = A.future_slot(g.now, rng, 0, 6)
            content = rng.choice(["mua sữa cho con", "gọi cho khách", "đóng tiền nhà", "lấy đồ giặt", "mang hồ sơ"])
            say = S.when(g.now, d, rng)
            u2 = render(
                rng.choice(["nhắc {p} {when} {content}", "{when} nhắc {p} {content} nha"]),
                {"when": (say, "value"), "content": (S.Say(content), "value")},
                p,
            )
            g.call("reminders.create", {"content": content, "time": A._iso(d)})
            sub = "multi_real_reminder"
        elif len(g.doms) > 1 and any(direct(t2) for t2 in g.doms[1].tools):
            t2 = rng.choice(g.tools_where(1, direct))
            u2 = g.write(1, t2.key)
            sub = "multi_two_synth"
        elif [t2 for t2 in g.tools_where(0, direct) if t2.key != t.key]:
            t2 = rng.choice([t2 for t2 in g.tools_where(0, direct) if t2.key != t.key])
            u2 = g.write(0, t2.key)
            sub = "multi_same_app"
        else:
            raise Skip("không có việc thứ hai")
        return u.add(rng.choice(vi.JOIN)).extend(u2), sub
    if kind == "hist_ref":
        return hist_ref(g)
    if kind == "read_only":
        return read_only(g)
    if kind in ("read_ask", "ask_only"):
        keys = [k for k in ASK if k[0] == dom.key]
        if kind == "ask_only":
            keys = [k for k in keys if direct(g.tool(0, k[1]))]
        else:
            keys = [k for k in keys if not direct(g.tool(0, k[1]))]
        if not keys:
            raise Skip("không có mẫu hỏi lại")
        k = rng.choice(keys)
        miss, tpl = rng.choice(ASK[k])
        if k == ("split", "request") and not g.contacts:
            raise Skip("không có danh bạ")
        if k == ("cinema", "book"):
            return g.write(0, "book", tpl=tpl, drop=miss), "ask_after_read"
        return g.write(0, k[1], tpl=tpl, drop=miss), "ask_missing" if kind == "ask_only" else "ask_after_read"
    if kind == "act_ask":
        firsts = g.tools_where(0, direct) + g.tools_where(0, refwrite)
        t = rng.choice(firsts)
        u = g.write(0, t.key)
        cand = [
            (i, k)
            for i in range(len(g.doms))
            for k in ASK
            if k[0] == g.doms[i].key and k[1] != t.key and not (k == ("split", "request") and not g.contacts)
        ]
        if not cand:
            raise Skip("không có vế hỏi")
        i, k = rng.choice(cand)
        miss, tpl = rng.choice(ASK[k])
        u2 = g.write(i, k[1], tpl=tpl, drop=miss)
        return u.add(rng.choice(vi.JOIN)).extend(u2), "act_then_ask"
    if kind == "stop":
        return stop(g)
    raise ValueError(kind)


def cond(g: Gen) -> tuple[Utt, str]:
    rng, dom, p = g.rng, g.doms[0], g.p
    use_hist = rng.random() < 0.25  # điều kiện đã có trong lịch sử: không đọc lại (luật 4)
    st_tool = next((t for t in dom.tools if t.op == "status"), None)
    if use_hist and st_tool is not None:
        r = g.read_status(0)
        q = rng.choice(st_tool.say) if st_tool.say else "xem trạng thái giúp {p}"
        g.hist_turn(finish(render(q, {}, p), rng, p).text, status_reply(dom, r), g.flush_hist())

    def check():
        return g.status(0) if (use_hist and st_tool is not None) else g.read_status(0)

    if dom.key == "garden":
        th = rng.choice([20, 30, 40, 50])
        st = check()
        ok = st["tank_percent"] > th
        thsay = S.integer(th, "phần trăm", rng)
        zone = rng.choice(st["zones"])
        u = render(
            "nếu bồn nước còn trên {th} thì tưới {obj}, không thì thôi",
            {"th": (thsay, "value"), "obj": (S.Say(zone), "entity")},
            p,
        )
        if ok:
            t = g.tool(0, "water")
            g.call(g.tname(0, "water"), {A.pname(g.metas[0]["names"], t, t.params[0]): zone})
        return u, f"cond_{ok}"
    if dom.key == "heater":
        st = check()
        if rng.random() < 0.5:
            u = render("nếu bình nóng lạnh đang tắt thì bật lên giúp {p}", {}, p)
            ok = st["power"] == "off"
            if ok:
                t = g.tool(0, "power")
                g.call(g.tname(0, "power"), {A.pname(g.metas[0]["names"], t, t.params[0]): "on"})
        else:
            T1, T2 = rng.choice([40, 45, 50]), rng.choice([55, 60, 65])
            u = render(
                "nước nóng chưa tới {a} thì chỉnh lên {b} nha",
                {"a": (S.integer(T1, "độ", rng), "value"), "b": (S.integer(T2, "độ", rng), "value")},
                p,
            )
            ok = st["current_temp"] < T1
            if ok:
                t = g.tool(0, "temp")
                g.call(g.tname(0, "temp"), {A.pname(g.metas[0]["names"], t, t.params[0]): T2})
        return u, f"cond_{ok}"
    if dom.key == "parking":
        st = check()
        if rng.random() < 0.5:
            months = rng.randint(1, 3)
            veh = g.pick_item(0, "vehicles")
            u = render(
                "nếu tài khoản gửi xe còn đủ tiền thì gia hạn vé tháng cho {obj} {m}, không đủ thì thôi",
                {"obj": (S.Say(veh["name"]), "entity"), "m": (S.integer(months, "tháng", rng), "value")},
                p,
            )
            ok = st["balance"] >= st["price_per_month"] * months
            if ok:
                g.read_ref(0, "vehicles", veh["name"])
                t = g.tool(0, "buy")
                n = g.metas[0]["names"]
                g.call(g.tname(0, "buy"), {A.pname(n, t, t.params[0]): veh["id"], A.pname(n, t, t.params[1]): months})
        else:
            X = rng.choice([100000, 200000, 300000, 500000])
            amt = rng.choice([200000, 300000, 500000])
            src = rng.choice(["bank", "ewallet"])
            t = g.tool(0, "topup")
            u = render(
                "nếu tài khoản gửi xe dưới {x} thì nạp thêm {a} {s}",
                {
                    "x": (S.money(X, rng), "value"),
                    "a": (S.money(amt, rng), "value"),
                    "s": (S.Say(rng.choice(t.params[1].say[src])), "value"),
                },
                p,
            )
            ok = st["balance"] < X
            if ok:
                n = g.metas[0]["names"]
                g.call(g.tname(0, "topup"), {A.pname(n, t, t.params[0]): amt, A.pname(n, t, t.params[1]): src})
        return u, f"cond_{ok}"
    if dom.key == "school":
        fees = g.items(0, "fees")
        f = g.pick_item(0, "fees")
        src = rng.choice(["bank", "ewallet"])
        t = g.tool(0, "pay")
        u = render(
            "nếu {obj} chưa đóng thì đóng {s}, đóng rồi thì thôi",
            {"obj": (S.Say(f["title"]), "entity"), "s": (S.Say(rng.choice(t.params[1].say[src])), "value")},
            p,
        )
        g.read_ref(0, "fees", f["title"])
        ok = f["status"] == "unpaid"
        if ok:
            n = g.metas[0]["names"]
            g.call(g.tname(0, "pay"), {A.pname(n, t, t.params[0]): f["id"], A.pname(n, t, t.params[1]): src})
        _ = fees
        return u, f"cond_{ok}"
    if dom.key == "split":
        reqs = g.items(0, "requests")
        if rng.random() < 0.4:
            rng.choice(reqs)["status"] = "paid"  # trước lượt cuối: một khoản đã trả (state ban đầu, không qua call)
            for m in g.metas[:1]:
                m["state"] = copy.deepcopy(g.env.states[m["names"]["app"]])
        r = g.pick_item(0, "requests")
        u = render("nếu {obj} chưa trả tiền thì nhắc giúp {p}", {"obj": (S.Say(r["contact_name"]), "entity")}, p)
        g.read_ref(0, "requests", r["contact_name"])
        ok = r["status"] == "pending"
        if ok:
            t = g.tool(0, "remind")
            g.call(g.tname(0, "remind"), {A.pname(g.metas[0]["names"], t, t.params[0]): r["id"]})
        return u, f"cond_{ok}"
    raise Skip("không có điều kiện")


def status_reply(dom: A.Domain, r: dict) -> str:
    if dom.key == "garden":
        return f"Bồn nước còn {r['tank_percent']}%, tưới tự động đang {'bật' if r['auto'] else 'tắt'} ạ."
    if dom.key == "heater":
        return f"Bình đang {'bật' if r['power'] == 'on' else 'tắt'}, nước {r['current_temp']} độ ạ."
    if dom.key == "parking":
        return f"Tài khoản gửi xe còn {r['balance']:,} đồng, vé tháng {r['price_per_month']:,} đồng ạ.".replace(
            ",", "."
        )
    return "Dạ đây ạ."


def ambiguous(g: Gen, pick: bool) -> tuple[Utt, str]:
    """Hai mục cùng tên khác thuộc tính: nói thiếu chi tiết → đọc rồi hỏi lại; nói đủ chi tiết → đọc rồi làm."""
    rng, dom, p = g.rng, g.doms[0], g.p
    ck, attr, says = {
        "pet": ("pets", "species", lambda v: f"con {v}"),
        "school": (
            "children",
            "class",
            lambda v: (
                re.sub(r"\d", lambda m: S.DIGITS[int(m.group())] + " ", v).strip() if rng.random() < S.P_WORDS else v
            ),
        ),
        "clinic": ("doctors", "specialty", lambda v: f"khoa {v}"),
    }[dom.key]
    items = g.items(0, ck)
    x = rng.choice(items)
    pool = {
        "pets": ["mèo", "chó"],
        "children": ["lớp 3B", "lớp lá 2", "lớp 5A"],
        "doctors": ["nội tổng quát", "nhi", "sản"],
    }[ck]
    other = rng.choice([v for v in pool if v != x[attr]])
    app = g.metas[0]["names"]
    dup = {**copy.deepcopy(x), attr: other}
    dup["id"] = g.env.apps[app["app"]].new_id(app["colls"][ck]["name"], app["colls"][ck]["prefix"])
    items.append(dup)
    g.metas[0]["state"] = copy.deepcopy(g.env.states[app["app"]])
    g.env = g.new_env()
    wt = [
        t
        for t in dom.tools
        if t.say
        and any(q.kind == "ref" and q.ref == ck for q in t.params)
        and not any(q.kind == "contact" for q in t.params)
    ]
    t = rng.choice(wt)
    target = rng.choice([x, dup])
    name = target[dom.colls[ck].key]
    if pick:
        detail = says(target[attr])
        tpl = rng.choice(t.say).replace("{obj}", "{obj} {detail}", 1)
        got = g.read_ref(0, ck, name)
        if sum(g.key_of(0, ck, y) == name for y in got) < 2:
            raise Skip("không mơ hồ")
        return g.write(
            0, t.key, target=target, tpl=tpl, extra={"detail": (S.Say(detail), "value")}, from_hist=True
        ), "pick_detail"
    # mơ hồ: đọc rồi hỏi ID
    vals = {}
    tpl = rng.choice(t.say)
    for q in t.params:
        slot = "obj" if (q.kind == "ref" and q.ref == ck) else q.canon
        if slot == "obj":
            vals["obj"] = (S.Say(name), "entity")
        elif ("{" + slot + "}") in tpl and q.kind not in ("ref", "contact", "nameref"):
            vals[slot] = (g.value(0, t, q)[1], "value")
    if any(s not in vals and s != "p" for s in SLOT.findall(re.sub(r"\[[^\]]*\]", "", tpl))):
        raise Skip("mẫu mơ hồ thiếu slot")
    got = g.read_ref(0, ck, name)
    if sum(g.key_of(0, ck, y) == name for y in got) < 2:
        raise Skip("không mơ hồ")
    ref = next(q for q in t.params if q.kind == "ref" and q.ref == ck)
    g.ask(g.tname(0, t.key), [A.pname(app, t, ref)])
    return render(tpl, vals, p), "ambiguous_ask"


def hist_ref(g: Gen) -> tuple[Utt, str]:
    """Lịch sử đã đọc danh sách (hoặc vừa tạo mục): lượt cuối dùng lại ID, không đọc lại (luật 7)."""
    rng, dom, p = g.rng, g.doms[0], g.p
    acts = g.tools_where(0, lambda t: t.op == "act" and t.say)
    t = rng.choice(acts)
    ck = t.coll
    if rng.random() < 0.5:
        f = A.finder_of(dom, ck)
        if f.op != "list":
            raise Skip("cần list")
        r = g.call(g.tname(0, f.key), {})
        lst = r[g.metas[0]["names"]["colls"][ck]["name"]]
        if not lst:
            raise Skip("rỗng")
        q = rng.choice(f.say) if f.say else f"xem {dom.colls[ck].noun} giúp {{p}}"
        names_txt = "; ".join(str(x.get(dom.colls[ck].key)) for x in lst[:4])
        g.hist_turn(finish(render(q, {}, p), rng, p).text, f"Dạ có: {names_txt}.", g.flush_hist())
        ok = g.ok_for(t, next(x for x in t.params if x.kind == "ref"))
        target = g.pick_item(0, ck, ok)
        return g.write(0, t.key, target=target, from_hist=True), "hist_list_then_act"
    # vừa tạo trong lịch sử, lượt cuối "huỷ cái đó"
    creates = [
        c
        for c in dom.tools
        if c.op == "create" and c.coll == ck and c.say and not any(x.kind == "contact" for x in c.params)
    ]
    if not creates:
        raise Skip("không có tool tạo")
    c = rng.choice(creates)
    before = {x["id"] for x in g.items(0, ck)}
    u0 = g.write(0, c.key)
    g.hist_turn(finish(u0, rng, p).text, rng.choice(REPLIES), g.flush_hist())
    new = next(x for x in g.items(0, ck) if x["id"] not in before)
    if rng.random() < 0.3:
        return Utt().add(rng.choice(vi.THANKS)), "thanks_after_action"
    tpl = (
        rng.choice(["huỷ {obj} vừa đặt luôn giúp {p}", "thôi huỷ {obj} đi", "à huỷ {obj} giúp {p}"])
        if "cancel" in t.key
        else rng.choice(t.say)
    )
    return g.write(
        0, t.key, target=new, tpl=tpl, from_hist=True, anaphor=rng.choice(["cái đó", "cái vừa rồi", "cái lịch đó"])
    ), "hist_create_then_act"


def read_only(g: Gen) -> tuple[Utt, str]:
    rng, dom, p = g.rng, g.doms[0], g.p
    if dom.key == "cinema" and rng.random() < 0.7:
        movies = sorted({x["movie"] for x in g.items(0, "showtimes")})
        m = rng.choice(movies)
        f = A.finder_of(dom, "showtimes")
        g.call(g.tname(0, f.key), {g.metas[0]["names"]["params"][f"{f.key}.$q"]: m})
        return render(
            rng.choice(["phim {obj} có suất nào không", "{obj} chiếu mấy giờ vậy", "xem lịch chiếu phim {obj}"]),
            {"obj": (S.Say(m), "entity")},
            p,
        ), "read_search"
    if dom.key == "clinic" and rng.random() < 0.6:
        spec = rng.choice(sorted({x["specialty"] for x in g.items(0, "doctors")}))
        f = A.finder_of(dom, "doctors")
        g.call(g.tname(0, f.key), {g.metas[0]["names"]["params"][f"{f.key}.$q"]: spec})
        return render(
            rng.choice(["có bác sĩ {obj} nào không", "tìm giúp {p} bác sĩ khoa {obj}"]),
            {"obj": (S.Say(spec), "entity")},
            p,
        ), "read_search"
    ts = g.tools_where(0, lambda t: t.op in ("list", "status") and t.say)
    if not ts:
        raise Skip("không có câu hỏi")
    t = rng.choice(ts)
    g.call(g.tname(0, t.key), {})
    return render(rng.choice(t.say), {}, p), f"read_{t.op}"


def stop(g: Gen) -> tuple[Utt, str]:
    rng, dom, p = g.rng, g.doms[0], g.p
    mode = rng.choice(["cancel", "cancel", "unsupported", "unsupported", "thanks", "answered"])
    if mode == "cancel":
        ts = g.tools_where(0, direct) or g.tools_where(0, refwrite)
        t = rng.choice(ts)
        u = g.write(0, t.key)
        g.turn, g.cache = [], {}  # bỏ cả yêu cầu: không gọi gì
        g.env = g.new_env()
        return u.add(rng.choice(vi.CANCEL).replace("{p}", p)), "stop_cancel"
    if mode == "unsupported":
        installed = {m["domain"] for m in g.metas}
        k = rng.choice([d for d in UNSUPPORTED_DOMS if d not in installed])
        d = A.BY_KEY[k]
        t = rng.choice([t for t in d.tools if t.say and t.op not in ("list", "status", "search")])
        # mượn một app tạm để dựng câu, không cài vào mẫu
        tmp = Gen(g.ctx, g.host, rng, "tmp")
        tmp.setup([k])
        u = tmp.write(0, t.key) if not any(x.kind == "contact" for x in t.params) else None
        if u is None:
            raise Skip("không dựng được câu")
        return u, "stop_unsupported"
    if mode == "thanks":
        ts = g.tools_where(0, direct) or g.tools_where(0, refwrite)
        u0 = g.write(0, rng.choice(ts).key)
        g.hist_turn(finish(u0, rng, p).text, rng.choice(REPLIES), g.flush_hist())
        return Utt().add(rng.choice(vi.THANKS)), "stop_thanks"
    st = next((t for t in dom.tools if t.op in ("status", "list") and t.say), None)
    if st is None:
        raise Skip("không có câu hỏi")
    q = rng.choice(st.say)
    r = g.call(g.tname(0, st.key), {})
    reply = status_reply(dom, r) if st.op == "status" else "Dạ, em liệt kê ở trên rồi ạ."
    g.hist_turn(finish(render(q, {}, p), rng, p).text, reply, g.flush_hist())
    return render(rng.choice([q, "vậy hả, " + q]), {}, p), "stop_answered_in_history"


# ---------------------------------------------------------------------------------------------------------------------
def _wb(x: str) -> str:
    return r"(?<!\w)" + re.escape(x) + r"(?!\w)"


def noisy(u: Utt, ctx, rng: random.Random, fuzzy_ents: set) -> tuple[dict, dict]:
    """(transcript lượt cuối kiểu ASR, thống kê nhiễu). Bản chính đúng phần lớn; ~1/7 mẫu có thực thể sai ở bản chính nhưng
    đúng ở một phương án (tỉ lệ thật 14–19%, AUDIT_1115 #3); vài % thực thể sai ở mọi bản khi app đọc bằng list/status
    (danh sách cho thấy tên đúng, chọn mục gần nhất). Thực thể sửa theo đúng vị trí mảnh, không thay chuỗi con."""
    clean = u.text
    spans = u.spans()
    prot = [(a, b) for a, b, *_ in spans]
    primary = clean
    info = {"primary_noise": False, "entity_mode": "none", "n_alts": 0}
    ents = [(a, b, t) for a, b, k, s, t in spans if k == "entity"]
    wrong = None
    r = rng.random()
    fz = [e for e in ents if e[2] in fuzzy_ents]
    if ents and (r < 0.25 or (fz and r < 0.30)):
        a, b, t = rng.choice(ents) if r < 0.25 else rng.choice(fz)
        words = t.split()
        cand = [
            i
            for i, w in enumerate(words)
            if len(w) > 1 and w.lower() not in ("bé", "anh", "chị", "cô", "chú", "bác", "sĩ", "xe")
        ]
        if cand and len(re.findall(_wb(t), clean)) == 1:
            i = rng.choice(cand)
            w2 = S.corrupt_word(words[i], ctx.err, rng)
            if w2 and w2 != words[i] and (S.fold(w2) != S.fold(words[i]) or rng.random() < 0.3):
                bad = " ".join(words[:i] + [w2] + words[i + 1 :])
                primary = clean[:a] + bad + clean[b:]
                d = len(bad) - len(t)
                prot = [(x if x < a else x + d, y if y <= a else y + d) for x, y in prot]
                wrong = (t, bad)
                info["entity_mode"] = "all_wrong" if r >= 0.25 else "fix_in_alt"
    if rng.random() < 0.35:
        x = S._sub_once(primary, ctx.err, rng, prot, min_count=2)
        if x:
            primary, info["primary_noise"] = x, True
    k = rng.choices([0, 1, 2, 3], [0.04, 0.22, 0.29, 0.45])[0]
    if info["entity_mode"] == "fix_in_alt":
        k = max(k, 1)
    alts, tries = [], 0
    nums = [(t, s) for a, b, kd, s, t in spans if s is not None and s.digits and s.digits != t]
    vals = [x[4] for x in spans] + ([wrong[1]] if wrong else [])
    while len(alts) < k and tries < 25:
        tries += 1
        cand = primary
        for _ in range(rng.choice([1, 1, 2])):
            if nums and rng.random() < 0.35:
                t, s = rng.choice(nums)
                cand = re.sub(_wb(t), lambda m: s.digits, cand, count=1)
            else:
                prot2 = [(m.start(), m.end()) for t in vals for m in re.finditer(_wb(t), cand)]
                cand = S._sub_once(cand, ctx.nb, rng, prot2) or cand
        if cand != primary and cand not in alts:
            alts.append(cand)
    has_fix = lambda a: re.search(_wb(wrong[0]), a) is not None
    if info["entity_mode"] == "fix_in_alt":
        if alts:
            j = rng.randrange(len(alts))
            alts[j] = re.sub(_wb(wrong[1]), lambda m: wrong[0], alts[j], count=1)
        if not any(has_fix(a) for a in alts):
            alts = [clean] + alts[:2]
    if len(alts) == 3 and rng.random() < 0.25:  # 25% đầu vào public có 4 phương án, cái đầu từ ASR thứ hai
        second = clean if info["entity_mode"] != "all_wrong" else primary
        for t, s in nums:
            second = re.sub(_wb(t), lambda m: s.digits, second, count=1)
        if second not in alts and second != primary:
            alts = [second] + alts
    if info["entity_mode"] == "all_wrong":
        alts = [re.sub(_wb(wrong[0]), lambda m: wrong[1], a) for a in alts]
        alts = [a for a in dict.fromkeys(alts) if a != primary]
    info["n_alts"] = len(alts)
    if wrong:
        info["entity"] = {"gold": wrong[0], "heard": wrong[1], "fix_in": sum(has_fix(a) for a in alts)}
        assert not has_fix(primary) or re.search(_wb(wrong[0]), wrong[1])
        assert info["entity_mode"] != "fix_in_alt" or info["entity"]["fix_in"] >= 1
        assert info["entity_mode"] != "all_wrong" or info["entity"]["fix_in"] == 0 or re.search(_wb(wrong[0]), wrong[1])
    return {"final": primary, "alts": alts}, info


def light_noise(text: str, ctx, rng: random.Random) -> str:
    if rng.random() < 0.3:
        return S._sub_once(text, ctx.err, rng, [], min_count=2) or text
    return text


# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    pass


def attempt(ctx, rng: random.Random, index: int) -> dict | None:
    kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
    host = rng.choice(ctx.hosts)
    doms = [d for d in A.DOMAINS if CAPS[kind](d)]
    if not doms:
        return None
    first = rng.choice(doms)
    keys = [first.key] + ([rng.choice([d.key for d in A.DOMAINS if d is not first])] if rng.random() < 0.4 else [])
    g = Gen(ctx, host, rng, kind)
    try:
        g.setup(keys)
        u, sub = scenario(g, kind)
    except (Skip, IndexError):  # IndexError: domain không có tool hợp kịch bản
        return None
    calls, results = [c for c, _ in g.turn], [r for _, r in g.turn]
    if sub not in ("stop_thanks", "thanks_after_action"):
        u = finish(u, rng, g.p)
    tr, info = noisy(u, ctx, rng, fuzzy_ents=set() if sub in ("ambiguous_ask", "pick_detail") else g.fuzzy_ents)
    # lịch sử: của mẫu chủ (thật) + lượt tổng hợp; bản chép lịch sử: ASR theo fold của mẫu chủ, lượt tổng hợp nhiễu nhẹ
    hist = copy.deepcopy(host.get("history", [])) + g.synth_hist
    turns = dict(ctx.asr_turns.get(host["id"], {}))
    k_host = sum(1 for t in host.get("history", []) if t.get("role") == "user")
    for j in range(1, k_host + 1):
        turns.setdefault(f"u{j}", [t for t in host["history"] if t.get("role") == "user"][j - 1]["audio_text"])
    j = k_host
    for t in g.synth_hist:
        if t["role"] == "user":
            j += 1
            turns[f"u{j}"] = light_noise(t["audio_text"], ctx, rng)
    tr["turns"] = turns
    transcript = {"final": tr["final"], "turns": turns, "alts": {"final": tr["alts"]} if tr["alts"] else {}}
    sample = {
        "id": f"synth3_{index:05d}",
        "audio_text": u.text,
        "context": {
            **{k: v for k, v in host["context"].items() if k != "installed_apps"},
            "installed_apps": g.installed,
        },
        "history": hist,
        "label": {"tool_calls": calls, "tool_results": results},
        "synth3": {
            "host": host["id"],
            "kind": kind,
            "sub": sub,
            "apps": g.metas,
            "state_hash": g.env.get_state_hash(),
            "transcript": transcript,
            "noise": info,
        },
    }
    return sample


def replay(sample: dict, guards: dict) -> list[str]:
    """Phát lại gold trên ToolEnv thật dựng lại từ metadata (qua middleware của config): rỗng là khớp."""
    sy = sample["synth3"]
    env = A.make_env(
        lambda: data.zaic_tools(SPLIT), data.data_root(), sy["host"], sy["apps"], data.installed_tools(sample)[:-1]
    )
    gd = GuardedEnv(env, SPLIT, sample, sy["transcript"]["final"], sy["transcript"]["turns"], **(guards or {}))
    out = []
    for c, r in zip(sample["label"]["tool_calls"], sample["label"]["tool_results"]):
        if c["name"] == "sys.ask_missing_info":
            got = gd.call(c["name"], c["arguments"])
        else:
            got = gd.call(c["name"], c["arguments"])
        if got != r:
            out.append(f"{c['name']}: kết quả khác ({got})")
    out += [w for x in gd.log for w in x.get("warnings", [])]
    if env.get_tool_calls() != sample["label"]["tool_calls"]:
        out.append("danh sách call khác gold")
    if env.get_state_hash() != sy["state_hash"]:
        out.append("state_hash khác gold")
    return out


def specs_of(sample: dict) -> dict:
    return {
        **data.tool_specs(SPLIT),
        **{
            t["name"]: t
            for m in sample["synth3"]["apps"]
            for t in A.catalog(A.BY_KEY[m["domain"]], m["names"])["tools"]
        },
    }


def load_ctx(seed: int) -> Ctx:
    ctx = Ctx()
    ctx.cfg = config.load(REPO / CFG)
    ctx.agent = build("agent", ctx.cfg["agent"])
    ctx.hosts = [s for s in splits.select(SPLIT, "train_rest") if not build_sft.heldout_in_history(s)]
    held = {s["id"] for sub in ("val_random", "val_app") for s in splits.select(SPLIT, sub)}
    assert not held & {s["id"] for s in ctx.hosts}
    ids = {s["id"] for s in ctx.hosts}
    ctx.err, ctx.nb = S.build_confusions(str(GOLD), str(ASR181), ids)
    ctx.asr_turns = {}
    for line in open(ASR181, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in ids:
            ctx.asr_turns[r["id"]] = r.get("turns", {})
    ctx.real_apps = [a["id"] for a in data.catalog(SPLIT)["apps"]]
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3003)
    ap.add_argument("--out", default=str(LM51 / "sft/synth_v3.jsonl"))
    a = ap.parse_args(argv)
    ctx = load_ctx(a.seed)
    rng = random.Random(a.seed)
    out, side = Path(a.out), Path(a.out).with_suffix(".samples.jsonl")
    stats = Counter()
    n, tries = 0, 0
    with out.open("w", encoding="utf-8") as fo, side.open("w", encoding="utf-8") as fs:
        while n < a.n:
            tries += 1
            if tries > a.n * 40:
                raise RuntimeError("quá nhiều lần dựng hỏng")
            s = attempt(ctx, rng, n + 1)
            if s is None:
                stats["skip"] += 1
                continue
            bad = replay(s, ctx.cfg.get("guards"))
            if bad:
                stats["drop_replay"] += 1
                stats[f"drop:{bad[0][:60]}"] += 1
                continue
            msgs = build_sft.conversation(ctx.agent, s, s["synth3"]["transcript"], None, specs=specs_of(s))
            n += 1
            stats[f"kind:{s['synth3']['kind']}"] += 1
            fo.write(
                json.dumps({"id": s["id"], "synth": f"v3:{s['synth3']['sub']}", "messages": msgs}, ensure_ascii=False)
                + "\n"
            )
            fs.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=1))
    print("→", out, side)


if __name__ == "__main__":
    main()
