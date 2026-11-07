"""Sinh mẫu tổng hợp: lời nói, lịch sử và quỹ đạo gold trên app tổng hợp, chạy thật qua `SynthEnv`.

Mỗi mẫu mượn ngữ cảnh của một mẫu `train_rest` (giờ, hồ sơ, app thật đã cài) rồi cài thêm 1 đến 3 app tổng hợp; yêu
cầu cuối nhắm vào app tổng hợp, nên mô hình phải đọc schema để chọn đúng app giữa nhiều app. Gold theo đúng quy ước
của train (prompts/rules/verified.md):

- ID chỉ lấy từ kết quả đọc hoặc lịch sử: tool list gọi với {}, tool search với query là đúng tên người dùng nói;
  ID đã có trong lịch sử thì dùng lại, không đọc lại.
- Thiếu tham số bắt buộc hoặc tên gọi khớp nhiều mục: làm xong phần đủ thông tin, gọi `sys.ask_missing_info` ở cuối.
- Hai mục cùng tên gốc ("Mích lớn", "Mích nhỏ") mà người dùng nói đủ chi tiết ("bé Mích lớn"): đọc rồi làm, không hỏi
  (`pick_detail`; đối trọng của `ask_ambiguous` để mô hình không hỏi lại thừa, docs/BASELINE.md mục synth).
- Không gọi gì khi người dùng bỏ yêu cầu, khi yêu cầu nhắm vào app chưa cài, hoặc chỉ là lời cảm ơn.
- Câu điều kiện: đọc để kiểm tra trước, điều kiện không thoả thì dừng. Tự sửa giữa câu: chỉ làm phần chốt cuối.
- Nhiều việc trong một câu: làm theo đúng thứ tự nói.
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from typing import Callable

from .. import data
from ..guards import GuardedEnv
from ..scoring import fold
from . import apps, vi
from .domains import DOMAINS

KINDS = {"create": 18, "write": 20, "read": 8, "hist_list": 7, "hist_create": 5, "ask_missing": 8, "ask_ambiguous": 6,
         "pick_detail": 6, "cancel": 3, "unsupported": 3, "thanks": 2, "conditional": 5, "correction": 5, "multi": 10,
         "not_found": 2}
LIST_SAY = ["cho {p} xem danh sách {noun}", "kiểm tra giúp {p} {noun} hiện có", "{noun} của {p} có những gì",
            "xem {noun} giúp {p}", "liệt kê {noun} cho {p}"]
SEARCH_SAY = ["tìm {obj}", "tìm giúp {p} {obj}", "xem có {obj} không"]
GET_SAY = ["xem chi tiết {obj}", "mở thông tin {obj} giúp {p}", "{obj} ghi gì vậy, xem giúp {p}"]
CHECK_SAY = ["kiểm tra {obj} giúp {p}", "{obj} sao rồi, xem giúp {p}", "xem lại {obj}"]
ANAPHOR = ["cái đó", "cái vừa rồi", "nó"]
AMBIG = [("lớn", "nhỏ"), ("cũ", "mới"), ("1", "2"), ("ở nhà", "ở công ty")]
REMIND = ["gọi điện cho khách", "đóng tiền nhà", "mua sữa", "đi lấy đồ giặt", "chuẩn bị tài liệu", "đón con"]
SLOT = re.compile(r"\{(\w+)\}")


class Skip(Exception):
    """Kịch bản không dựng được với app và state này: thử kịch bản khác."""


def slots(tpl: str) -> tuple[set, set]:
    """(mọi chỗ trống, chỗ trống bắt buộc nằm ngoài [...]) của một mẫu câu, không tính {p}."""
    hard = set(SLOT.findall(re.sub(r"\[[^\]]*\]", "", tpl))) - {"p"}
    return set(SLOT.findall(tpl)) - {"p"}, hard


def render(tpl: str, spoken: dict) -> str:
    out = re.sub(r"\[([^\]]*)\]", lambda m: m.group(1) if set(SLOT.findall(m.group(1))) - {"p"} <= set(spoken) else "", tpl)
    return SLOT.sub(lambda m: m.group(0) if m.group(1) == "p" else spoken[m.group(1)], out)


@dataclass
class Val:
    gold: object
    spoken: str
    item: dict | None = None    # ref: mục đích
    said: str = ""              # ref: tên gọi đã nói (query của tool search)


@dataclass
class Intent:
    core: str
    emit: Callable[[], None]


def _harvest(obj, into: set):
    """ID của các mục đã thấy tận mắt (khoá "id"), không tính ID tham chiếu như restaurant_id: biết restaurant_id của
    một bàn chưa cho biết đó là nhà hàng nào."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k == "id":
                into.add(v)
            _harvest(v, into)
    elif isinstance(obj, list):
        for x in obj:
            _harvest(x, into)


class Builder:
    """Dựng một mẫu: chọn kịch bản, nói ra, rồi chạy gold trên môi trường để có kết quả và hash thật."""

    def __init__(self, host: dict, insts: list[dict], states: dict, installed: dict, rng: random.Random):
        self.host, self.insts, self.states, self.installed, self.rng = host, insts, states, installed, rng
        self.now = apps.now_of(host)
        self.p = vi.pronoun(host["context"]["user_profile"].get("name"), rng)
        self.env = None
        self.known: set[str] = set()
        self.turn: list = []
        self.done: dict = {}
        self.deferred: list = []
        self.history: list[dict] = []
        self.after = None  # state các app tổng hợp ngay sau lịch sử

    # -- môi trường và một lượt ---------------------------------------------------------------------------
    def tools_installed(self) -> list[str]:
        return [t for a in self.installed.values() for t in a["tools"]] + ["sys.ask_missing_info"]

    def call(self, name: str, args: dict):
        sig = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if sig in self.done:
            return self.done[sig]
        if self.env is None:
            self.env = apps.SynthEnv(self.host["id"], self.insts, self.states, self.tools_installed())
        r = self.env.call(name, args)
        if r.get("ok") is False or r.get("done") is False:
            raise Skip(f"{name} {args}: {r}")
        self.turn.append(({"name": name, "arguments": args}, r))
        self.done[sig] = r
        _harvest(r, self.known)
        return r

    def end_turn(self) -> tuple[list, list]:
        for name, args in self.deferred:
            self.call(name, args)
        calls, results = [c for c, _ in self.turn], [r for _, r in self.turn]
        self.turn, self.done, self.deferred = [], {}, []
        return calls, results

    def live(self, inst: dict) -> dict:
        """State hiện tại của app (sau các call đã chạy)."""
        return (self.env.states if self.env else self.states)[inst["app"]]

    # -- tra cứu ------------------------------------------------------------------------------------------
    def dtool(self, inst: dict, short: str):
        key = inst["exec"][short]["key"]
        return next(t for t in apps.domain(inst).tools if t.key == key)

    def shorts(self, inst: dict, ops: tuple) -> list[str]:
        return [s for s, e in inst["exec"].items() if e["op"] in ops]

    def pick(self, inst: dict, ops: tuple) -> str:
        xs = self.shorts(inst, ops)
        if not xs:
            raise Skip(f"app không có tool {ops}")
        return self.rng.choice(xs)

    def items(self, inst: dict, cn: str) -> list[dict]:
        return self.live(inst).get(inst["colls"][cn]["name"], [])

    def key(self, inst: dict, cn: str, item: dict) -> str:
        return apps.key_of(inst, cn, item, {inst["app"]: self.live(inst)})

    def speakable(self, inst: dict, cn: str) -> bool:
        cs = apps.domain(inst).colls[cn]
        return bool(cs.derived) or (cs.key and cs.fields[cs.key].kind == "text")

    def unique(self, inst: dict, cn: str, item: dict) -> bool:
        """Tên gọi của mục không nằm trong tên mục khác (tool search so chuỗi con không dấu)."""
        k = fold(self.key(inst, cn, item))
        return bool(k) and not any(k in fold(self.key(inst, cn, x)) for x in self.items(inst, cn) if x is not item)

    def obj(self, inst: dict, cn: str, item: dict | None = None, key: str | None = None) -> tuple[str, str]:
        """(cụm từ gọi một mục, tên đã nói): "bé Mích", "lịch tiêm phòng dại của bé Mích"."""
        cs = apps.domain(inst).colls[cn]
        k = key if key is not None else self.key(inst, cn, item)
        tpl = self.rng.choice(cs.say)
        kf = apps.domain(inst).colls[cs.fields[cs.derived].ref] if cs.derived else cs
        said = vi.low(k) if "{lkey}" in tpl or kf.fields[kf.key].lower else k
        phrase = tpl.replace("{key}", said).replace("{lkey}", said)
        parent = next((fn for fn, f in cs.fields.items() if f.kind == "ref" and fn != cs.derived), None)
        if item is not None and parent and self.rng.random() < 0.3:
            pf = cs.fields[parent]
            pid = item.get(inst["colls"][cn]["fields"][parent])
            pitem = next((x for x in self.items(inst, pf.ref) if x["id"] == pid), None)
            if pitem is not None:
                phrase += " của " + self.obj(inst, pf.ref, pitem)[0]
        return phrase, said

    # -- đọc để có ID --------------------------------------------------------------------------------------
    def finder(self, inst: dict, cn: str, query: str | None = None):
        short = inst["colls"][cn]["finder"]
        ex = inst["exec"][short]
        args = {ex["params"]["$query"]: query} if ex["op"] == "search" and query else {}
        return self.call(f"{inst['app']}.{short}", args)

    def need(self, inst: dict, cn: str, item: dict, said: str):
        if item["id"] in self.known:
            return
        self.finder(inst, cn, said)
        if item["id"] not in self.known:
            raise Skip(f"đọc không thấy {item['id']}")

    # -- giá trị nói ra ----------------------------------------------------------------------------------
    def value(self, inst: dict, cn: str, fn: str, avoid=None) -> Val:
        rng, now = self.rng, self.now
        cs = apps.domain(inst).colls[cn]
        f = cs.fields[fn]
        for _ in range(10):
            v = self._value(inst, cn, cs, fn, f, rng, now)
            if v is not None and v.gold != avoid and (v.item is None or v.item.get("id") != avoid):
                return v
        raise Skip(f"không chọn được giá trị cho {fn}")

    def _value(self, inst, cn, cs, fn, f, rng, now) -> Val | None:
        k = f.kind
        if k == "text":
            if fn == cs.key:
                have = {fold(self.key(inst, cn, x)) for x in self.items(inst, cn)}
                pool = [x[fn] for x in cs.pool if not any(fold(x[fn]) in h or h in fold(x[fn]) for h in have)]
            else:
                pool = list(f.pool)
            if not pool:
                return None
            v = rng.choice(pool)
            return Val(v, vi.low(v) if f.lower else v)
        if k == "enum":
            v = rng.choice(list(f.values))
            return Val(v, rng.choice(f.values[v]))
        if k == "int":
            step = 1 if f.hi <= 20 else 5 if f.hi <= 200 else 100
            v = rng.randint(f.lo // step, f.hi // step) * step or f.lo
            return Val(v, f"{vi.num(v, rng)} {rng.choice(f.unit)}" if f.unit else vi.num(v, rng))
        if k == "money":
            v = vi.money_value(rng, f.lo, f.hi)
            return Val(v, vi.money(v, rng))
        if k == "bool":
            v = rng.random() < 0.5
            return Val(v, rng.choice(f.values[v]))
        if k == "date":
            d = rng.randint(f.lo, f.hi)
            return Val(vi.iso_date(now, d), vi.day(now, d, rng))
        if k == "datetime":
            d, h, m = vi.pick_dt(now, rng, max(f.lo, 0), f.hi)
            return Val(vi.iso_dt(now, d, h, m), vi.when(now, d, h, m, rng))
        if k == "ref":
            cands = [x for x in self.items(inst, f.ref) if self.unique(inst, f.ref, x)]
            if not cands:
                return None
            item = rng.choice(cands)
            phrase, said = self.obj(inst, f.ref, item)
            return Val(item["id"], phrase, item, said)
        return None

    def choose(self, templates: list, must: set, may: set, need_obj: bool) -> tuple[str, set]:
        cands = []
        for t in templates:
            alls, hard = slots(t)
            if ("obj" in alls) != need_obj or must - alls or hard - must - may - {"obj"}:
                continue
            cands.append(t)
        if not cands:
            raise Skip("không có mẫu câu")
        t = self.rng.choice(cands)
        alls, hard = slots(t)
        said = set(must) | (hard & may) | {s for s in (alls & may) - hard if self.rng.random() < 0.5}
        return t, said

    # -- ý định ------------------------------------------------------------------------------------------
    def create(self, inst: dict, short: str, omit: bool = False, fix: bool = False, ambiguous: bool = False,
               detail: bool = False) -> Intent:
        """Tạo mới. omit: bỏ một tham số bắt buộc rồi hỏi lại; fix: nói nhầm một giá trị rồi sửa;
        ambiguous: mục cha gọi bằng tên khớp hai mục, đọc xong thì hỏi lại ID;
        detail: mục cha có mục khác cùng tên gốc nhưng người dùng nói đủ tên, đọc xong thì làm (không hỏi)."""
        t, rng = self.dtool(inst, short), self.rng
        cn, pm, full = t.coll, inst["exec"][short]["params"], f"{inst['app']}.{short}"
        cs = apps.domain(inst).colls[cn]
        req = [p.rstrip("!") for p in t.params if p.endswith("!")]
        opt = [p for p in t.params if not p.endswith("!")]
        missing = None
        if omit:
            cands = [p for p in req if cs.fields[p].kind != "ref" and any(p not in slots(x)[1] and p in slots(x)[0] for x in t.say)]
            if not cands:
                raise Skip("không có tham số bỏ được")
            missing = rng.choice(cands)
        tpl, said = self.choose(t.say, set(req) - {missing}, set(opt), False)
        vals = {fn: self.value(inst, cn, fn) for fn in said}
        spoken = {fn: v.spoken for fn, v in vals.items()}
        amb = None
        if ambiguous or detail:
            refs = [fn for fn in said if cs.fields[fn].kind == "ref" and not apps.domain(inst).colls[cs.fields[fn].ref].derived]
            if not refs:
                raise Skip("không có mục cha để gây nhập nhằng")
            fn = rng.choice(refs)
            base = self.make_ambiguous(inst, cs.fields[fn].ref, vals[fn].item)
            if detail:  # tên đầy đủ sau khi đổi ("Mích lớn"): gold đọc bằng tên đó rồi dùng ID
                phrase, full_said = self.obj(inst, cs.fields[fn].ref, vals[fn].item)
                spoken[fn] = phrase
                vals[fn] = Val(vals[fn].gold, phrase, vals[fn].item, full_said)
            else:
                spoken[fn], said = self.obj(inst, cs.fields[fn].ref, None, base)
                amb = (fn, said)
        if fix:
            cands = [fn for fn in said if cs.fields[fn].kind in ("datetime", "date", "int", "money", "enum")]
            if not cands:
                raise Skip("không có giá trị để nói nhầm")
            fn = rng.choice(cands)
            wrong = self.value(inst, cn, fn, avoid=vals[fn].gold)
            if wrong.spoken == vals[fn].spoken:
                raise Skip("nói nhầm trùng chữ")
            spoken[fn] = wrong.spoken + rng.choice(vi.FIX) + vals[fn].spoken
        core = render(tpl, spoken)

        def emit():
            args = {}
            for fn, v in vals.items():
                if cs.fields[fn].kind == "ref":
                    if amb and amb[0] == fn:
                        self.finder(inst, cs.fields[fn].ref, amb[1])
                        continue
                    self.need(inst, cs.fields[fn].ref, v.item, v.said)
                args[pm[fn]] = v.gold
            if amb:
                self.deferred.append(("sys.ask_missing_info", {"tool_name": full, "params": [pm[amb[0]]]}))
            elif missing:
                self.deferred.append(("sys.ask_missing_info", {"tool_name": full, "params": [pm[missing]]}))
            else:
                self.call(full, args)
        return Intent(core, emit)

    def write(self, inst: dict, short: str, item: dict | None = None, omit: bool = False, fix: bool = False,
              ambiguous: bool = False, anaphor: bool = False, missing_key: str | None = None, detail: bool = False) -> Intent:
        """Sửa, xoá hoặc động từ riêng trên một mục gọi bằng tên. missing_key: gọi một tên không có (đọc rồi dừng).
        detail: có mục khác cùng tên gốc nhưng người dùng nói đủ tên ("Mích lớn"): đọc rồi làm, không hỏi."""
        t, rng = self.dtool(inst, short), self.rng
        ex = inst["exec"][short]
        cn, pm, full = t.coll, ex["params"], f"{inst['app']}.{short}"
        cc, cs = inst["colls"][cn], apps.domain(inst).colls[cn]
        if not self.speakable(inst, cn):
            raise Skip("mục không gọi được bằng tên")
        fmap = cc["fields"]
        ok = [x for x in self.items(inst, cn) if self.unique(inst, cn, x)
              and not (ex["op"] == "set" and all(x.get(fmap[k]) == (self.now.date().isoformat() if v == "$today" else v)
                                                  for k, v in ex["set"].items()))]
        if item is None:
            if not ok:
                raise Skip("không có mục phù hợp")
            item = rng.choice(ok)
        req = [p.rstrip("!") for p in t.params if p.endswith("!")]
        opt = [p for p in t.params if not p.endswith("!")]
        must = set(req)
        if not must and opt:  # tool sửa không có tham số bắt buộc: nói ít nhất một trường
            must = {rng.choice([p for p in opt if any(p in slots(x)[0] for x in t.say)])}
        missing = None
        if omit:
            cands = [p for p in req if any(p in slots(x)[0] and p not in slots(x)[1] for x in t.say)]
            if not cands:
                raise Skip("không có tham số bỏ được")
            missing = rng.choice(cands)
        tpl, said = self.choose(t.say, must - {missing}, set(opt), True)
        vals = {fn: self.value(inst, cn, fn, avoid=item.get(fmap.get(fn))) for fn in said}
        spoken = {fn: v.spoken for fn, v in vals.items()}
        amb_key = None
        if missing_key is not None:
            phrase, said_key = self.obj(inst, cn, None, missing_key)
        elif ambiguous or detail:
            if cs.derived:
                raise Skip("nhập nhằng chỉ dựng cho mục có tên riêng")
            base = self.make_ambiguous(inst, cn, item)
            if detail:
                phrase, said_key = self.obj(inst, cn, item)  # tên đầy đủ sau khi đổi
            else:
                amb_key = base
                phrase, said_key = self.obj(inst, cn, None, amb_key)
        elif anaphor or not self.unique(inst, cn, item):
            phrase, said_key = rng.choice(ANAPHOR), ""
        else:
            phrase, said_key = self.obj(inst, cn, item)
        if fix:
            others = [x for x in ok if x is not item]
            if not others:
                raise Skip("không có mục để nói nhầm")
            phrase = self.obj(inst, cn, rng.choice(others))[0] + rng.choice(vi.FIX) + phrase
        core = render(tpl, {**spoken, "obj": phrase})

        def emit():
            if missing_key is not None:
                self.finder(inst, cn, said_key)
                return
            if amb_key is not None:
                self.finder(inst, cn, said_key)
                self.deferred.append(("sys.ask_missing_info", {"tool_name": full, "params": [cc["idp"]]}))
                return
            self.need(inst, cn, item, said_key)
            if missing:
                self.deferred.append(("sys.ask_missing_info", {"tool_name": full, "params": [pm[missing]]}))
                return
            self.call(full, {cc["idp"]: item["id"], **{pm[fn]: v.gold for fn, v in vals.items()}})
        return Intent(core, emit)

    def make_ambiguous(self, inst: dict, cn: str, item: dict) -> str:
        """Đổi tên mục `item` và một mục khác thành "<tên> lớn", "<tên> nhỏ"; trả về phần tên chung."""
        if self.env is not None:
            raise Skip("state đã chạy")
        cs, cc = apps.domain(inst).colls[cn], inst["colls"][cn]
        others = [x for x in self.items(inst, cn) if x is not item]
        if not others or not cs.key:
            raise Skip("cần ít nhất hai mục")
        base = item[cc["fields"][cs.key]]
        a, b = self.rng.choice(AMBIG)
        other = self.rng.choice(others)
        item[cc["fields"][cs.key]] = f"{base} {a}"
        other[cc["fields"][cs.key]] = f"{base} {b}"
        if any(fold(base) in fold(self.key(inst, cn, x)) for x in self.items(inst, cn) if x is not item and x is not other):
            raise Skip("tên chung khớp quá hai mục")
        return base

    def read(self, inst: dict, cn: str | None = None) -> Intent:
        rng = self.rng
        cn = cn or rng.choice(list(inst["colls"]))
        cc = inst["colls"][cn]
        gets = [s for s, e in inst["exec"].items() if e["op"] == "get" and e["coll"] == cn]
        search = inst["exec"][cc["finder"]]["op"] == "search"
        items = [x for x in self.items(inst, cn) if self.unique(inst, cn, x) and x["id"] not in self.known]
        if items and self.speakable(inst, cn) and (gets or search or rng.random() < 0.4):
            item = rng.choice(items)
            phrase, said = self.obj(inst, cn, item)
            core = render(rng.choice(GET_SAY if gets else SEARCH_SAY if search else CHECK_SAY), {"obj": phrase})

            def emit():
                self.need(inst, cn, item, said)
                if gets:
                    self.call(f"{inst['app']}.{gets[0]}", {cc["idp"]: item["id"]})
            return Intent(core, emit)
        core = render(rng.choice(LIST_SAY), {"noun": cc["noun"]})
        return Intent(core, lambda: self.finder(inst, cn))

    def addon(self) -> Intent:
        """Một việc trên app thật của mẫu chủ: báo thức hoặc nhắc việc."""
        rng, now = self.rng, self.now
        tools = self.tools_installed()
        opts = [t for t in ("alarm.create", "reminders.create") if t in tools]
        if not opts:
            raise Skip("mẫu chủ không có báo thức hay nhắc việc")
        name = rng.choice(opts)
        d, h, m = vi.pick_dt(now, rng, 0, 6)
        when = vi.when(now, d, h, m, rng)
        if name == "alarm.create":
            core = rng.choice(["đặt báo thức {w}", "hẹn báo thức {w}"]).format(w=when)
            args = {"time": vi.iso_dt(now, d, h, m)}
        else:
            what = rng.choice(REMIND)
            core = rng.choice(["nhắc {p} {c} lúc {w}", "{w} nhắc {p} {c}"]).replace("{c}", what).replace("{w}", when)
            args = {"content": what, "time": vi.iso_dt(now, d, h, m)}
        return Intent(core, lambda: self.call(name, args))

    # -- kịch bản ------------------------------------------------------------------------------------------
    def any_intent(self, inst: dict, **kw) -> Intent:
        ops = self.rng.choice([("create",), ("update", "delete", "set")])
        short = self.rng.choice(self.shorts(inst, ops) or self.shorts(inst, ("create", "update", "delete", "set")))
        op = inst["exec"][short]["op"]
        return self.create(inst, short, **kw) if op == "create" else self.write(inst, short, **kw)

    def final(self, intents: list[Intent], tail: str = "") -> str:
        core = intents[0].core
        for it in intents[1:]:
            core += self.rng.choice([j for j in vi.JOIN if j.strip(" ,.").lower() != core.split()[-1]]) + it.core
        for it in intents:
            it.emit()
        return vi.utter(core + tail, self.p, self.rng, suffix=not tail)

    def hist(self, intent: Intent, text: str):
        """Một lượt user + assistant trong lịch sử, đã chạy trên môi trường."""
        user = vi.utter(intent.core, self.p, self.rng)
        intent.emit()
        calls, results = self.end_turn()
        self.history += [{"role": "user", "audio_text": user},
                         {"role": "assistant", "text": text, "tool_calls": calls, "tool_results": results}]
        self.after = copy.deepcopy(self.env.states)

    def scenario(self, kind: str) -> str:
        rng = self.rng
        inst = self.insts[0]
        if kind == "create":
            return self.final([self.create(inst, self.pick(inst, ("create",)))])
        if kind == "write":
            return self.final([self.write(inst, self.pick(inst, ("update", "delete", "set")))])
        if kind == "read":
            return self.final([self.read(inst)])
        if kind == "ask_missing":
            return self.final([self.any_intent(inst, omit=True)])
        if kind == "ask_ambiguous":
            return self.final([self.any_intent(inst, ambiguous=True)])
        if kind == "pick_detail":
            return self.final([self.any_intent(inst, detail=True)])
        if kind == "correction":
            return self.final([self.any_intent(inst, fix=True)])
        if kind == "not_found":
            short = self.pick(inst, ("update", "delete", "set"))
            cn = inst["exec"][short]["coll"]
            cs = apps.domain(inst).colls[cn]
            if not cs.key:
                raise Skip("không có tên để gọi nhầm")
            have = {fold(self.key(inst, cn, x)) for x in self.items(inst, cn)}
            absent = [x[cs.key] for x in cs.pool if not any(fold(x[cs.key]) in h or h in fold(x[cs.key]) for h in have)]
            if not absent:
                raise Skip("pool hết tên")
            return self.final([self.write(inst, short, missing_key=rng.choice(absent))])
        if kind == "cancel":
            it = self.any_intent(inst)
            return vi.utter(it.core + rng.choice(vi.CANCEL).replace("{p}", self.p), self.p, rng, suffix=False)
        if kind == "unsupported":
            return self.unsupported()
        if kind == "conditional":
            return self.conditional()
        if kind == "multi":
            return self.multi()
        if kind in ("hist_list", "hist_create", "thanks"):
            return self.with_history(kind)
        raise ValueError(kind)

    def multi(self) -> str:
        rng = self.rng
        inst = self.insts[0]
        first = self.any_intent(inst, omit=rng.random() < 0.2)
        r = rng.random()
        if r < 0.45:
            second = self.addon()
        elif r < 0.6 and len(self.insts) > 1:
            second = self.any_intent(self.insts[1])
        else:
            second = self.any_intent(inst)
        pair = [first, second] if rng.random() < 0.6 else [second, first]
        text = self.final(pair)
        writes = [c for c, _ in self.turn if self.kind(c["name"]) == "write"]
        if len(writes) + len(self.deferred) < 2:  # hai ý định trùng nhau (cùng mục, cùng việc): gold chỉ còn một
            raise Skip("hai ý định trùng")
        return text

    def kind(self, name: str) -> str:
        for i in self.insts:
            for t in i["catalog"]["tools"]:
                if t["name"] == name:
                    return t["kind"]
        return data.tool_kind(apps.SPLIT, name)

    def conditional(self) -> str:
        rng = self.rng
        inst = self.insts[0]
        if not apps.domain(inst).conds:
            raise Skip("app không có câu điều kiện")
        cond = rng.choice(apps.domain(inst).conds)
        cn, cc = cond["coll"], inst["colls"][cond["coll"]]
        items = [x for x in self.items(inst, cn) if self.unique(inst, cn, x)]
        if not items:
            raise Skip("không có mục")
        item = rng.choice(items)
        truth = rng.random() < 0.55
        f = apps.domain(inst).colls[cn].fields[cond["field"]]
        other = [v for v in f.values if v != cond["value"]]
        item[cc["fields"][cond["field"]]] = cond["value"] if truth else rng.choice(other)
        short = next(s for s, e in inst["exec"].items() if e["key"] == cond["tool"])
        ex, t = inst["exec"][short], self.dtool(inst, short)
        tcs = apps.domain(inst).colls[t.coll]
        vals = {fn: self.value(inst, t.coll, fn) for fn in cond.get("slots", [])}
        phrase, said = self.obj(inst, cn, item)
        core = render(rng.choice(cond["say"]), {"obj": phrase, **{fn: v.spoken for fn, v in vals.items()}})

        def emit():
            self.need(inst, cn, item, said)
            if not truth:
                return
            args = {ex["params"][fn]: v.gold for fn, v in vals.items()}
            args.update({ex["params"][fn]: v for fn, v in (cond.get("fixed") or {}).items()})
            if ex["op"] == "create":
                ref = next(fn for fn, ff in tcs.fields.items() if ff.kind == "ref" and ff.ref == cn)
                args[ex["params"][ref]] = item["id"]
            else:
                args[cc["idp"]] = item["id"]
            self.call(f"{inst['app']}.{short}", args)
        return self.final([Intent(core, emit)])

    def unsupported(self) -> str:
        """Yêu cầu nhắm vào một app tổng hợp không cài trong mẫu: không gọi gì."""
        rng = self.rng
        have = {i["domain"] for i in self.insts}
        dom = rng.choice([d for d in DOMAINS if d.ids[0] in UNSUPPORTED and d.ids[0] not in have])
        inst = apps.instance(dom, rng, set())
        other = Builder(self.host, [inst], {inst["app"]: apps.initial_state(inst, self.now, rng)}, {}, rng)
        other.p = self.p
        short = other.pick(inst, ("update", "delete", "set"))
        return vi.utter(other.write(inst, short).core, self.p, rng)

    def with_history(self, kind: str) -> str:
        rng = self.rng
        inst = self.insts[0]
        if kind == "hist_list":
            short = self.pick(inst, ("update", "delete", "set"))
            cn = inst["exec"][short]["coll"]
            if inst["exec"][inst["colls"][cn]["finder"]]["op"] != "list":
                raise Skip("lịch sử liệt kê cần tool list")
            keys = [self.key(inst, cn, x) for x in self.items(inst, cn)]
            self.hist(Intent(render(rng.choice(LIST_SAY), {"noun": inst["colls"][cn]["noun"]}), lambda: self.finder(inst, cn)),
                      f"Dạ, danh sách {inst['colls'][cn]['noun']}: {', '.join(keys)}." if keys else "Dạ, hiện chưa có gì ạ.")
            return self.final([self.write(inst, short)])
        creates = [s for s in self.shorts(inst, ("create",))
                   if self.speakable(inst, inst["exec"][s]["coll"])
                   and any(e["op"] in ("update", "delete", "set") and e["coll"] == inst["exec"][s]["coll"]
                           for e in inst["exec"].values())]
        if not creates:
            raise Skip("không có tool tạo rồi sửa")
        short = rng.choice(creates)
        cn = inst["exec"][short]["coll"]
        before = {x["id"] for x in self.items(inst, cn)}
        self.hist(self.create(inst, short), rng.choice(["Dạ, em đã làm xong rồi ạ.", "Dạ xong rồi ạ.", "Đã xong ạ."]))
        if kind == "thanks":
            return rng.choice(vi.THANKS)
        new = next(x for x in self.items(inst, cn) if x["id"] not in before)
        writes = [s for s, e in inst["exec"].items() if e["op"] in ("update", "delete", "set") and e["coll"] == cn]
        return self.final([self.write(inst, rng.choice(writes), item=new, anaphor=rng.random() < 0.5)])


def _installed(host: dict, insts: list[dict], rng: random.Random) -> dict:
    """installed_apps của mẫu chủ, chèn app tổng hợp vào vị trí ngẫu nhiên."""
    items = list(host["context"]["installed_apps"].items())
    for inst in insts:
        items.insert(rng.randint(0, len(items)), (inst["app"], {"tools": [t["name"] for t in inst["catalog"]["tools"]]}))
    return dict(items)


def real_app_ids() -> set[str]:
    """Tên app của train và public: app tổng hợp không được trùng."""
    out = {a["id"] for a in data.catalog("train")["apps"]}
    try:
        out |= {a["id"] for a in data.catalog("public_test")["apps"]}
    except FileNotFoundError:
        pass
    return out


def specs(sample: dict) -> dict:
    """Catalog cho prompt của mẫu tổng hợp: catalog train cộng các app tổng hợp của mẫu."""
    return {**data.tool_specs(apps.SPLIT), **{t["name"]: t for i in sample["synth"]["apps"] for t in i["catalog"]["tools"]}}


def transcript(sample: dict) -> dict:
    """Bản chép lời như tầng ASR `gold` trả về: lượt cuối và các lượt user trong lịch sử."""
    users = [t for t in sample.get("history", []) if t.get("role") == "user"]
    return {"final": sample["audio_text"], "turns": {f"u{k + 1}": u["audio_text"] for k, u in enumerate(users)}, "alts": {}}


def replay(sample: dict, guards: dict | None = None) -> list[str]:
    """Phát lại gold qua middleware trên môi trường dựng lại từ chính mẫu. Trả về các chỗ lệch; rỗng là đúng."""
    env = apps.SynthEnv.of(sample)
    g = GuardedEnv(env, apps.SPLIT, sample, sample["audio_text"], **(guards or {}))
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    out = [f"{c['name']}: kết quả khác" for c, r in zip(calls, results) if g.call(c["name"], c["arguments"]) != r]
    out += [w for x in g.log for w in x.get("warnings", [])]
    if env.get_tool_calls() != calls:
        out.append("danh sách call khác gold")
    if env.get_state_hash() != sample["synth"]["state_hash"]:
        out.append("state_hash khác gold")
    return out


UNSUPPORTED = ["plant_care", "watchlist", "photo_album", "home_wifi", "expense_tracker", "warranty"]


def _domains(kind: str, rng: random.Random) -> list:
    """App đích (đầu danh sách) và 0-2 app tổng hợp cài thêm cho cảnh."""
    k_extra = rng.choices([0, 1, 2], [0.45, 0.4, 0.15])[0]
    pool = [d for d in DOMAINS if d.conds] if kind == "conditional" else DOMAINS
    first = rng.choice(pool)
    return [first] + rng.sample([d for d in DOMAINS if d is not first], k_extra)


def generate(hosts: list[dict], n: int, seed: int = 0, kinds: dict | None = None) -> list[dict]:
    """n mẫu tổng hợp, mỗi mẫu mượn ngữ cảnh của một mẫu trong `hosts`. Tất định theo seed và thứ tự `hosts`."""
    rng = random.Random(seed)
    kinds = kinds or KINDS
    avoid = real_app_ids()
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 20:
            raise RuntimeError("quá nhiều lần dựng hỏng")
        kind = rng.choices(list(kinds), list(kinds.values()))[0]
        host = rng.choice(hosts)
        taken = avoid | set(host["context"]["installed_apps"])
        insts = []
        for d in _domains(kind, rng):
            inst = apps.instance(d, rng, taken)
            taken.add(inst["app"])
            insts.append(inst)
        now = apps.now_of(host)
        states = {i["app"]: apps.initial_state(i, now, rng) for i in insts}
        b = Builder(host, insts, states, _installed(host, insts, rng), rng)
        try:
            text = b.scenario(kind)
            calls, results = b.end_turn()
        except Skip:
            continue
        start = b.after if b.history else b.states  # state lúc người dùng nói câu cuối (lịch sử đã áp dụng)
        env = b.env or apps.SynthEnv(host["id"], insts, start, b.tools_installed())
        out.append({
            "id": f"synth_{len(out) + 1:05d}", "audio_text": text,
            "context": {**{k: v for k, v in host["context"].items() if k != "installed_apps"}, "installed_apps": b.installed},
            "history": b.history, "label": {"tool_calls": calls, "tool_results": results},
            "synth": {"host": host["id"], "kind": kind, "apps": insts, "state": copy.deepcopy(start),
                      "state_hash": env.get_state_hash()}})
    return out
