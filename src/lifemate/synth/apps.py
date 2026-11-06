"""Dựng một app tổng hợp từ `domains.py` (tên ngẫu nhiên, catalog, state ban đầu) và chạy tool của nó thật.

`instance()` trả về một dict thuần JSON: mục catalog đúng định dạng `<split>_tools_catalog.json`, cộng phần `exec`
(mỗi tool làm gì trên collection nào) để dựng lại môi trường từ chính mẫu đã lưu. `SynthEnv` có cùng giao diện với
phiên `zaic_tools` (`validate`, `call`, `get_tool_calls`, `get_state_hash`): call của app tổng hợp chạy trên lớp `App`
của `zaic_tools` với kiểm schema của chính `zaic_tools`, còn call của app thật chuyển cho ToolEnv của mẫu chủ. Vì vậy
kết quả tool và state_hash của mẫu tổng hợp là thật, không phải chép tay.
"""
from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta

from .. import data
from . import vi
from .domains import DOMAINS, Domain

SPLIT = "train"
DT_DESC = "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"
DATE_DESC = "YYYY-MM-DD, ví dụ 2026-11-29"


def _pick(rng, xs):
    return xs[0] if len(xs) == 1 else rng.choice(list(xs))


def _param_schema(f, app: str, colls: dict, rng: random.Random) -> dict:
    """JSON Schema của một tham số theo kiểu trường, cùng quy ước với catalog BTC (x-ref, x-match, mô tả ISO)."""
    k = f.kind
    if k == "text":
        s = {"type": "string", "x-match": "fuzzy"}
    elif k in ("enum", "status"):
        s = {"type": "string", "enum": list(f.values)}
    elif k == "int":
        s = {"type": "integer", "minimum": 1 if f.lo >= 1 else 0}
        if f.hi <= 10:
            s["maximum"] = f.hi
    elif k == "money":
        s = {"type": "integer", "minimum": 1000}
    elif k == "bool":
        s = {"type": "boolean"}
    elif k == "date":
        s = {"type": "string", "description": DATE_DESC}
        if rng.random() < 0.5:
            s["format"] = "date"
        return s
    elif k == "datetime":
        s = {"type": "string", "description": DT_DESC}
        if rng.random() < 0.5:
            s["format"] = "date-time"
        return s
    elif k == "ref":
        c = colls[f.ref]
        return {"type": "string", "description": f"ID {c['noun']}, lấy từ {app}.{c['finder']}", "x-ref": f"{app}.{c['name']}"}
    else:
        raise ValueError(k)
    if f.desc:
        s["description"] = f.desc
    return s


def instance(dom: Domain, rng: random.Random, taken: set[str]) -> dict:
    """Một lần dựng app: chọn tên app, collection, trường, tool; dựng mục catalog và bảng thực thi."""
    for _ in range(20):
        app = _pick(rng, dom.ids)
        if app not in taken:
            break
    else:
        raise ValueError(f"hết tên cho {dom.ids}")
    colls = {}
    for cn, c in dom.colls.items():
        name, sing = _pick(rng, c.names)
        fields = {fn: (_pick(rng, f.names) if f.names else fn) for fn, f in c.fields.items() if f.kind != "ref"}
        colls[cn] = {"name": name, "sing": sing, "prefix": _pick(rng, c.prefix), "noun": c.noun, "fields": fields,
                     "idp": f"{sing}_id"}
    for cn, c in dom.colls.items():  # tham số ref mang tên ID của collection đích
        for fn, f in c.fields.items():
            if f.kind == "ref":
                colls[cn]["fields"][fn] = colls[f.ref]["idp"]
        if c.derived:  # mục con gọi theo tên mục cha: lưu kèm tên đó, như API thật, để đọc danh sách là nhận ra
            colls[cn]["label"] = f"{colls[c.fields[c.derived].ref]['sing']}_name"
    tools, exec_ = [], {}
    used = set()
    for t in dom.tools:
        cc = colls[t.coll]
        for _ in range(10):
            short = _pick(rng, t.names).format(coll=cc["name"], sing=cc["sing"])
            if short not in used:
                break
        else:
            raise ValueError(f"trùng tên tool {t.key}")
        used.add(short)
        exec_[short] = {"key": t.key, "op": t.op, "coll": t.coll}
        if t.op in ("list", "search") and "finder" not in cc:
            cc["finder"] = short
    for cn, cc in colls.items():
        if "finder" not in cc:
            raise ValueError(f"{dom.ids[0]}.{cn} không có tool đọc")
    for t in dom.tools:
        short = next(s for s, e in exec_.items() if e["key"] == t.key)
        cc, cs = colls[t.coll], dom.colls[t.coll]
        props, req, pmap = {}, [], {}
        if t.op in ("get", "update", "delete", "set"):
            props[cc["idp"]] = {"type": "string", "description": f"ID {cc['noun']}, lấy từ {app}.{cc['finder']}",
                                "x-ref": f"{app}.{cc['name']}"}
            req.append(cc["idp"])
        if t.op == "search":
            q = _pick(rng, t.qnames)
            props[q] = {"type": "string", "description": f"Tên {cc['noun']}"}
            pmap["$query"] = q
        for p in t.params:
            fn, must = p.rstrip("!"), p.endswith("!")
            pn = cc["fields"][fn]
            props[pn] = _param_schema(cs.fields[fn], app, colls, rng)
            pmap[fn] = pn
            if must:
                req.append(pn)
        desc = t.desc + (f" (bắt buộc: {', '.join(req)})" if req and rng.random() < 0.3 else "")
        tools.append({"name": f"{app}.{short}", "kind": "read" if t.op in ("list", "search", "get") else "write",
                      "description": desc, "parameters": {"type": "object", "properties": props, "required": req},
                      "returns": f"{cc['name']}[]" if t.op in ("list", "search") else "", "app": app})
        exec_[short].update(params=pmap, set=t.set or {}, defaults=t.defaults or {}, by=t.by)
    if rng.random() < 0.3:
        rng.shuffle(tools)
    catalog = {"id": app, "name_vi": dom.vi, "group": dom.group, "description": _pick(rng, dom.desc), "tools": tools}
    return {"domain": dom.ids[0], "app": app, "catalog": catalog, "exec": exec_, "colls": colls}


def domain(inst: dict) -> Domain:
    return next(d for d in DOMAINS if d.ids[0] == inst["domain"])


def _value(f, now: datetime, rng: random.Random):
    """Giá trị ngẫu nhiên cho một trường lúc dựng state (không phải giá trị người dùng nói)."""
    k = f.kind
    if k == "text":
        return _pick(rng, f.pool) if f.pool else None
    if k == "enum":
        return rng.choice(list(f.values))
    if k == "status":
        return f.init
    if k == "int":
        return rng.randint(f.lo, f.hi)
    if k == "money":
        return vi.money_value(rng, f.lo, f.hi)
    if k == "bool":
        return rng.random() < 0.5
    if k == "date":
        return vi.iso_date(now, rng.randint(f.lo, f.hi))
    if k == "datetime":
        return vi.iso_dt(now, *vi.pick_dt(now, rng, max(f.lo, 0), f.hi))
    return None


def _ids(prefix: str, n: int, rng: random.Random) -> list[str]:
    """ID kiểu BTC: tuần tự "pet_01" hoặc số ngẫu nhiên "pet_731"."""
    if rng.random() < 0.5:
        return [f"{prefix}_{i + 1:02d}" for i in range(n)]
    return [f"{prefix}_{x}" for x in rng.sample(range(100, 9999), n)]


def initial_state(inst: dict, now: datetime, rng: random.Random) -> dict:
    """State ban đầu của app: mỗi collection vài mục lấy từ pool, collection con trỏ tới mục cha có thật."""
    dom = domain(inst)
    state: dict[str, list] = {}
    order = sorted(dom.colls, key=lambda cn: any(f.kind == "ref" for f in dom.colls[cn].fields.values()))
    for cn in order:
        cs, cc = dom.colls[cn], inst["colls"][cn]
        lo, hi = cs.n
        parents = None
        if cs.derived:  # tên gọi lấy theo mục cha: mỗi mục một cha khác nhau để còn gọi được
            parents = list(state[inst["colls"][cs.fields[cs.derived].ref]["name"]])
            rng.shuffle(parents)
            hi = min(hi, len(parents))
        n = rng.randint(min(lo, hi), hi)
        pool = [dict(x) for x in cs.pool]
        if cs.key and len(pool) >= n:
            pool = rng.sample(pool, n)
        else:
            pool = [dict(rng.choice(pool)) for _ in range(n)] if pool else []
        items = []
        for i, (base, iid) in enumerate(zip(pool, _ids(cc["prefix"], n, rng))):
            item = {"id": iid}
            for fn, f in cs.fields.items():
                if f.kind == "ref":
                    target = state[inst["colls"][f.ref]["name"]]
                    v = parents[i]["id"] if parents and fn == cs.derived else rng.choice(target)["id"]
                else:
                    v = base.get(fn, _value(f, now, rng))
                if v is not None:
                    item[cc["fields"][fn]] = v
                if fn == cs.derived:
                    pc = inst["colls"][f.ref]
                    pkey = dom.colls[f.ref].key
                    item[cc["label"]] = next(x for x in state[pc["name"]] if x["id"] == v)[pc["fields"][pkey]]
            items.append(item)
        state[cc["name"]] = items
    return state


def key_of(inst: dict, cn: str, item: dict, states: dict) -> str:
    """Tên người dùng gọi một mục: trường khoá, hoặc tên của mục cha với collection `derived`."""
    dom = domain(inst)
    cs, cc = dom.colls[cn], inst["colls"][cn]
    return item.get(cc["fields"][cs.key] if cs.key else cc["label"], "")


def make_app(zaic, session, inst: dict, state: dict):
    """App tổng hợp chạy trên lớp `App` của zaic_tools, dùng đúng các hàm đọc, tạo, sửa, xoá của nó."""
    dom = domain(inst)

    class SynthApp(zaic.App):
        name = inst["app"]

        def run(self, short: str, args: dict) -> dict:
            ex = inst["exec"][short]
            cn = ex["coll"]
            cc, cs = inst["colls"][cn], dom.colls[cn]
            coll, idp, fmap = cc["name"], cc["idp"], cc["fields"]
            pm = ex["params"]
            given = {fn: args.get(pn) for fn, pn in pm.items() if not fn.startswith("$")}
            drop = tuple(fmap[d] for d in cs.detail)
            has_get = any(e["op"] == "get" and e["coll"] == cn for e in inst["exec"].values())
            op = ex["op"]
            if op == "list":
                return self.read(coll, drop=drop if has_get else ())
            if op == "search":
                fields = (fmap[cs.key],) if cs.key else ()
                return self.read(coll, contains={fields: args.get(pm["$query"])}, drop=drop if has_get else ())
            if op == "get":
                item = self.find(coll, args.get(idp))
                return {"ok": True, cc["sing"]: item} if item else self.noop(f"không tìm thấy {idp}={args.get(idp)!r}")
            for fn, v in given.items():  # ID trỏ sang collection khác phải có thật
                f = cs.fields[fn]
                if f.kind == "ref" and v is not None:
                    tc = inst["colls"][f.ref]
                    if self.find(tc["name"], v) is None:
                        return self.noop(f"không tìm thấy {tc['idp']}={v!r}")
            fields = {fmap[fn]: v for fn, v in given.items() if v is not None}
            if cs.derived and given.get(cs.derived) is not None:
                pc = inst["colls"][cs.fields[cs.derived].ref]
                parent = self.find(pc["name"], given[cs.derived])
                fields[cc["label"]] = parent[pc["fields"][dom.colls[cs.fields[cs.derived].ref].key]]
            if op == "create":
                defaults = {fmap[k]: v for k, v in ex["defaults"].items()}
                for fn, f in cs.fields.items():
                    if f.kind == "status":
                        defaults.setdefault(fmap[fn], f.init)
                out = self.create(coll, cc["prefix"], defaults=defaults, **fields)
                return {"ok": True, "id": out["id"], cc["sing"]: out["item"]}
            if op == "upsert":
                by = fmap[ex["by"]]
                old = next((x for x in self.state.get(coll, []) if x.get(by) == fields.get(by)), None)
                if old is None:
                    out = self.create(coll, cc["prefix"], **fields)
                    return {"ok": True, "id": out["id"], cc["sing"]: out["item"]}
                return self.patch(coll, idp, old["id"], **fields)
            if op == "update":
                return self.patch(coll, idp, args.get(idp), **fields)
            if op == "delete":
                return self.delete(coll, idp, args.get(idp))
            if op == "set":
                item = self.find(coll, args.get(idp))
                if item is None:
                    return self.noop(f"không tìm thấy {idp}={args.get(idp)!r}")
                today = (session.now or "")[:10]
                upd = {fmap[k]: (today if v == "$today" else v) for k, v in ex["set"].items()}
                if all(item.get(k) == v for k, v in upd.items()):
                    return self.noop(f"{cc['noun']} {item['id']} không có gì để đổi")
                return self.patch(coll, idp, item["id"], set=upd)
            raise ValueError(op)

    return SynthApp(session, state)


class SynthEnv:
    """Phiên của một mẫu tổng hợp: app tổng hợp chạy ở đây, app thật chuyển cho ToolEnv của mẫu chủ."""

    def __init__(self, host_id: str, insts: list[dict], states: dict, installed: list[str]):
        self.zaic = data.zaic_tools(SPLIT)
        self.host = data.tool_env(SPLIT, host_id)
        self.now = self.host.now
        self.states = copy.deepcopy(states)
        self.insts = {i["app"]: i for i in insts}
        self.apps = {a: make_app(self.zaic, self, i, self.states[a]) for a, i in self.insts.items()}
        self.specs = {t["name"]: t for i in insts for t in i["catalog"]["tools"]}
        self.allowed = set(installed) | {"sys.ask_missing_info"}
        self.calls: list[dict] = []

    @classmethod
    def of(cls, sample: dict) -> "SynthEnv":
        sy = sample["synth"]
        return cls(sy["host"], sy["apps"], sy["state"], data.installed_tools(sample))

    def validate(self, name: str, arguments: dict | None = None) -> dict:
        if name.split(".", 1)[0] not in self.apps:
            return self.host.validate(name, arguments)
        err = self.zaic.schema_error(name, arguments or {}, self.specs, self.allowed)
        return {"ok": err is None, "error": err}

    def call(self, name: str, arguments: dict | None = None) -> dict:
        arguments = arguments or {}
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        app = name.split(".", 1)[0]
        if app not in self.apps:
            return self.host.call(name, arguments)
        err = self.zaic.schema_error(name, arguments, self.specs, self.allowed)
        if err is not None:
            return {"ok": False, "error": err}
        try:
            out = self.apps[app].run(name.split(".", 1)[1], arguments)
        except Exception as e:  # như Simulator: không ném lỗi vào vòng agent, call vẫn tính và không đổi gì
            out = {"ok": True, "done": False, "message": f"{type(e).__name__}: {e}"}
        return copy.deepcopy(out)

    def get_tool_calls(self) -> list[dict]:
        return copy.deepcopy(self.calls)

    def get_state(self) -> dict:
        return {**self.host.get_state(), **copy.deepcopy(self.states)}

    def get_state_hash(self) -> str:
        return self.zaic.state_hash(self.get_state())


def now_of(sample_or_time) -> datetime:
    t = sample_or_time if isinstance(sample_or_time, str) else sample_or_time["context"]["current_time"]
    return datetime.fromisoformat(t)


def days_between(now: datetime, iso: str) -> int:
    return (datetime.fromisoformat(iso[:10]).date() - now.date()) // timedelta(days=1)
