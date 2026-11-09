"""Các phương án agent. Mỗi agent nhận mẫu, transcript và môi trường có bảo vệ, rồi trả về vết từng bước.

    none          bỏ qua tầng agent (chỉ đo ASR)
    empty         không gọi gì: điểm sàn
    oracle        phát lại call gold của train: điểm trần, dùng để đo tác động của middleware
    react_json    ReAct, mỗi bước LLM trả một hành động JSON theo prompt tự viết
    native_tools  ReAct qua tool calling gốc của mô hình (API tools của OpenAI)
"""
from __future__ import annotations

import json
import re
import time

from .. import context, data, normalize, splits
from ..config import REPO
from ..numbers import calc
from ..registry import register
from .llm import LLM, LLMTimeout


class Agent:
    def run(self, split: str, sample: dict, transcript: dict, env) -> dict:
        raise NotImplementedError


@register("agent", "none")
class NoAgent(Agent):
    """Bỏ qua tầng agent: chỉ chạy và đo ASR."""

    def __init__(self, **_):
        pass


@register("agent", "empty")
class EmptyAgent(Agent):
    """Không gọi tool nào: điểm sàn (đúng ở các mẫu gold rỗng)."""

    def __init__(self, **_):
        pass

    def run(self, split, sample, transcript, env):
        return {"steps": []}


@register("agent", "oracle")
class OracleAgent(Agent):
    """Phát lại call gold qua middleware: trần điểm, cho thấy middleware làm mất bao nhiêu."""

    def __init__(self, **_):
        pass

    def run(self, split, sample, transcript, env):
        steps = []
        for c in sample["label"]["tool_calls"]:
            obs = env.call(c["name"], c["arguments"])
            steps.append({"action": "call", "name": c["name"], "arguments": c["arguments"], "observation": obs})
        return {"steps": steps}


def _render(template: str, values: dict) -> str:
    """Thay {{khoá}} trong prompt; không dùng str.format để JSON trong prompt không bị vỡ."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), "")), template)


def _read(path: str) -> str:
    p = REPO / path
    return p.read_text(encoding="utf-8") if p.exists() else ""


class LLMAgent(Agent):
    def __init__(self, llm: dict | None = None, prompt="prompts/react_json.md", rules="prompts/rules/verified.md",
                 timetable="two_weeks", schema="compact", history="full", ledger=True, max_steps=8,
                 show_alts=True, obs_chars=2500, calc=False, normalize_hint=False, **_):
        self.llm = LLM(**(llm or {}))
        self.template = _read(prompt)
        self.rules = _read(rules) if rules else ""
        self.opts = dict(timetable=timetable, schema=schema, history=history, ledger=ledger)
        self.max_steps = max_steps
        self.show_alts = show_alts
        self.obs_chars = obs_chars
        self.calc = calc
        self.normalize_hint = normalize_hint  # khối giá trị máy đọc từ lời nói (normalize.py) trong prompt

    def system_prompt(self, split, sample, specs=None, rules=None) -> str:
        """specs, rules: thay catalog và luật (bản đổi tên khi dựng dữ liệu SFT); mặc định lấy của agent."""
        o = self.opts
        return _render(self.template, {
            "rules": self.rules if rules is None else rules,
            "timetable": context.timetable(sample["context"]["current_time"], o["timetable"]),
            "profile": context.profile(sample),
            "schema": (context.schema(split, sample, o["schema"], specs) + (CALC_LINE if self.calc else "")
                       if o["schema"] != "none" else ""),
            "ledger": context.ledger(sample) if o["ledger"] else "",
        })

    def user_prompt(self, sample, transcript) -> str:
        parts = []
        hist = context.history(sample, transcript.get("turns", {}), self.opts["history"])
        if hist:
            parts.append("LỊCH SỬ HỘI THOẠI:\n" + hist)
        parts.append(f'YÊU CẦU CUỐI CỦA NGƯỜI DÙNG (bản chép lời):\n"{transcript["final"]}"')
        alts = transcript.get("alts", {}).get("final")
        if self.show_alts and alts:
            parts.append("Bản nghe khác của cùng câu: " + " | ".join(f'"{a}"' for a in alts))
        if self.normalize_hint:
            block = normalize.hint(transcript if self.show_alts else {k: v for k, v in transcript.items() if k != "alts"},
                                   sample["context"]["current_time"])
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    def _obs(self, obs) -> str:
        return json.dumps(obs, ensure_ascii=False)[: self.obs_chars]


CALC = "sys.calc"
CALC_LINE = (f"\n{CALC}(expression: string) [tính nhẩm, không gửi đi] Tính biểu thức: số với + - * / ( ) %; "
             "ngày giờ ISO cộng trừ thời lượng như 30m, 2h, 1d; hiệu hai giờ HH:MM ra số phút")


def _parse_action(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _action_schema(tools: list[str], actions=("call", "stop")) -> dict:
    """JSON Schema cho một hành động: vLLM ép đầu ra theo schema này (giải mã có ràng buộc)."""
    return {"type": "object", "properties": {
        "thought": {"type": "string", "maxLength": 300},
        "action": {"type": "string", "enum": list(actions)},
        "name": {"type": "string", "enum": tools},
        "arguments": {"type": "object"}},
        "required": ["thought", "action"]}


def _json_schema(schema: dict) -> dict:
    return {"response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}}


def _nudge(split: str, tools: list[str]) -> str:
    specs = data.tool_specs(split)
    lines = "\n".join(f"- {t}: {specs.get(t, {}).get('description', '')}" for t in tools)
    return ("Bạn định dừng mà chưa gọi tool nào, trong khi máy có cài app bạn chưa dùng bao giờ:\n" + lines +
            "\nNếu một tool ở trên làm được yêu cầu của người dùng thì gọi nó; nếu thật sự không tool nào phù hợp, "
            'trả lại {"thought": "...", "action": "stop"}.')


@register("agent", "react_json")
class ReactJSON(LLMAgent):
    """ReAct với prompt tự viết: mỗi bước LLM trả {"thought", "action": call|stop, "name", "arguments"}."""

    def __init__(self, json_mode="schema", time_budget=None, min_step_sec=1.5, recheck_stop=None, **kw):
        """json_mode: off | object (chỉ ép JSON hợp lệ) | schema (ép action và tên tool trong tập đã cài).
        time_budget: giây cho cả mẫu, tính cả ASR (luật 30 s). Trước mỗi lượt LLM mà còn dưới min_step_sec thì
            dừng; lượt nào chạy quá phần còn lại cũng dừng. Các call đã gọi vẫn được nộp.
        recheck_stop: {mode: nudge | force, known: train_rest | all | [app, ...]}. Bước đầu đã stop mà máy có app
            không nằm trong call gold của tập `known` (app mô hình SFT chưa thấy được dùng; danh sách tên app khi
            không có dữ liệu train, vd. trong Docker) thì hỏi lại đúng một lần:
            nudge nhắc tên và mô tả tool của các app đó, mô hình vẫn được stop; force (cần json_mode: schema)
            bắt gọi một tool của chúng hoặc sys.ask_missing_info."""
        super().__init__(**kw)
        self.json_mode = {True: "object", False: "off"}.get(json_mode, json_mode)
        self.time_budget = time_budget
        self.min_step_sec = min_step_sec
        self.recheck = recheck_stop or {}

    def _unseen_tools(self, sample) -> list[str]:
        if not self.recheck:
            return []
        known = self.recheck.get("known", "train_rest")
        known = set(known) if isinstance(known, list) else splits.known_apps(known)
        return [t for app, a in sample["context"]["installed_apps"].items() if app not in known for t in a["tools"]]

    def run(self, split, sample, transcript, env):
        t0 = time.perf_counter() - transcript.get("sec", 0.0)  # đồng hồ của mẫu chạy từ lúc bắt đầu ASR
        msgs = [{"role": "system", "content": self.system_prompt(split, sample)},
                {"role": "user", "content": self.user_prompt(sample, transcript)}]
        steps, retries = [], 0
        kw = {}
        if self.json_mode == "object":
            kw = {"response_format": {"type": "json_object"}}
        elif self.json_mode == "schema":
            kw = _json_schema(_action_schema(data.installed_tools(sample) + ([CALC] if self.calc else [])))
        tools = set(data.installed_tools(sample))
        unseen = self._unseen_tools(sample)
        once = None  # kw riêng cho đúng lượt kế tiếp (recheck force)
        for _ in range(self.max_steps + (4 if self.calc else 0)):  # bước sys.calc không tính vào max_steps
            req = once or kw
            once = None
            if self.time_budget:
                left = self.time_budget - (time.perf_counter() - t0)
                if left < self.min_step_sec:
                    steps.append({"action": "budget_stop", "sec_left": round(left, 3)})
                    break
                req = {**req, "timeout": left}
            try:
                msg, meta = self.llm.chat(msgs, **req)
            except LLMTimeout:
                steps.append({"action": "timeout"})
                break
            act = _parse_action(msg.content)
            if isinstance(act, dict) and act.get("action") in tools:  # tên tool đặt nhầm vào "action"
                act = {**act, "name": act.get("name") or act["action"], "action": "call"}
            step = {"raw": msg.content, **meta}
            if not isinstance(act, dict) or act.get("action") not in ("call", "stop"):
                step["action"] = "parse_error"
                steps.append(step)
                retries += 1
                if retries > 1:
                    break
                msgs += [{"role": "assistant", "content": msg.content or ""},
                         {"role": "user", "content": 'Sai định dạng. Trả đúng một JSON: {"thought": "...", "action": "call", "name": "...", "arguments": {...}} hoặc {"thought": "...", "action": "stop"}'}]
                continue
            step.update(action=act["action"], thought=act.get("thought", ""))
            if act["action"] == "stop":
                steps.append(step)
                if unseen and not any(s.get("action") == "call" or s.get("recheck") for s in steps[:-1]):
                    step["recheck"] = self.recheck.get("mode", "nudge")
                    if step["recheck"] == "force" and self.json_mode == "schema":
                        once = _json_schema(_action_schema(unseen + ["sys.ask_missing_info"], actions=("call",)))
                    else:
                        msgs += [{"role": "assistant", "content": msg.content},
                                 {"role": "user", "content": _nudge(split, unseen)}]
                    continue
                break
            name, args = act.get("name", ""), act.get("arguments") or {}
            if self.calc and name == CALC:  # agent tự tính, không qua môi trường nên không vào file nộp
                obs = {"result": calc(args.get("expression", ""))}
            else:
                obs = env.call(name, args)
            step.update(name=name, arguments=args, observation=obs)
            steps.append(step)
            msgs += [{"role": "assistant", "content": msg.content},
                     {"role": "user", "content": f"KẾT QUẢ {name}: {self._obs(obs)}"}]
        return {"steps": steps, "system_chars": len(msgs[0]["content"])}


def _tool_schema(spec: dict) -> dict:
    """Schema của catalog sang dạng tools của OpenAI: bỏ khoá x-*, đưa x-ref và fuzzy vào mô tả."""
    def clean(node):
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if not k.startswith("x-")}
            notes = []
            if node.get("x-ref"):
                notes.append(f"ID lấy từ {node['x-ref']}")
            if node.get("x-match") == "fuzzy":
                notes.append("văn bản tự do")
            if notes:
                out["description"] = (node.get("description", "") + " (" + "; ".join(notes) + ")").strip()
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    params = clean(spec.get("parameters") or {"type": "object", "properties": {}})
    desc = f"[{spec['kind']}] {spec.get('description', '')}"
    if spec.get("parameters", {}).get("x-one-of"):
        desc += f" Bắt buộc một trong: {spec['parameters']['x-one-of']}"
    return {"type": "function", "function": {"name": spec["name"].replace(".", "__"), "description": desc, "parameters": params}}


@register("agent", "native_tools")
class NativeTools(LLMAgent):
    """ReAct qua tool calling gốc (vLLM cần --enable-auto-tool-choice --tool-call-parser hermes với Qwen)."""

    def __init__(self, prompt="prompts/native_tools.md", schema="none", **kw):
        super().__init__(prompt=prompt, schema=schema, **kw)

    def run(self, split, sample, transcript, env):
        specs = data.tool_specs(split)
        tools = [_tool_schema(specs[n]) for n in data.installed_tools(sample) if n in specs]
        msgs = [{"role": "system", "content": self.system_prompt(split, sample)},
                {"role": "user", "content": self.user_prompt(sample, transcript)}]
        steps = []
        for _ in range(self.max_steps):
            msg, meta = self.llm.chat(msgs, tools=tools, tool_choice="auto")
            calls = msg.tool_calls or []
            if not calls:
                steps.append({"action": "stop", "raw": msg.content, **meta})
                break
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for i, c in enumerate(calls):
                name = c.function.name.replace("__", ".")
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = env.call(name, args)
                steps.append({"action": "call", "name": name, "arguments": args, "observation": obs,
                              **(meta if i == 0 else {"sec": 0.0})})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": self._obs(obs)})
        return {"steps": steps, "system_chars": len(msgs[0]["content"])}
