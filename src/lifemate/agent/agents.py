"""Các phương án agent. Mỗi agent nhận mẫu, transcript và môi trường có bảo vệ, rồi trả về vết từng bước.

    none          bỏ qua tầng agent (chỉ đo ASR)
    empty         không gọi gì: điểm sàn
    oracle        phát lại call gold của train: điểm trần, dùng để đo tác động của middleware
    react_json    ReAct, mỗi bước LLM trả một hành động JSON theo prompt tự viết
    native_tools  ReAct qua tool calling gốc của mô hình (API tools của OpenAI)
"""
from __future__ import annotations

import json
import math
import re
import time

from .. import context, data, normalize, splits
from ..config import REPO
from ..numbers import calc
from ..registry import register
from .llm import LLM, LLMTimeout
from .review import InferenceReview, action_key, valid_action


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
                 show_alts=True, obs_chars=2500, calc=False, normalize_hint=False, history_alts=False, **_):
        self.llm = LLM(**(llm or {}))
        self.template = _read(prompt)
        self.rules = _read(rules) if rules else ""
        self.opts = dict(timetable=timetable, schema=schema, history=history, ledger=ledger)
        self.max_steps = max_steps
        self.show_alts = show_alts
        self.history_alts = history_alts  # N-best của các lượt user trong lịch sử (mặc định chỉ lượt cuối)
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
        hist = context.history(sample, transcript.get("turns", {}), self.opts["history"],
                               transcript.get("alts") if getattr(self, "history_alts", False) else None)
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


ASK = "sys.ask_missing_info"


def _option_mass(logprobs: list, text: str, key: str, options: dict[str, list[str]], start: int = 0):
    """Khối xác suất của từng nhóm lựa chọn ở token đầu tiên của giá trị chuỗi sau khoá `key` (vd. "action").

    logprobs: [(token, logprob, [(token khác, logprob), ...]), ...] của câu đã sinh; text là câu đó.
    options: tên nhóm → các giá trị đầy đủ thuộc nhóm. Một token khác chỉ được tính khi nó bắt đầu bằng đúng phần
    đứng trước giá trị trong token đã chọn (vd. ' "') và phần còn lại khớp tiền tố của đúng một nhóm.
    Trả về ({nhóm: khối xác suất}, vị trí ký tự đầu giá trị) hoặc (None, start) khi không tìm thấy,
    token không căn được với text, hoặc có token còn chung cho nhiều nhóm (không đoán phần chia khối đó)."""
    m = re.compile(r'"' + re.escape(key) + r'"\s*:\s*"').search(text, start)
    if not m:
        return None, start
    vs, pos, i = m.end(), 0, None
    for k, (tok, _, _) in enumerate(logprobs):
        if pos <= vs < pos + len(tok):
            i = k
            break
        pos += len(tok)
    if i is None or not text.startswith("".join(t[0] for t in logprobs[:i])):
        return None, start  # token trước vị trí quyết định không căn được với text: không đoán
    pre = text[pos:vs]
    mass = dict.fromkeys(options, 0.0)
    seen = set()
    # API trả logprob của token đã chọn riêng; nó có thể nằm ngoài top-k trước mặt nạ schema.
    candidates = [(logprobs[i][0], logprobs[i][1]), *(logprobs[i][2] or [])]
    for tok, lp in candidates:
        if tok in seen or not tok.startswith(pre):
            continue
        seen.add(tok)
        rest = tok[len(pre):]
        if not rest:
            continue
        # Một token đi hết giá trị phải đóng chuỗi JSON, không phải 'calling' cho lựa chọn 'call'.
        hits = [g for g, vals in options.items() if any(v.startswith(rest) or rest.startswith(v + '"') for v in vals)]
        if len(hits) > 1 and math.exp(lp) > 0:
            return None, start
        if len(hits) == 1:
            mass[hits[0]] += math.exp(lp)
    return mass, vs


def decision_probs(logprobs: list, text: str, tools: list[str]) -> dict | None:
    """Xác suất quyết định của một bước, chuẩn hoá trên các lựa chọn hợp lệ.

    p_stop, p_call: P(action = stop | call). p_ask_call: P(tên tool = sys.ask_missing_info | call), chỉ có khi bước
    đó chọn call (tên tool chỉ được sinh sau "call"); khi đó p_ask = p_call · p_ask_call và p_act = p_call − p_ask."""
    if not logprobs:
        return None
    act, vs = _option_mass(logprobs, text, "action", {"call": ["call"], "stop": ["stop"]})
    if not act or not sum(act.values()):
        return None
    tot = act["call"] + act["stop"]
    out = {"p_stop": act["stop"] / tot, "p_call": act["call"] / tot}
    name, _ = _option_mass(logprobs, text, "name", {"ask": [ASK], "act": [t for t in tools if t != ASK]}, vs)
    if name and sum(name.values()):
        out["p_ask_call"] = name["ask"] / (name["ask"] + name["act"])
        out["p_ask"] = out["p_call"] * out["p_ask_call"]
        out["p_act"] = out["p_call"] - out["p_ask"]
    return {k: round(v, 6) for k, v in out.items()}


def _nudge(split: str, tools: list[str]) -> str:
    specs = data.tool_specs(split)
    lines = "\n".join(f"- {t}: {specs.get(t, {}).get('description', '')}" for t in tools)
    return ("Bạn định dừng mà chưa gọi tool nào, trong khi máy có cài app bạn chưa dùng bao giờ:\n" + lines +
            "\nNếu một tool ở trên làm được yêu cầu của người dùng thì gọi nó; nếu thật sự không tool nào phù hợp, "
            'trả lại {"thought": "...", "action": "stop"}.')


@register("agent", "react_json")
class ReactJSON(LLMAgent):
    """ReAct với prompt tự viết: mỗi bước LLM trả {"thought", "action": call|stop, "name", "arguments"}."""

    def __init__(self, json_mode="schema", time_budget=None, min_step_sec=1.5, recheck_stop=None, decision=None,
                 act_after_retry=False, inference_review=None, **kw):
        """json_mode: off | object (chỉ ép JSON hợp lệ) | schema (ép action và tên tool trong tập đã cài).
        time_budget: giây cho cả mẫu, tính cả ASR (luật 30 s). Trước mỗi lượt LLM mà còn dưới min_step_sec thì
            dừng; lượt nào chạy quá phần còn lại cũng dừng. Các call đã gọi vẫn được nộp.
        inference_review: {enabled: false, mode: source | verify, max_reviews: 2, known: train_rest}
            LM-97: rà bản nháp trước thực thi, tối đa hai lượt thêm; xem agent/review.py.
            Chỉ mẫu có app đã cài ngoài known; lỗi review giữ bản nháp; cờ tắt giữ trace cũ.
        recheck_stop: {mode: nudge | force, known: train_rest | all | [app, ...]}. Bước đầu đã stop mà máy có app
            không nằm trong call gold của tập `known` (app mô hình SFT chưa thấy được dùng; danh sách tên app khi
            không có dữ liệu train, vd. trong Docker) thì hỏi lại đúng một lần:
            nudge nhắc tên và mô tả tool của các app đó, mô hình vẫn được stop; force (cần json_mode: schema)
            bắt gọi một tool của chúng hoặc sys.ask_missing_info.
        decision: chỉnh quyết định làm / hỏi lại / đứng yên theo xác suất của chính model (LM-25), mặc định tắt:
            record: true       ghi p_stop, p_call, p_ask_call, p_ask, p_act vào từng bước (logprobs của vLLM; không đổi
                               hành vi ở temperature 0)
            top_logprobs: 20   số token khác đọc ở mỗi vị trí
            force_act_first    bước quyết định đầu tiên bị cấm stop và sys.ask_missing_info (chỉ để đo phản thực)
            mode: threshold    khi model chọn hỏi lại mà p_ask < ask_min, hoặc stop mà p_stop < stop_min, sinh lại bước
                               đó với cả hỏi lại và stop bị cấm (model tự chọn đọc hay ghi), trước khi gọi tool;
                               ghi decision_override vào bước
            scope: first       chỉ bước quyết định đầu; pre_write: mọi bước trước lệnh ghi đầu tiên và trước khi đã
                               hỏi lại
            stop_scope: pre_write  mặc định giữ nguyên `scope` ở trên; any: cho phép ghi đè stop cả sau lệnh ghi,
                               nhưng không sau khi đã hỏi lại hoặc stop. Không mở rộng phạm vi ask.
            unseen_only: false chỉ ghi đè stop khi có app đã cài (có tool, khác sys) không nằm trong `known`;
                               stop không có tool đích, nên đây là điều kiện trên app đã cài, không đoán ý định
            ask_min, stop_min  ngưỡng; 0 là không bao giờ ghi đè
            later_ask_min, later_stop_min  ngưỡng riêng cho các bước sau bước quyết định đầu (kể cả sau lệnh ghi);
                               mặc định bằng ask_min / stop_min
            max_overrides      số lần ghi đè tối đa mỗi mẫu (mặc định không giới hạn ngoài max_steps)
            ask_apps: all      unseen: chỉ ghi đè câu hỏi lại về tool của app KHÔNG nằm trong `known` (app model SFT
                               chưa thấy được dùng; ở app đã học model hỏi lại đúng phần lớn, ở app lạ thì quá dè dặt)
            known: train_rest  train_rest | all | [app, ...] như recheck_stop (danh sách tên app khi không có dữ liệu
                               train, vd. trong Docker)"""
        super().__init__(**kw)
        self.json_mode = {True: "object", False: "off"}.get(json_mode, json_mode)
        self.time_budget = time_budget
        self.min_step_sec = min_step_sec
        self.recheck = recheck_stop or {}
        self.decision = decision or {}
        if self.decision.get("stop_scope", "pre_write") not in ("pre_write", "any"):
            raise ValueError("decision.stop_scope must be pre_write or any")
        # act_after_retry: ngay sau khi guards.empty_retry trả về danh sách không lọc có mục (tên bị nghe sai nên lọc ra rỗng),
        # bước kế tiếp bị cấm stop và sys.ask_missing_info đúng một lần (cần json_mode: schema): model phải chọn mục trong danh
        # sách hoặc đọc tiếp. Vết 240: 11/22 lần như vậy model hỏi lại hoặc dừng, cả 11 mẫu có gold là làm và đang 0 điểm.
        self.act_after_retry = act_after_retry
        self.inference_review = InferenceReview(inference_review)
        if self.inference_review.enabled and self.json_mode != "schema":
            raise ValueError("inference_review requires json_mode: schema")

    def _in_scope(self, split, steps) -> bool:
        """Bước sắp tới có thuộc phạm vi chỉnh ngưỡng không."""
        decided = [s for s in steps if s.get("action") in ("call", "stop")]
        if self.decision.get("scope", "first") == "first":
            return not decided
        specs = data.tool_specs(split)
        for s in decided:
            if s.get("action") == "stop" or s.get("name") == ASK:
                return False
            if specs.get(s.get("name"), {}).get("kind") == "write":
                return False
        return True

    def _ask_app_ok(self, act) -> bool:
        """Câu hỏi lại này có thuộc phạm vi ask_apps không (xem `decision`)."""
        if self.decision.get("ask_apps", "all") != "unseen":
            return True
        known = self.decision.get("known", "train_rest")
        known = set(known) if isinstance(known, list) else splits.known_apps(known)
        tool = (act.get("arguments") or {}).get("tool_name")
        return isinstance(tool, str) and "." in tool and tool.split(".")[0] not in known

    def _stop_in_scope(self, split, steps, sample) -> bool:
        """Giữ phạm vi cũ trừ khi bật any; luôn tôn trọng stop / ask đã chốt."""
        if self.decision.get("stop_scope", "pre_write") == "any":
            in_scope = not any(s.get("action") == "stop" or s.get("name") == ASK for s in steps)
        else:
            in_scope = self._in_scope(split, steps)
        if not in_scope or not self.decision.get("unseen_only", False):
            return in_scope
        known = self.decision.get("known", "train_rest")
        known = set(known) if isinstance(known, list) else splits.known_apps(known)
        return any(app != "sys" and app not in known and a.get("tools")
                   for app, a in sample["context"]["installed_apps"].items())

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
        installed = data.installed_tools(sample) + ([CALC] if self.calc else [])
        tools = set(data.installed_tools(sample))
        unseen = self._unseen_tools(sample)
        dec = self.decision
        threshold = dec.get("mode") == "threshold" and self.json_mode == "schema"
        lp_kw = {"logprobs": True, "top_logprobs": dec.get("top_logprobs", 20)} if dec.get("record") or threshold else {}
        no_ask = _json_schema(_action_schema([t for t in installed if t != ASK], actions=("call",)))
        review = self.inference_review
        review_on = review.sample_enabled(sample)
        reviews_used = 0
        once = None  # kw riêng cho đúng lượt kế tiếp (recheck force)
        for _ in range(self.max_steps + (4 if self.calc else 0)):  # bước sys.calc không tính vào max_steps
            req = once or kw
            once = None
            if (dec.get("force_act_first") and self.json_mode == "schema"
                    and not any(s.get("action") in ("call", "stop") for s in steps)):
                req = no_ask
            if self.time_budget:
                left = self.time_budget - (time.perf_counter() - t0)
                if left < self.min_step_sec:
                    steps.append({"action": "budget_stop", "sec_left": round(left, 3)})
                    break
                req = {**req, "timeout": left}
            in_scope = threshold and self._in_scope(split, steps)
            try:
                msg, meta = self.llm.chat(msgs, **lp_kw, **req)
            except LLMTimeout:
                steps.append({"action": "timeout"})
                break
            lps = meta.pop("logprobs", None)
            act = _parse_action(msg.content)
            if isinstance(act, dict) and act.get("action") in tools:  # tên tool đặt nhầm vào "action"
                act = {**act, "name": act.get("name") or act["action"], "action": "call"}
            step = {"raw": msg.content, **meta}
            ok = isinstance(act, dict) and act.get("action") in ("call", "stop")
            probs = decision_probs(lps, msg.content, installed) if ok and lps else None
            if probs:
                step.update(probs)
            chosen = None
            if ok and threshold and probs and sum(1 for s in steps if s.get("decision_override")) < dec.get(
                    "max_overrides", self.max_steps):
                later = any(s.get("action") in ("call", "stop") for s in steps)
                ask_min = dec.get("later_ask_min", dec.get("ask_min", 0)) if later else dec.get("ask_min", 0)
                stop_min = dec.get("later_stop_min", dec.get("stop_min", 0)) if later else dec.get("stop_min", 0)
                if (act["action"] == "stop" and probs["p_stop"] < stop_min
                        and self._stop_in_scope(split, steps, sample)):
                    chosen = "stop"
                elif (in_scope and act.get("name") == ASK and probs.get("p_ask", 1) < ask_min
                      and self._ask_app_ok(act)):
                    chosen = "ask"
            if chosen:
                left = self.time_budget - (time.perf_counter() - t0) if self.time_budget else None
                if left is None or left >= self.min_step_sec:
                    from openai import APIError

                    try:
                        msg2, meta2 = self.llm.chat(msgs, **no_ask, **({"timeout": left} if left else {}))
                        act2 = _parse_action(msg2.content)
                    except (LLMTimeout, APIError):
                        act2 = None
                    if isinstance(act2, dict) and act2.get("action") == "call" and act2.get("name") in tools:
                        step["decision_override"] = {"from": chosen, "p": probs["p_stop" if chosen == "stop" else "p_ask"],
                                                     "raw": msg.content}
                        step["raw"] = msg2.content
                        for k in ("sec", "prompt_tokens", "completion_tokens"):
                            step[k] = step.get(k, 0) + meta2.get(k, 0)
                        msg, act = msg2, act2
            if not ok:
                step["action"] = "parse_error"
                steps.append(step)
                retries += 1
                if retries > 1:
                    break
                msgs += [{"role": "assistant", "content": msg.content or ""},
                         {"role": "user", "content": 'Sai định dạng. Trả đúng một JSON: {"thought": "...", "action": "call", "name": "...", "arguments": {...}} hoặc {"thought": "...", "action": "stop"}'}]
                continue
            if review_on and review.eligible(split, act, steps, reviews_used):
                # A threshold override / retry may have narrowed the action mask. Keep it.
                review_req = no_ask if step.get("decision_override") else req
                left = self.time_budget - (time.perf_counter() - t0) if self.time_budget else None
                event = {"mode": review.mode, "proposal_raw": msg.content, "attempted": False,
                         "changed": False, "status": "budget_skip"}
                step["inference_review"] = event
                if left is None or left >= self.min_step_sec:
                    from openai import APIError

                    reviews_used += 1  # failures count too; never recursively review a replacement
                    event["attempted"] = True
                    started = time.perf_counter()
                    try:
                        review_req = {**review_req, **({"timeout": left} if left is not None else {})}
                        msg2, meta2 = self.llm.chat(review.messages(msgs, msg.content), **review_req)
                        event.update(raw=msg2.content,
                                     **{k: meta2.get(k, 0) for k in ("sec", "prompt_tokens", "completion_tokens")})
                        for k in ("sec", "prompt_tokens", "completion_tokens"):
                            step[k] = step.get(k, 0) + meta2.get(k, 0)
                        act2 = _parse_action(msg2.content)
                        if valid_action(act2, review_req):
                            event.update(status="accepted", changed=action_key(act2) != action_key(act))
                            # Keep original bytes if the proposed action is unchanged.
                            if event["changed"]:
                                step["raw"] = msg2.content
                                msg, act = msg2, act2
                        else:
                            event["status"] = "invalid"
                    except (LLMTimeout, APIError) as exc:
                        event.update(status="timeout" if isinstance(exc, LLMTimeout) else "api_error",
                                     sec=time.perf_counter() - started)
                        step["sec"] = step.get("sec", 0) + event["sec"]
                    # Review notes are local to the extra request. Only the selected action is
                    # appended to the main conversation below; no speculative call reaches env.
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
            if getattr(env, "retry_hit", False):
                env.retry_hit = False
                if self.act_after_retry and self.json_mode == "schema":
                    once = no_ask
                    step["act_after_retry"] = True
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
