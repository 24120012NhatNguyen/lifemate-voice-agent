"""Audit an LM-98 dataset against its input, without using numbers.calc as oracle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from lifemate import data, splits
from lifemate.agent.agents import CALC, CALC_LINE


def independent_calc(expr):
    shift = re.fullmatch(r"(.+) ([+-]) (\d+)([mhdw])", expr)
    if shift:
        base, sign, amount, unit = shift.groups()
        delta = timedelta(minutes=int(amount) * {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit])
        delta *= 1 if sign == "+" else -1
        if "T" in base:
            return "datetime_shift", (datetime.fromisoformat(base) + delta).isoformat()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", base):
            return "date_shift", (date.fromisoformat(base) + delta).isoformat()
        t = datetime.strptime(base, "%H:%M") + delta
        return "clock_shift", t.strftime("%H:%M")
    if re.fullmatch(r"\d{2}:\d{2} - \d{2}:\d{2}", expr):
        a, b = expr.split(" - ")
        minutes = lambda s: int(s[:2]) * 60 + int(s[3:])
        return "clock_difference", minutes(a) - minutes(b)
    if "T" in expr:
        a, b = expr.split(" - ")
        return "datetime_difference", int((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds() // 60)
    a, op, b = re.fullmatch(r"(-?\d+(?:\.\d+)?)([+*/-])(-?\d+(?:\.\d+)?)", expr).groups()
    a, b = Decimal(a), Decimal(b)
    value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[op]()
    return f"numeric_{op}", float(value)


def check(source: Path, output: Path):
    before = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    after = [json.loads(x) for x in output.read_text().splitlines() if x.strip()]
    assert len(before) == len(after)
    stats, by_kind, expressions, per_row, groups = Counter(), Counter(), Counter(), Counter(), Counter()
    samples, specs = data.by_id("train"), data.tool_specs("train")
    for row_no, (old, new) in enumerate(zip(before, after), 1):
        restored = copy.deepcopy(new)
        assert restored["messages"][0]["content"].count(CALC_LINE) == 1, row_no
        restored["messages"][0]["content"] = restored["messages"][0]["content"].replace(CALC_LINE, "", 1)
        msgs, kept, i, count = restored["messages"], [], 0, 0
        while i < len(msgs):
            msg = msgs[i]
            act = json.loads(msg["content"]) if msg["role"] == "assistant" else {}
            if act.get("name") != CALC:
                kept.append(msg)
                i += 1
                continue
            assert "asr" in old, (row_no, "calc inserted outside gold/fold")
            expr = act["arguments"]["expression"]
            kind, value = independent_calc(expr)
            obs = msgs[i + 1]
            assert obs["role"] == "user" and obs["content"].startswith(f"KẾT QUẢ {CALC}: ")
            result = json.loads(obs["content"].split(": ", 1)[1])["result"]
            assert value == result, (row_no, expr, value, result)
            # A calc chain must immediately precede a write with this exact result.
            j = i + 2
            while json.loads(msgs[j]["content"]).get("name") == CALC:
                j += 2
            target = json.loads(msgs[j]["content"])
            assert specs[target["name"]]["kind"] == "write"
            assert any(not isinstance(v, bool) and v == value for v in target["arguments"].values())
            by_kind[kind] += 1
            expressions[expr] += 1
            count += 1
            i += 2
        restored["messages"] = kept
        assert restored == old, (row_no, old["id"], "non-calc data changed")
        if "asr" in old:
            groups[splits.group(samples[old["id"].removesuffix("_valasr")])] += 1
        elif old.get("teacher") == "paraphrase":
            host = re.search(r"train_\d+", old["id"]).group()
            groups[splits.group(samples[host])] += 1
        stats["rows_with_calc"] += bool(count)
        stats["calc_steps"] += count
        per_row[count] += 1
    return {"rows": len(before), **stats, "independent_result_errors": 0, "roundtrip_errors": 0,
            "types": dict(by_kind), "expressions": dict(sorted(expressions.items())),
            "calc_steps_per_row": dict(sorted(per_row.items())), "gold_fold_paraphrase_groups": dict(groups),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.source, args.output)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "expressions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
