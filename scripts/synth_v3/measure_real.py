"""Đo phân bố hành động và đặc trưng schema của gold train_rest (chỉ split train)."""

import json
import collections
from lifemate import data, splits


def action_class(s, specs):
    calls = s["label"]["tool_calls"]
    names = [c["name"] for c in calls]
    if not calls:
        return "stop_no_call"
    ask = "sys.ask_missing_info" in names
    kinds = [specs.get(n, {}).get("kind", "write") for n in names if n != "sys.ask_missing_info"]
    if ask and not kinds:
        return "ask_only"
    if ask:
        return "act_then_ask" if "write" in kinds else "read_then_ask"
    if all(k == "read" for k in kinds):
        return "read_only"
    if kinds[0] == "read":
        return "read_then_write"
    return "write_first" if len(kinds) == 1 else "write_multi"


def main():
    specs = data.tool_specs("train")
    pool = splits.select("train", "train_rest")
    C = collections.Counter()
    H = collections.Counter()
    N = collections.Counter()
    A = collections.Counter()
    nw = collections.Counter()
    search_q = collections.Counter()
    for s in pool:
        C[action_class(s, specs)] += 1
        H[min(sum(1 for t in s.get("history", []) if t.get("role") == "user"), 3)] += 1
        N[min(len(s["label"]["tool_calls"]), 6)] += 1
        A[len(s["context"]["installed_apps"]) // 3 * 3] += 1
        kinds = [specs.get(c["name"], {}).get("kind") for c in s["label"]["tool_calls"]]
        nw[min(sum(k == "write" for k in kinds), 3)] += 1
        for c in s["label"]["tool_calls"]:
            if c["name"].endswith("search"):
                search_q["with_query" if c["arguments"].get("query") else "empty"] += 1
    n = len(pool)
    pct = lambda c: {k: f"{v} ({100 * v / n:.1f}%)" for k, v in sorted(c.items(), key=lambda x: str(x[0]))}
    out = {
        "n": n,
        "action": pct(C),
        "history_user_turns": pct(H),
        "n_calls": pct(N),
        "n_writes": pct(nw),
        "installed_apps_bucket": pct(A),
        "search_args": dict(search_q),
    }
    # schema stats on train catalog
    T = list(specs.values())
    P = [(t, p, sc) for t in T for p, sc in t.get("parameters", {}).get("properties", {}).items()]
    ids = [x for x in P if x[1].endswith("_id")]
    out["schema"] = {
        "tools": len(T),
        "read_share": round(sum(t["kind"] == "read" for t in T) / len(T), 3),
        "params": len(P),
        "optional_share": round(sum(p not in t["parameters"].get("required", []) for t, p, _ in P) / len(P), 3),
        "enum_share": round(sum("enum" in sc for *_, sc in P) / len(P), 3),
        "fuzzy_share": round(sum(sc.get("x-match") == "fuzzy" for *_, sc in P) / len(P), 3),
        "xref_share": round(sum("x-ref" in sc for *_, sc in P) / len(P), 3),
        "id_params": len(ids),
        "id_params_with_xref": sum("x-ref" in sc for *_, sc in ids),
        "id_params_cross_app": sum(
            sc.get("x-ref", "").split(".")[0] != t["name"].split(".")[0] for t, _, sc in ids if "x-ref" in sc
        ),
        "ref_like_without_id_suffix": sorted(
            {
                f"{t['name']}.{p}"
                for t, p, sc in P
                if not p.endswith("_id")
                and (
                    "lấy từ" in sc.get("description", "")
                    or "phải có trong" in t.get("description", "")
                    or "phải nằm trong" in sc.get("description", "")
                )
            }
        ),
        "no_param_tools": sum(not t.get("parameters", {}).get("properties") for t in T),
        "x_one_of_tools": sum(bool(t.get("parameters", {}).get("x-one-of")) for t in T),
        "iso_desc": sum("ISO-8601" in sc.get("description", "") for *_, sc in P),
        "date_desc": sum("YYYY-MM-DD" in sc.get("description", "") for *_, sc in P),
        "int_desc": sum("Số nguyên" in sc.get("description", "") for *_, sc in P),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
