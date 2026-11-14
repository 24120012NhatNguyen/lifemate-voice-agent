"""Đo luật tầng D trên vết đã lưu, không cần GPU: chạy lại mọi call LLM đã đề nghị trong một lượt chạy qua GuardedEnv với
guards của chính lượt đó cộng các cờ thử, trên ToolEnv mới, rồi chấm lại bằng bộ chấm của đội.

    python scripts/replay_rules.py RUN_DIR [RUN_DIR…] [--set '{"rule": true}'] [--ids 20] [--show N]

- Call được phát lại là `guard_log[].proposed` theo thứ tự, trừ các câu hỏi lại do `finish()` của lượt cũ thêm vào
  (GuardedEnv mới tự thêm lại ở `finish()`); call bị hoãn hay bị chặn cũng phát lại, nên luật ask_missing / pick_one của
  lượt cũ cho đúng kết quả cũ.
- Lời lượt cuối và các lượt user trước lấy từ transcripts.jsonl của chính lượt chạy (bản chép ASR, như lúc chạy thật).
- Không có `--set`: phải ra đúng điểm của lượt chạy (kiểm công cụ). Có `--set`: in điểm mới, số mẫu tốt lên / tệ đi và ID.
- Giới hạn: luật đổi kết quả mà LLM nhìn thấy (ghi chú, hỏi lại thay cho ghi) có thể làm LLM đổi các bước sau; phần đó
  chỉ đo được bằng lượt chạy GPU thật (config 197–199).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from lifemate import data
from lifemate.guards import GuardedEnv
from lifemate.runner import gold_hash
from lifemate.scoring import score_sample

SPLIT = "train"
FINISH = "chạy (hỏi lại đã hoãn)"


def run_guards(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return dict(cfg.get("guards") or {})


def replay_row(row: dict, sample: dict, tr: dict, guards: dict) -> tuple[list, str, list]:
    env = data.tool_env(SPLIT, sample["id"])
    g = GuardedEnv(env, SPLIT, sample, tr.get("final", ""), turns=tr.get("turns"), **guards)
    for rec in row.get("guard_log") or []:
        if rec.get("decision") == FINISH:
            continue
        g.call(rec["name"], rec.get("proposed", rec.get("arguments")) or {})
    g.finish()
    return env.get_tool_calls(), env.get_state_hash(), g.log


def score(calls, h, sample) -> float:
    return score_sample(SPLIT, calls, h, sample["label"]["tool_calls"], gold_hash(SPLIT, sample))["score"]


def analyse(run_dir: Path, extra: dict, show: int) -> dict:
    rows = [json.loads(x) for x in (run_dir / "samples.jsonl").open(encoding="utf-8") if x.strip()]
    trs = {}
    for x in (run_dir / "transcripts.jsonl").open(encoding="utf-8"):
        if x.strip():
            t = json.loads(x)
            trs[t["id"]] = t
    guards = {**run_guards(run_dir), **extra}
    by_id = data.by_id(SPLIT)
    old_total = new_total = 0.0
    better, worse, fired = [], [], 0
    for row in rows:
        s = by_id[row["id"]]
        calls, h, log = replay_row(row, s, trs.get(row["id"], {}), guards)
        new = score(calls, h, s)
        old = row["score"]
        old_total += old
        new_total += new
        if any(rec.get("fixes") or rec["decision"].startswith("hoãn") for rec in log) and extra:
            fired += 1
        if new > old + 1e-9:
            better.append((row["id"], round(old, 3), round(new, 3)))
        elif new < old - 1e-9:
            worse.append((row["id"], round(old, 3), round(new, 3)))
        if show and (new != old) and show > 0:
            show -= 1
            print(json.dumps({"id": row["id"], "old": row["tool_calls"], "new": calls}, ensure_ascii=False)[:1500])
    n = len(rows)
    return {"run": run_dir.name, "n": n, "score_run": round(100 * old_total / n, 2), "score_replay": round(100 * new_total / n, 2),
            "delta": round(100 * (new_total - old_total) / n, 2), "better": better, "worse": worse, "rows_with_rule_notes": fired}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", default="{}", help="cờ guards thêm vào guards của lượt chạy, JSON")
    ap.add_argument("--ids", type=int, default=20, help="số ID tốt lên / tệ đi in ra mỗi lượt")
    ap.add_argument("--show", type=int, default=0, help="in call cũ / mới của N mẫu đổi")
    a = ap.parse_args(argv)
    extra = json.loads(a.set)
    for r in a.runs:
        res = analyse(Path(r), extra, a.show)
        print(json.dumps({**res, "better": res["better"][:a.ids], "worse": res["worse"][:a.ids],
                          "n_better": len(res["better"]), "n_worse": len(res["worse"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
