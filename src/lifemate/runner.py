"""Chạy một thí nghiệm: ASR → agent → middleware → ToolEnv → chấm điểm, ghi mọi thứ vào runs/<thời gian>_<tên>/.

    config.yaml        config đã gộp
    predictions.jsonl  đúng định dạng nộp (lần lặp đầu)
    transcripts.jsonl  transcript sau hậu xử lý
    samples.jsonl      mỗi mẫu một dòng: điểm, nhãn lỗi, call dự đoán và gold, vết agent, thời gian
    metrics.json       số tổng: điểm, hash, ma trận kiểu phản ứng, lỗi, thời gian, ổn định, ASR
"""
from __future__ import annotations

import json
import statistics as st
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from . import data, splits
from .asr.base import run_asr
from .asr.metrics import asr_metrics, per_sample
from .config import REPO
from .guards import GuardedEnv
from .registry import build
from .scoring import score_sample

_GOLD_HASH: dict[tuple[str, str], str] = {}


def gold_hash(split, sample) -> str:
    key = (split, sample["id"])
    if key not in _GOLD_HASH:
        env = data.tool_env(split, sample["id"])
        for c in sample["label"]["tool_calls"]:
            env.call(c["name"], c["arguments"])
        _GOLD_HASH[key] = env.get_state_hash()
    return _GOLD_HASH[key]


def run_one(cfg, agent, split, sample, transcript) -> dict:
    env = data.tool_env(split, sample["id"])
    genv = GuardedEnv(env, split, sample, transcript["final"], turns=transcript.get("turns"), **(cfg.get("guards") or {}))
    t0 = time.perf_counter()
    try:
        trace = agent.run(split, sample, transcript, genv)
        err = None
    except Exception as e:  # một mẫu lỗi không được làm hỏng cả lượt chạy; vẫn nộp những call đã gọi
        trace, err = {"steps": []}, f"{type(e).__name__}: {e}"
    finally:
        genv.finish()
    sec = time.perf_counter() - t0
    calls, h = env.get_tool_calls(), env.get_state_hash()
    ran = [g for g in genv.log if g["decision"].startswith("chạy")]
    rejected = sum(1 for g in ran if isinstance(g["result"], dict) and g["result"].get("ok") is False)
    row = {
        "id": sample["id"],
        "tool_calls": calls,
        "state_hash": h,
        "sec_agent": round(sec, 3),
        "sec_asr": round(transcript.get("sec", 0.0), 3),
        "llm_calls": sum(1 for s in trace.get("steps", []) if "prompt_tokens" in s),
        "prompt_tokens": sum(s.get("prompt_tokens", 0) for s in trace.get("steps", [])),
        "steps": trace.get("steps", []),
        "guard_log": genv.log,
        "error_run": err,
        "rejected_calls": rejected,
    }
    if "label" in sample:
        row.update(score_sample(split, calls, h, sample["label"]["tool_calls"], gold_hash(split, sample)))
        if rejected and row["error"] not in ("đúng",) and row["hash_ok"]:
            row["error"] = "call bị từ chối"
        row["gold_calls"] = sample["label"]["tool_calls"]
    return row


def run(cfg: dict, log=print, out_root: Path | None = None) -> Path:
    split = cfg.get("split", "train")
    pool = splits.select(split, cfg.get("subset", "all"), cfg.get("limit"))
    log(f"[{cfg['name']}] {split}/{cfg.get('subset', 'all')}: {len(pool)} mẫu")

    t_asr = time.perf_counter()
    trs = run_asr(split, pool, cfg, log)
    t_asr = time.perf_counter() - t_asr

    run_dir = (out_root or REPO / "runs") / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (run_dir / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for s in pool:
            f.write(json.dumps(trs[s["id"]], ensure_ascii=False) + "\n")

    metrics: dict = {"name": cfg["name"], "split": split, "subset": cfg.get("subset", "all"), "n": len(pool),
                     "asr_total_sec": round(t_asr, 1)}
    if split == "train":
        metrics["asr"] = asr_metrics(pool, trs)

    agent_cfg = cfg.get("agent") or {"name": "none"}
    if agent_cfg.get("name") == "none":
        rows = [{"id": s["id"], "transcript": trs[s["id"]]["final"], "reference": s.get("audio_text"),
                 **per_sample(s, trs[s["id"]])} for s in pool]
        _write(run_dir, rows, metrics)
        log(f"[{cfg['name']}] xong ASR: {metrics.get('asr')} → {run_dir}")
        return run_dir

    agent = build("agent", agent_cfg)
    repeats = int(cfg.get("repeats", 1))
    workers = int(cfg.get("workers", 1))
    all_reps: list[list[dict]] = []
    t_run = time.perf_counter()
    for rep in range(repeats):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, cfg, agent, split, s, trs[s["id"]]) for s in pool]
            rows = []
            for i, f in enumerate(futs):
                rows.append(f.result())
                done = [r["score"] for r in rows if "score" in r]
                if (i + 1) % 10 == 0 or i + 1 == len(futs):  # web-app (tab Đang chạy) đọc file này
                    (run_dir / "progress.json").write_text(json.dumps(
                        {"repeat": rep + 1, "repeats": repeats, "done": i + 1, "total": len(futs),
                         "score": round(100 * st.mean(done), 2) if done else None}), encoding="utf-8")
                if (i + 1) % 50 == 0 or i + 1 == len(futs):
                    log(f"  lần {rep + 1}: {i + 1}/{len(futs)}" + (f", điểm tạm {100 * st.mean(done):.2f}" if done else ""))
        all_reps.append(rows)
    metrics["agent_total_sec"] = round(time.perf_counter() - t_run, 1)
    rows = all_reps[0]
    for r in rows:
        r["transcript"] = trs[r["id"]]["final"]
        r["reference"] = data.by_id(split)[r["id"]].get("audio_text")

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]}, ensure_ascii=False) + "\n")

    secs = sorted(r["sec_agent"] + r["sec_asr"] for r in rows)
    metrics.update({
        "workers": workers,
        "repeats": repeats,
        "sec_p50": round(st.median(secs), 2),
        "sec_p95": round(secs[int(0.95 * (len(secs) - 1))], 2),
        "sec_max": round(secs[-1], 2),
        "llm_calls_mean": round(st.mean(r["llm_calls"] for r in rows), 2),
        "prompt_tokens_mean": round(st.mean(r["prompt_tokens"] for r in rows)),
        "calls_mean": round(st.mean(len(r["tool_calls"]) for r in rows), 2),
        "run_errors": sum(1 for r in rows if r["error_run"]),
        "rejected_calls": sum(r["rejected_calls"] for r in rows),
        "guard_blocks": sum(1 for r in rows for g in r["guard_log"] if g["decision"].startswith("chặn")),
        "guard_warnings": sum(1 for r in rows for g in r["guard_log"] if g.get("warnings")),
    })
    if repeats > 1:
        same = sum(1 for i in range(len(rows))
                   if len({json.dumps([rep[i]["tool_calls"], rep[i]["state_hash"]], sort_keys=True) for rep in all_reps}) == 1)
        metrics["stability"] = round(100 * same / len(rows), 2)
        if "score" in rows[0]:
            metrics["score_per_repeat"] = [round(100 * st.mean(r["score"] for r in rep), 2) for rep in all_reps]
    if "score" in rows[0]:
        metrics.update({
            "score": round(100 * st.mean(r["score"] for r in rows), 2),
            "hash_acc": round(100 * st.mean(r["hash_ok"] for r in rows), 2),
            "f1_when_hash_ok": round(100 * st.mean([r["f1"] for r in rows if r["hash_ok"]] or [0]), 2),
            "errors": dict(Counter(r["error"] for r in rows).most_common()),
            "response_confusion": dict(Counter(f"{r['gold_type']} → {r['pred_type']}" for r in rows).most_common()),
        })
    _write(run_dir, rows, metrics)
    log(f"[{cfg['name']}] điểm {metrics.get('score', '-')}, hash {metrics.get('hash_acc', '-')}, p95 {metrics['sec_p95']}s → {run_dir}")
    return run_dir


def _write(run_dir: Path, rows, metrics):
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
