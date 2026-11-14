"""CPU/train-only evidence for LM-71; output contains aggregate metrics and IDs, never raw calls.

Run with LIFEMATE_DATA and PYTHONPATH=src; pass the two saved 211 run directories and --output.
Replays proposed calls through real GuardedEnv + fresh train ToolEnv, with the flag off/on.
Both scorers are recomputed (saved scores are strict), and every observation/hash is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from lifemate import data, splits
from lifemate.guards import GuardedEnv
from lifemate.runner import gold_hash
from lifemate.scoring import score_faq, score_sample
from replay_rules import FINISH, run_guards

TOOL = "calendar.create_event"
SCORERS = {"score": score_sample, "score_faq": score_faq}


def replay_proposals(row, sample, transcript, guards):
    """Like replay_rules.replay_row, also retain the actual observations returned to the agent."""
    env = data.tool_env("train", sample["id"])
    guard = GuardedEnv(env, "train", sample, transcript["final"], turns=transcript.get("turns"), **guards)
    observations = []
    for rec in row["guard_log"]:
        if rec.get("decision") != FINISH:
            observations.append(guard.call(rec["name"], rec.get("proposed", rec.get("arguments")) or {}))
    guard.finish()
    return env.get_tool_calls(), env.get_state_hash(), guard.log, observations


def scores(sample, calls, state_hash):
    return {key: fn("train", calls, state_hash, sample["label"]["tool_calls"], gold_hash("train", sample))["score"]
            for key, fn in SCORERS.items()}


def summary(deltas, before, after):
    n = len(deltas)
    return {"before": 100 * sum(before) / n, "after": 100 * sum(after) / n,
            "delta": 100 * sum(deltas) / n,
            "better": sum(d > 1e-9 for d in deltas), "worse": sum(d < -1e-9 for d in deltas)}


def train_evidence():
    samples = data.samples("train")
    counts = Counter({"end_present": 0, "end_absent": 0, "end_null": 0, "end_empty": 0})
    durations, pair_counts, edge_counts = Counter(), Counter(), Counter()
    gold_scores = {k: {"before": [], "after": []} for k in SCORERS}
    changed = []
    for sample in samples:
        calls = sample["label"]["tool_calls"]
        targets = [i for i, c in enumerate(calls) if c["name"] == TOOL]
        counts["samples_with_create"] += bool(targets)
        for index in targets:
            args = calls[index]["arguments"]
            counts["create_calls"] += 1
            ending = "absent" if "end" not in args else "null" if args["end"] is None else "empty" if args["end"] == "" else "present"
            counts["end_" + ending] += 1
            start = datetime.fromisoformat(args["start"])
            if ending == "present":
                seconds = (datetime.fromisoformat(args["end"]) - start).total_seconds()
                durations[str(seconds)] += 1
                counts["present_end_plus_one_hour"] += seconds == 3600
            # Same gold prefix, then omit end versus explicit +1h; compare every following observation/hash.
            envs = [data.tool_env("train", sample["id"]) for _ in range(4)]
            for c in calls[:index]:
                for env in envs:
                    env.call(c["name"], c["arguments"])
            no_end = {k: v for k, v in args.items() if k != "end"}
            with_end = {**no_end, "end": (start + timedelta(hours=1)).isoformat()}
            obs = [env.call(TOOL, arg) for env, arg in zip(envs, [
                no_end, with_end, {**no_end, "end": None}, {**no_end, "end": ""},
            ])]
            edge_counts["null_equal_observations"] += obs[0] == obs[2]
            edge_counts["null_equal_hashes"] += envs[0].get_state_hash() == envs[2].get_state_hash()
            edge_counts["empty_different_observations"] += obs[0] != obs[3]
            edge_counts["empty_different_hashes"] += envs[0].get_state_hash() != envs[3].get_state_hash()
            envs, obs = envs[:2], obs[:2]
            pair_counts["pairs"] += 1
            pair_counts["successful_pairs"] += all(o.get("ok") is True for o in obs)
            pair_counts["equal_observations"] += obs[0] == obs[1]
            pair_counts["equal_state_hashes"] += envs[0].get_state_hash() == envs[1].get_state_hash()
            suffix_equal = True
            for c in calls[index + 1:]:
                obs = [env.call(c["name"], c["arguments"]) for env in envs]
                suffix_equal &= obs[0] == obs[1] and envs[0].get_state_hash() == envs[1].get_state_hash()
            pair_counts["equal_suffixes"] += suffix_equal
        # Full gold replay on ALL train samples, not just samples containing calendar.create_event.
        results = []
        for enabled in (False, True):
            env = data.tool_env("train", sample["id"])
            g = GuardedEnv(env, "train", sample, sample["audio_text"], calendar_create_end=enabled)
            obs = [g.call(c["name"], c["arguments"]) for c in calls]
            g.finish()
            results.append((env.get_tool_calls(), env.get_state_hash(), obs))
        assert results[0][1:] == results[1][1:], sample["id"]
        if results[0][0] != results[1][0]:
            changed.append(sample["id"])
        for label, result in zip(("before", "after"), results):
            for key, value in scores(sample, *result[:2]).items():
                gold_scores[key][label].append(value)
    assert all(pair_counts[k] == pair_counts["pairs"] for k in (
        "successful_pairs", "equal_observations", "equal_state_hashes", "equal_suffixes")), pair_counts
    return {"n": len(samples), "counts": dict(counts), "duration_seconds": dict(sorted(durations.items())),
            "toolenv_pairs": dict(pair_counts), "edge_cases": dict(edge_counts), "gold_replay_changed_ids": changed,
            "gold_replay": {k: summary([b-a for a, b in zip(v["before"], v["after"])], **v)
                            for k, v in gold_scores.items()}}


def replay_evidence(run_dir):
    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines() if line.strip()]
    transcripts = {r["id"]: r for r in map(json.loads, (run_dir / "transcripts.jsonl").read_text().splitlines())}
    cfg = run_guards(run_dir)
    sample_map = data.by_id("train")
    subset = "val_app" if run_dir.name.endswith("val-app") else "val_random"
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["id"] for r in rows} == {s["id"] for s in splits.select("train", subset)}
    values = {k: {"before": [], "after": []} for k in SCORERS}
    changed, counters = [], Counter()
    for row in rows:
        sample = sample_map[row["id"]]
        assert row["gold_calls"] == sample["label"]["tool_calls"]
        old = replay_proposals(row, sample, transcripts[row["id"]], {**cfg, "calendar_create_end": False})
        new = replay_proposals(row, sample, transcripts[row["id"]], {**cfg, "calendar_create_end": True})
        assert old[0] == row["tool_calls"] and old[1] == row["state_hash"], row["id"]
        counters["baseline_calls_hash_match"] += 1
        assert [r["result"] for r in old[2]] == [r["result"] for r in row["guard_log"]], row["id"]
        counters["baseline_observations_match"] += 1
        assert old[1] == new[1], row["id"]
        counters["equal_state_hashes"] += 1
        assert [r["result"] for r in old[2]] == [r["result"] for r in new[2]], row["id"]
        assert json.dumps(old[3], ensure_ascii=False) == json.dumps(new[3], ensure_ascii=False), row["id"]
        counters["equal_observations"] += 1
        old_scores, new_scores = scores(sample, *old[:2]), scores(sample, *new[:2])
        assert abs(old_scores["score"] - row["score"]) < 1e-8, row["id"]
        for key in SCORERS:
            values[key]["before"].append(old_scores[key])
            values[key]["after"].append(new_scores[key])
        if old[0] != new[0]:
            counters["changed_calls"] += sum(a != b for a, b in zip(old[0], new[0]))
            for a, b in zip(old[0], new[0]):
                if a != b:
                    args = a["arguments"]
                    ending = "absent" if "end" not in args else "null" if args["end"] is None else "empty"
                    counters["changed_end_" + ending] += 1
            changed.append({"id": row["id"], "before": old_scores, "after": new_scores})
    return {"run": str(run_dir), "n": len(rows), "checks": dict(counters), "changed_samples": len(changed),
            "changed": changed, "metrics": {k: summary([b-a for a, b in zip(v["before"], v["after"])], **v)
                                            for k, v in values.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"train": train_evidence(), "replays": [replay_evidence(p) for p in args.runs]}
    paths = [data.data_root() / "train/train.jsonl", data.data_root() / "train/train_tools_catalog.json",
             Path(__file__), *[Path("src/lifemate") / n for n in ("guards.py", "scoring.py", "data.py")]]
    paths += [p / n for p in args.runs for n in ("samples.jsonl", "transcripts.jsonl", "config.yaml")]
    paths += sorted((data.data_root() / "train/zaic_tools").rglob("*.py"))
    result["sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
