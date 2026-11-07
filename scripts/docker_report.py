"""Tóm tắt một hay nhiều thư mục ra của docker/predict.sh: tham số, thời gian từng pha, độ trễ mỗi mẫu, bộ nhớ GPU,
và so file nộp giữa các lượt (nấc 6: hai lượt phải ra cùng từng byte).

    .venv/bin/python scripts/docker_report.py OUT [--models models/phowhisper-large-ft models/qwen3-4b-sft-full]
    .venv/bin/python scripts/docker_report.py OUT1 OUT2 [--ref runs/<lượt 98>/predictions.jsonl]

Độ trễ mỗi mẫu = sec_agent + sec_asr. sec_asr của lượt nộp là phần chia đều của ASR theo lô; có OUT/asr_latency.json
(scripts/asr_latency.py, mỗi mẫu chạy ASR một mình) thì báo thêm độ trễ khi mỗi mẫu đi trọn một mình.
Bộ nhớ GPU đo cả card (nvidia-smi), trừ mức nền lúc bắt đầu; card dùng chung nên chỉ đúng khi không ai khác chạy.
Với một thư mục, ghi OUT/report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def params(model_dir: Path) -> int:
    """Đếm tham số từ header safetensors, không nạp trọng số."""
    n = 0
    for f in sorted(model_dir.glob("*.safetensors")):
        with f.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in header.items():
            if k != "__metadata__":
                c = 1
                for s in v["shape"]:
                    c *= s
                n += c
    return n


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 2)


def lat(xs: list[float]) -> dict:
    return {"p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99), "max": round(max(xs), 2),
            "over_20s": sum(x > 20 for x in xs), "over_30s": sum(x > 30 for x in xs)}


def summarize(out: Path, models: list[str]) -> dict:
    run = sorted(out.glob("runs/*_submit"))[-1]
    rows = [json.loads(x) for x in (run / "samples.jsonl").open(encoding="utf-8")]
    rep: dict = {"run": str(run), "n": len(rows)}
    if models:
        counts = {m: params(Path(m) if Path(m).is_absolute() else REPO / m) for m in models}
        rep["params"] = {**counts, "total": sum(counts.values())}
    timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    rep["phases_sec"] = {k: round(b - a, 1) for k, (a, b) in timing.items()}
    rep["latency_batch_asr"] = lat([r["sec_agent"] + r["sec_asr"] for r in rows])
    rep["agent_only"] = lat([r["sec_agent"] for r in rows])
    if (out / "asr_latency.json").exists():
        single = json.loads((out / "asr_latency.json").read_text())
        rep["asr_single"] = lat(list(single.values()))
        rep["latency_single_asr"] = lat([r["sec_agent"] + single[r["id"]] for r in rows])
    rep["budget_stops"] = sum(1 for r in rows for s in r["steps"] if s.get("action") in ("budget_stop", "timeout"))
    rep["run_errors"] = sum(1 for r in rows if r.get("error_run"))
    rep["llm_calls_max"] = max(r["llm_calls"] for r in rows)
    if (out / "gpu_mem.csv").exists():
        mem = []
        for line in (out / "gpu_mem.csv").read_text().splitlines():
            t, _, m = line.partition(",")
            if m.strip().isdigit():
                mem.append((float(t), int(m)))
        if mem:
            base = mem[0][1]
            rep["gpu_mib"] = {"baseline": base, "peak_total": max(m for _, m in mem)}
            for k, (a, b) in timing.items():
                seg = [m for t, m in mem if a <= t <= b]
                if seg:
                    rep["gpu_mib"][f"peak_{k}_over_baseline"] = max(seg) - base
    sub = out / "submission.jsonl"
    if sub.exists():
        rep["submission_sha256"] = hashlib.sha256(sub.read_bytes()).hexdigest()
    return rep


def diff_ids(a: Path, b: Path) -> list[str]:
    ra = {r["id"]: r for r in map(json.loads, a.open(encoding="utf-8"))}
    rb = {r["id"]: r for r in map(json.loads, b.open(encoding="utf-8"))}
    return sorted(i for i in ra.keys() | rb.keys() if ra.get(i) != rb.get(i))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--ref", type=Path, help="file nộp để so số mẫu khác (vd. lượt 98 ngoài Docker)")
    a = ap.parse_args(argv)
    bad = [o for o in a.outs if not list(o.glob("runs/*_submit/samples.jsonl"))]
    if bad:
        sys.exit(f"không có lượt agent hoàn tất trong {', '.join(map(str, bad))}: xem predict.log, vllm.log")
    reps = [summarize(o, a.models) for o in a.outs]
    if len(reps) == 1:
        (a.outs[0] / "report.json").write_text(json.dumps(reps[0], ensure_ascii=False, indent=1), encoding="utf-8")
    for o, r in zip(a.outs, reps):
        print(f"== {o}\n{json.dumps(r, ensure_ascii=False, indent=1)}")
    subs = [o / "submission.jsonl" for o in a.outs]
    ok = True
    for s in subs[1:]:
        same = s.read_bytes() == subs[0].read_bytes()
        ok &= same
        print(f"{subs[0]} vs {s}: " + ("giống từng byte" if same else f"KHÁC ở {len(diff_ids(subs[0], s))} mẫu"))
    if a.ref:
        d = diff_ids(subs[0], a.ref)
        print(f"{subs[0]} vs {a.ref}: khác ở {len(d)} mẫu {d[:20]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
