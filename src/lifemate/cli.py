"""Dòng lệnh.

    lifemate run configs/experiments/10_qwen3-4b_gold_react.yaml [--set agent.max_steps=6 --set limit=50]
    lifemate list                      các phương án đã đăng ký theo tầng
    lifemate validate runs/<run>       kiểm schema file nộp bằng zaic_tools của split
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config, data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lifemate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="chạy một hoặc nhiều config")
    r.add_argument("configs", nargs="+")
    r.add_argument("--set", action="append", default=[], help="ghi đè, vd. --set limit=50")
    r.add_argument("--runs-dir", type=Path, default=None, help="thư mục ghi lượt chạy thay cho runs/ (Docker: /result/runs)")
    sub.add_parser("list", help="liệt kê phương án theo tầng")
    v = sub.add_parser("validate", help="kiểm schema predictions.jsonl của một run")
    v.add_argument("run_dir")
    rs = sub.add_parser("rescore", help="tính lại độ đo ASR của các run từ transcripts.jsonl (sau khi sửa metrics)")
    rs.add_argument("run_dirs", nargs="+")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .runner import run
        for path in a.configs:
            run(config.load(path, a.set), out_root=a.runs_dir)
    elif a.cmd == "list":
        from .registry import available
        for stage, entries in available().items():
            print(f"[{stage}]")
            for e in entries:
                print(f"  {e.name:20s} {e.status:11s} {e.doc}")
    elif a.cmd == "rescore":
        from .asr.metrics import asr_metrics, per_sample
        for d in map(Path, a.run_dirs):
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            if m["split"] != "train":
                continue
            trs = {r["id"]: r for r in map(json.loads, (d / "transcripts.jsonl").open(encoding="utf-8"))}
            pool = [data.by_id("train")[i] for i in trs]
            m["asr"] = asr_metrics(pool, trs)
            rows = [json.loads(x) for x in (d / "samples.jsonl").open(encoding="utf-8")]
            for r in rows:
                r.update(per_sample(data.by_id("train")[r["id"]], trs[r["id"]]))
            (d / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
            (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
            print(d.name, m["asr"])
    elif a.cmd == "validate":
        run_dir = Path(a.run_dir)
        split = config.load(run_dir / "config.yaml").get("split", "train")
        root = data.data_root()
        cmd = [sys.executable, "-m", "zaic_tools", "--split", split, "--root", str(root),
               "--validate", str((run_dir / "predictions.jsonl").resolve())]
        sys.exit(subprocess.call(cmd, cwd=root, env={"PYTHONPATH": str(root / split)}))


if __name__ == "__main__":
    main()
