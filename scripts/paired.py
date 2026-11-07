"""So ghép cặp hai lượt chạy trên cùng mẫu: chênh điểm trung bình, khoảng tin cậy 95% bootstrap, số mẫu tốt hơn, tệ hơn.

    .venv/bin/python scripts/paired.py 54f 54g                 # một tập
    .venv/bin/python scripts/paired.py 54f,55f 54g,55g         # cặp val_random và val_app, kèm trung bình hai tập

Mỗi tên là tiền tố config (lấy lượt mới nhất trong runs/ có tên `*_<tiền tố>_*`) hoặc đường dẫn thư mục run.
Cột "nghe sạch": trong các mẫu mà bản chép của lượt A khớp hẳn audio_text (sau chuẩn hoá), lượt B tệ hơn ở bao nhiêu mẫu;
một thay đổi ở tầng ASR không nên làm hỏng mẫu đã nghe đúng.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from lifemate.asr.metrics import norm
from lifemate.config import REPO


def load(ref: str) -> dict[str, dict]:
    path = Path(ref)
    if not path.is_dir():
        runs = sorted(d for d in (REPO / "runs").glob(f"*_{ref}_*") if (d / "metrics.json").exists())
        runs = [d for d in runs if d.name.split("_", 2)[2].startswith(ref + "_") or d.name.endswith("_" + ref)] or runs
        if not runs:
            sys.exit(f"không thấy run cho {ref}")
        path = runs[-1]
    return {r["id"]: r for r in map(json.loads, (path / "samples.jsonl").open(encoding="utf-8"))}


def compare(a: dict[str, dict], b: dict[str, dict], n_boot=10000, seed=0) -> dict:
    ids = sorted(set(a) & set(b))
    d = [100 * (b[i]["score"] - a[i]["score"]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    clean = [i for i in ids if a[i].get("reference") is not None
             and norm(a[i].get("transcript") or "", True) == norm(a[i]["reference"], True)]
    return {"n": len(ids), "a": 100 * sum(a[i]["score"] for i in ids) / len(ids),
            "b": 100 * sum(b[i]["score"] for i in ids) / len(ids), "diff": sum(d) / len(d),
            "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)],
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d),
            "clean": len(clean), "clean_worse": sum(b[i]["score"] < a[i]["score"] for i in clean)}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pairs = list(zip(sys.argv[1].split(","), sys.argv[2].split(",")))
    diffs = []
    for ra, rb in pairs:
        r = compare(load(ra), load(rb))
        diffs.append(r["diff"])
        print(f"{ra} → {rb}: n={r['n']}  {r['a']:.2f} → {r['b']:.2f}  {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
              f"tốt hơn {r['better']}, tệ hơn {r['worse']}; nghe sạch {r['clean']} mẫu, tệ hơn {r['clean_worse']}")
    if len(diffs) == 2:
        print(f"trung bình hai tập: {sum(diffs) / 2:+.2f}")


if __name__ == "__main__":
    main()
