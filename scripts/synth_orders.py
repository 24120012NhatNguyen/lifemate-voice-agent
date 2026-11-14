"""Tìm lại thứ tự duyệt `set` của một lượt sinh mẫu tổng hợp cũ, để bộ sinh phát lại đúng file đó từng byte (LM-48).

Trước LM-48, src/lifemate/synth/scenarios.py duyệt thẳng `set` các tên trường ở hai chỗ của `Builder.choose`, nên
thứ tự phụ thuộc hàm băm chuỗi của tiến trình Python (PYTHONHASHSEED ngẫu nhiên): cùng `--seed` mà mỗi lần chạy ra một
bộ 600 mẫu khác, và file đã dùng để train không sinh lại được từ seed. Bộ sinh giờ duyệt theo bảng chữ cái (tất định).
Script này dò, cho từng lần thử dựng mẫu của lượt cũ, thứ tự nào làm bộ sinh ra đúng dòng kế tiếp của file tham chiếu,
rồi ghi chuỗi thứ tự đó ra JSON; `scripts/build_sft.py --synth-replay FILE` phát lại chuỗi ấy.

    .venv/bin/python scripts/synth_orders.py find --ref sft/train_synth.jsonl --out docs/repro/synth_v1_orders.json
    .venv/bin/python scripts/synth_orders.py check --ref sft/train_synth.jsonl --replay docs/repro/synth_v1_orders.json

Cách dò: mỗi lần thử (một vòng của `generate`) chạy lại từ cùng trạng thái bộ sinh số ngẫu nhiên với từng hoán vị ở
từng chỗ duyệt; nhánh nào dựng ra câu nói không có trong mẫu tham chiếu thì cắt ngay, nhánh ra đúng dòng tham chiếu
thì đi tiếp sang mẫu sau, nhánh bỏ kịch bản (Skip) thì thử lần kế cho cùng mẫu; đi nhầm thì quay lui. Dò ra đủ mọi dòng
nghĩa là file tham chiếu chỉ phụ thuộc mã, seed, tập train và thứ tự duyệt: không có gì khác (kể cả catalog của tập
public mà bản cũ từng đọc để tránh trùng tên app) ảnh hưởng tới nó.

Chỉ dùng tập train, một tiến trình CPU. File tham chiếu sinh từ dữ liệu BTC nên chỉ nằm ở máy có nó; file thứ tự chỉ
chứa tên trường của src/lifemate/synth/domains.py và mã băm của file tham chiếu, đưa lên GitHub được.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from lifemate import config, splits, synth
from lifemate.config import REPO
from lifemate.registry import build
from lifemate.synth import scenarios

_spec = importlib.util.spec_from_file_location("build_sft", REPO / "scripts" / "build_sft.py")
build_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft)

KINDS = {"v1": scenarios.KINDS_V1, "v2": scenarios.KINDS}
MAX_SKIPS = 12  # số lần bỏ kịch bản (Skip) liên tiếp tối đa trước một mẫu


class Mismatch(Exception):
    """Nhánh đang thử dựng ra câu nói không có trong mẫu tham chiếu."""


class Explorer:
    """Hàm thứ tự cho một lần thử: theo `plan` (chỉ số hoán vị ở từng chỗ duyệt); chỗ chưa có trong plan lấy hoán vị 0.

    Hoán vị 0 của một tập là thứ tự đã thấy đúng cho tập đó ở một mẫu trước (`prefer`), nếu có: trong một tiến trình,
    cùng một tập thường được duyệt theo cùng một thứ tự, nên nhánh thử đầu tiên hầu như luôn là nhánh đúng."""

    def __init__(self, plan: list[int], prefer: dict):
        self.plan, self.prefer, self.points = plan, prefer, []

    def perms(self, xs: list) -> list[tuple]:
        first = self.prefer.get(frozenset(xs))
        rest = itertools.permutations(xs)
        return ([tuple(first)] + [p for p in rest if p != tuple(first)]) if first else list(rest)

    def __call__(self, xs) -> list:
        xs = sorted(xs)
        if len(xs) < 2:
            return xs
        k = self.plan[len(self.points)] if len(self.points) < len(self.plan) else 0
        self.points.append((xs, k))
        return list(self.perms(xs)[k])

    def orders(self) -> list[list[str]]:
        return [list(self.perms(xs)[k]) for xs, k in self.points]


def next_plan(points: list) -> list[int] | None:
    """Kế hoạch kế tiếp theo kiểu đồng hồ số trên các chỗ duyệt đã gặp; None khi đã hết."""
    pts = [(len(xs), k) for xs, k in points]
    while pts and pts[-1][1] + 1 >= math.factorial(pts[-1][0]):
        pts.pop()
    return None if not pts else [k for _, k in pts[:-1]] + [pts[-1][1] + 1]


def sft_line(agent, sample: dict) -> str:
    """Dòng JSONL mà scripts/build_sft.py ghi cho một mẫu tổng hợp."""
    msgs = build_sft.conversation(agent, sample, synth.transcript(sample), None, specs=synth.specs(sample))
    return json.dumps({"id": sample["id"], "synth": sample["synth"]["kind"], "messages": msgs}, ensure_ascii=False)


def lines_sha(lines: list[str]) -> str:
    """sha256 của các dòng mẫu tổng hợp, mỗi dòng kết thúc bằng xuống dòng như trong file."""
    return hashlib.sha256("".join(x + chr(10) for x in lines).encode("utf-8")).hexdigest()


def load_ref(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    lines = [x for x in raw.decode("utf-8").splitlines() if x.startswith('{"id": "synth_')]
    return lines, hashlib.sha256(raw).hexdigest()


def setup(a):
    cfg = config.load(REPO / a.config)
    agent = build("agent", cfg["agent"])
    hosts = splits.select(build_sft.SPLIT, a.subset)
    return cfg, agent, hosts


def find(a) -> None:
    cfg, agent, hosts = setup(a)
    ref, ref_sha = load_ref(Path(a.ref))
    n = min(len(ref), a.limit or len(ref))
    kinds, avoid = KINDS[a.kinds], scenarios.real_app_ids()
    rng = random.Random(a.seed)
    target = {"text": "", "prune": False}
    prefer: dict[frozenset, list] = {}
    real_intent = scenarios.Intent
    stats = {"attempts": 0, "skips": 0, "runs": 0, "max_skips_in_a_row": 0}

    def checked_intent(core, emit):
        """Mọi ý định của lần thử ra đúng mẫu tham chiếu đều có mặt trong lời nói của mẫu đó (lịch sử hoặc câu cuối)."""
        if target["prune"]:
            pat = re.escape(core).replace(re.escape("{p}"), r"\w+")
            if not re.search(pat, target["text"], re.I):
                raise Mismatch(core)
        return real_intent(core, emit)

    scenarios.Intent = checked_intent

    def leaves(state, idx: int, prune: bool):
        """Mọi nhánh thứ tự của một lần thử cho mẫu `idx`: (thứ tự, "match" | "skip" | None, trạng thái rng sau đó).
        prune: cắt nhánh ngay khi một ý định không có trong mẫu tham chiếu; khi đó chỉ kết cục "match" là đáng tin."""
        target.update(text=json.loads(ref[idx])["messages"][1]["content"], prune=prune)
        plan: list[int] | None = []
        while plan is not None:
            rng.setstate(state)
            ex = Explorer(plan, prefer)
            stats["runs"] += 1
            try:
                sample = scenarios.attempt(rng, hosts, kinds, avoid, ex, idx + 1)
                kind = "skip" if sample is None else "match" if prune and sft_line(agent, sample) == ref[idx] else None
            except Exception:  # Mismatch, hoặc lỗi chỉ có ở nhánh sai (lượt cũ đã chạy hết, nên nhánh đúng không lỗi)
                kind = None
            yield ex.orders(), kind, rng.getstate()
            plan = next_plan(ex.points)

    def solve(state, idx: int, depth: int):
        """Chuỗi lần thử từ `state` gồm tối đa `depth` lần bỏ kịch bản rồi một lần ra đúng mẫu `idx`:
        ([thứ tự của từng lần thử], trạng thái rng sau đó), hoặc None."""
        for orders, kind, after in leaves(state, idx, True):
            if kind == "match":
                return [orders], after
        if depth == 0:
            return None
        seen = set()
        for orders, kind, after in leaves(state, idx, False):
            if kind != "skip" or after in seen:
                continue
            seen.add(after)
            sub = solve(after, idx, depth - 1)
            if sub:
                return [orders] + sub[0], sub[1]
        return None

    trace: list[list[str]] = []
    state, t0 = rng.getstate(), time.time()
    for idx in range(n):
        found = next((r for r in (solve(state, idx, d) for d in range(MAX_SKIPS + 1)) if r), None)
        if not found:
            scenarios.Intent = real_intent
            sys.exit(f"KHÔNG dò được: dừng ở mẫu synth_{idx + 1:05d} (đã khớp {idx}/{n}). File tham chiếu không sinh từ mã "
                     f"này với --kinds {a.kinds}, --seed {a.seed}, --subset {a.subset}, hoặc có yếu tố khác ngoài thứ tự duyệt.")
        tries, state = found
        for orders in tries:
            trace.extend(orders)
            for o in orders:
                prefer[frozenset(o)] = o
        stats["attempts"] += len(tries)
        stats["skips"] += len(tries) - 1
        stats["max_skips_in_a_row"] = max(stats["max_skips_in_a_row"], len(tries) - 1)
        if (idx + 1) % 50 == 0 or idx + 1 == n:
            print(f"  {idx + 1}/{n} mẫu, {stats['attempts']} lần thử, {stats['runs']} lượt chạy, {time.time() - t0:.0f} giây",
                  file=sys.stderr, flush=True)
    scenarios.Intent = real_intent
    stats.update(points=len(trace), max_perms=max([math.factorial(len(o)) for o in trace] or [1]))
    out = {
        "about": "Thứ tự duyệt set của lượt sinh mẫu tổng hợp cũ; phát lại bằng scripts/build_sft.py --synth-replay. "
                 "Dò bằng scripts/synth_orders.py find, chỉ với tập train.",
        "ref_sha256": ref_sha, "synth_lines_sha256": lines_sha(ref[:n]),
        "n": n, "seed": a.seed, "subset": a.subset, "kinds": a.kinds, "config": a.config, "stats": stats,
        "orders": trace,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "orders"}, ensure_ascii=False), "→", a.out)


def check(a) -> None:
    cfg, agent, hosts = setup(a)
    ref, ref_sha = load_ref(Path(a.ref))
    rep = json.loads(Path(a.replay).read_text(encoding="utf-8"))
    samples = synth.generate(hosts, rep["n"], seed=rep["seed"], kinds=KINDS[rep["kinds"]], orders=rep["orders"])
    lines = [sft_line(agent, s) for s in samples]
    dropped = sum(1 for s in samples if synth.replay(s, cfg.get("guards")))
    same = sum(x == y for x, y in zip(lines, ref))
    print(json.dumps({"ref_sha256": ref_sha, "ref_matches_recorded": ref_sha == rep["ref_sha256"], "n": rep["n"],
                      "identical_lines": same, "dropped_by_replay": dropped,
                      "synth_lines_sha256_ok": lines_sha(lines) == rep["synth_lines_sha256"]},
                     ensure_ascii=False))
    if (same != rep["n"] or len(ref) < rep["n"] or dropped or ref_sha != rep["ref_sha256"]
            or lines_sha(lines) != rep["synth_lines_sha256"]):
        sys.exit("KHÁC file tham chiếu")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("find", "check"):
        p = sub.add_parser(name)
        p.add_argument("--ref", required=True, help="file SFT tham chiếu (dòng synth_* là các mẫu tổng hợp)")
        p.add_argument("--config", default="configs/experiments/10_qwen3-4b_gold_react.yaml")
        p.add_argument("--subset", default=build_sft.SUBSET, choices=[build_sft.SUBSET, "all"], help="tập mẫu chủ")
    f = sub.choices["find"]
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=2026)
    f.add_argument("--kinds", default="v1", choices=list(KINDS), help="v1: bảng kịch bản trước khi có pick_detail")
    f.add_argument("--limit", type=int, default=None, help="chỉ dò N mẫu đầu (thử nhanh)")
    sub.choices["check"].add_argument("--replay", required=True)
    a = ap.parse_args(argv)
    {"find": find, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
