"""Chọn ngưỡng hỏi lại / stop cho agent.decision (LM-25) trên CPU, từ hai lượt chạy có sẵn của cùng một tập:

    record   config 130: hành vi mặc định, có p_stop / p_ask ở từng bước
    force    config 131: bước quyết định đầu bị cấm stop và hỏi lại

Với ngưỡng (ask_min, stop_min), mỗi mẫu lấy trajectory của force nếu bước quyết định đầu của record là hỏi lại với
p_ask < ask_min hoặc stop với p_stop < stop_min, ngược lại lấy record. Điểm hỗn hợp là trung bình `score` từng mẫu,
không cần chấm lại. Chính xác cho phạm vi `scope: first`; phạm vi pre_write phải đo đầu-cuối (config 133).

Ngưỡng chọn bằng 2-fold theo app trên val_app: chia 10 app giữ lại thành hai nửa cố định, chọn trên nửa này, chấm
trên nửa kia rồi đổi vai. val_random không dùng để chọn, chỉ để kiểm.

    .venv/bin/python scripts/decision_sweep.py --app RECORD_VA FORCE_VA --random RECORD_VR FORCE_VR [--groups CSV]

--groups: lm33-first-wrong.csv (relay-codex/analysis) để báo mẫu đổi điểm theo nhóm lỗi của zs0p.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lifemate.splits import HELDOUT_APPS  # noqa: E402

ASK = "sys.ask_missing_info"
GRID = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
WEIGHTS = {"val_random": 0.35, "val_app": 0.65}


def load(run: str) -> dict:
    out = {}
    with open(os.path.join(run, "samples.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in out:
                raise ValueError(f"id trùng trong {run}: {r['id']}")
            out[r["id"]] = r
    return out


def first_decision(sample: dict) -> dict | None:
    """Bước đầu có action call / stop: {kind: stop | ask | act, p_stop, p_ask}."""
    for s in sample.get("steps", []):
        if s.get("action") in ("call", "stop"):
            kind = "stop" if s["action"] == "stop" else ("ask" if s.get("name") == ASK else "act")
            return {"kind": kind, "p_stop": s.get("p_stop"), "p_ask": s.get("p_ask")}
    return None


def overridden(d: dict | None, ask_min: float, stop_min: float) -> bool:
    if not d:
        return False
    if d["kind"] == "stop" and d["p_stop"] is not None:
        return d["p_stop"] < stop_min
    if d["kind"] == "ask" and d["p_ask"] is not None:
        return d["p_ask"] < ask_min
    return False


def apps_of(sample: dict) -> set[str]:
    return {c["name"].split(".")[0] for c in sample.get("gold_calls") or []} & set(HELDOUT_APPS)


def app_folds(pair) -> list[list[str]]:
    """Các app cùng xuất hiện phải cùng fold; tránh rò app khi một mẫu dùng nhiều app giữ lại."""
    groups = []
    for i in pair.ids:
        group = apps_of(pair.rec[i])
        if not group:
            continue
        keep = []
        for old in groups:
            if old & group:
                group |= old
            else:
                keep.append(old)
        groups = [*keep, group]
    halves = [[], []]
    for group in sorted(groups, key=lambda g: (-len(g), sorted(g))):
        k = min(range(2), key=lambda j: (len(halves[j]), j))
        halves[k].extend(sorted(group))
    if not all(halves):
        raise ValueError("không đủ nhóm app độc lập để chia 2-fold")
    return [sorted(h) for h in halves]


class Pair:
    def __init__(self, subset: str, record: str, force: str):
        self.subset = subset
        self.rec, self.frc = load(record), load(force)
        if self.rec.keys() != self.frc.keys():
            raise ValueError(f"id record/force không khớp: record-only {len(self.rec.keys() - self.frc.keys())}, "
                             f"force-only {len(self.frc.keys() - self.rec.keys())}")
        if not self.rec:
            raise ValueError("cặp run không có id")
        self.ids = sorted(self.rec)
        self.dec = {i: first_decision(self.rec[i]) for i in self.ids}

    def check(self) -> str:
        """Mẫu record chọn làm ở bước đầu thì force phải ra cùng điểm (ràng buộc không đổi lựa chọn argmax)."""
        acts = [i for i in self.ids if self.dec[i] and self.dec[i]["kind"] == "act"]
        diff = [i for i in acts if self.rec[i]["score"] != self.frc[i]["score"]]
        kinds = Counter(d["kind"] if d else "none" for d in self.dec.values())
        missing = sum(1 for d in self.dec.values() if d and d["kind"] != "act" and
                      d["p_stop" if d["kind"] == "stop" else "p_ask"] is None)
        return (f"{self.subset}: n {len(self.ids)}; bước đầu {dict(kinds)}; thiếu xác suất {missing}; "
                f"chọn làm mà force khác điểm {len(diff)}/{len(acts)}" + (f" ({', '.join(diff[:5])})" if diff else ""))

    def scores(self, ask_min: float, stop_min: float, ids=None) -> list[float]:
        ids = self.ids if ids is None else ids
        return [(self.frc if overridden(self.dec[i], ask_min, stop_min) else self.rec)[i]["score"] for i in ids]

    def delta(self, ask_min, stop_min, ids=None) -> tuple[float, float, int, int, int]:
        """(mean diff × 100, SE ghép cặp × 100, số mẫu ghi đè, tốt lên, tệ đi) so với record."""
        ids = self.ids if ids is None else ids
        mix = self.scores(ask_min, stop_min, ids)
        d = [100 * (m - self.rec[i]["score"]) for m, i in zip(mix, ids)]
        n = len(d)
        mean = sum(d) / n
        se = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 0.0
        k = sum(overridden(self.dec[i], ask_min, stop_min) for i in ids)
        return mean, se, k, sum(x > 0 for x in d), sum(x < 0 for x in d)

    def base(self) -> float:
        return 100 * sum(self.rec[i]["score"] for i in self.ids) / len(self.ids)


def best(pair: Pair, ids) -> tuple[float, float]:
    """Ngưỡng có điểm cao nhất trên `ids`; hoà thì chọn ghi đè ít hơn (ngưỡng thấp hơn)."""
    cands = []
    for a in GRID:
        for s in GRID:
            mean = sum(pair.scores(a, s, ids)) / len(ids)
            k = sum(overridden(pair.dec[i], a, s) for i in ids)
            cands.append((round(mean, 9), -k, -a, -s, a, s))
    c = max(cands)
    return c[4], c[5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", nargs=2, required=True, metavar=("RECORD", "FORCE"))
    ap.add_argument("--random", nargs=2, metavar=("RECORD", "FORCE"))
    ap.add_argument("--groups", help="lm33-first-wrong.csv")
    ap.add_argument("--json", help="ghi kết quả ra file JSON")
    a = ap.parse_args()
    va = Pair("val_app", *a.app)
    vr = Pair("val_random", *a.random) if a.random else None
    pairs = [p for p in (va, vr) if p]
    for p in pairs:
        print(p.check())
    print()

    print("Lưới ngưỡng (Δ so với record, ×100; ± SE ghép cặp; k mẫu ghi đè, tốt / tệ):")
    print(f"{'ask_min':>7} {'stop_min':>8} | " + " | ".join(f"{p.subset:^34}" for p in pairs) + " | công khai")
    rows = []
    for am in GRID:
        for sm in GRID:
            cells, pub = [], 0.0
            for p in pairs:
                mean, se, k, up, dn = p.delta(am, sm)
                cells.append(f"{mean:+6.2f} ± {se:4.2f}  k {k:3d}  {up:2d}↑ {dn:2d}↓")
                pub += WEIGHTS[p.subset] * mean
            rows.append((am, sm, pub))
            if am in (0.0, 0.5, 0.7, 0.9, 0.99, 1.01) and sm in (0.0, 0.5, 0.7, 0.9, 0.99, 1.01):
                print(f"{am:7.2f} {sm:8.2f} | " + " | ".join(cells) + (f" | {pub:+.2f}" if vr else ""))

    halves = app_folds(va)
    print(f"\n2-fold theo app trên val_app: nửa A {halves[0]}, nửa B {halves[1]}")
    oof, report = {}, []
    for k, (fit, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        fit_ids = [i for i in va.ids if apps_of(va.rec[i]) and apps_of(va.rec[i]) <= set(fit)]
        test_ids = [i for i in va.ids if apps_of(va.rec[i]) and apps_of(va.rec[i]) <= set(test)]
        am, sm = best(va, fit_ids)
        for i, s in zip(test_ids, va.scores(am, sm, test_ids)):
            oof[i] = s
        line = f"fold {k + 1}: chọn trên {len(fit_ids)} mẫu → ask_min {am}, stop_min {sm}; chấm {len(test_ids)} mẫu"
        mean, se, kk, up, dn = va.delta(am, sm, test_ids)
        line += f": Δ {mean:+.2f} ± {se:.2f} (k {kk}, {up}↑ {dn}↓)"
        if vr:
            m2, s2, k2, u2, d2 = vr.delta(am, sm)
            line += f"; val_random Δ {m2:+.2f} ± {s2:.2f} (k {k2}, {u2}↑ {d2}↓)"
        report.append({"fold": k + 1, "ask_min": am, "stop_min": sm})
        print(line)
    rest = [i for i in va.ids if i not in oof]
    d = [100 * (oof[i] - va.rec[i]["score"]) for i in oof] + [0.0] * len(rest)
    n = len(d)
    oof_mean = sum(d) / n
    oof_se = (sum((x - oof_mean) ** 2 for x in d) / (n - 1)) ** 0.5 / n ** 0.5
    print(f"val_app ngoài fold: Δ {oof_mean:+.2f} ± {oof_se:.2f} trên {n} mẫu ({len(rest)} mẫu không gắn app, giữ record)")

    am, sm = best(va, va.ids)
    print(f"\nNgưỡng chọn trên cả val_app (cho config 132): ask_min {am}, stop_min {sm}")
    for p in pairs:
        mean, se, k, up, dn = p.delta(am, sm)
        print(f"  {p.subset}: {p.base():.2f} → {p.base() + mean:.2f} (Δ {mean:+.2f} ± {se:.2f}; k {k}, {up}↑ {dn}↓)")

    if a.groups:
        lab = {(r["subset"], r["id"]): r["group"] for r in csv.DictReader(open(a.groups, encoding="utf-8"))}
        for p in pairs:
            by = Counter()
            for i in p.ids:
                if overridden(p.dec[i], am, sm):
                    g = lab.get((p.subset, i), "OK")
                    dd = p.frc[i]["score"] - p.rec[i]["score"]
                    by[(g, "↑" if dd > 0 else "↓" if dd < 0 else "=")] += 1
            print(f"  {p.subset} mẫu ghi đè theo nhóm LM-33: " +
                  ", ".join(f"{g}{s} {c}" for (g, s), c in sorted(by.items())))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"folds": report, "oof_delta": oof_mean, "oof_se": oof_se, "chosen": {"ask_min": am, "stop_min": sm},
                       "grid": [{"ask_min": x, "stop_min": y, "public_weighted": z} for x, y, z in rows]}, f, indent=1)


if __name__ == "__main__":
    main()
