"""scripts/decision_sweep.py: hỗn hợp record / force theo ngưỡng và 2-fold theo app."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decision_sweep.py"
APPS = ["appliances", "door_lock", "fitness", "learning"]


def write_run(d: Path, rows):
    d.mkdir(parents=True)
    with open(d / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sample(i, kind, p, score, app):
    step = {"action": "stop"} if kind == "stop" else {"action": "call", "name": "sys.ask_missing_info" if kind == "ask" else f"{app}.x"}
    if kind == "stop":
        step["p_stop"] = p
    elif kind == "ask":
        step["p_ask"] = p
    return {"id": f"s{i}", "score": score, "steps": [step], "gold_calls": [{"name": f"{app}.x"}]}


def test_sweep_mixes_force_only_below_the_threshold(tmp_path):
    rec, frc = [], []
    for i in range(16):
        app = APPS[i % 4]
        kind = ["ask", "stop", "act", "ask"][i % 4]
        p = 0.2 if (i // 4) % 2 == 0 else 0.95  # nghi ngờ ở nửa số mẫu của mỗi kiểu
        rec.append(sample(i, kind, p, 0.0 if kind != "act" else 1.0, app))
        frc.append(sample(i, "act", None, 1.0 if kind == "act" or p < 0.5 else 0.0, app))
    write_run(tmp_path / "rec", rec)
    write_run(tmp_path / "frc", frc)
    out = subprocess.run([sys.executable, str(SCRIPT), "--app", str(tmp_path / "rec"), str(tmp_path / "frc"),
                          "--json", str(tmp_path / "o.json")], capture_output=True, text=True, check=True).stdout
    assert "chọn làm mà force khác điểm 0/" in out
    res = json.loads((tmp_path / "o.json").read_text())
    ch = res["chosen"]
    # mẫu nghi ngờ (p = 0,2) được lợi khi ép làm, mẫu chắc (p = 0,95) bị hại: ngưỡng phải nằm giữa
    assert 0.2 < ch["ask_min"] <= 0.95 and 0.2 < ch["stop_min"] <= 0.95
    assert res["oof_delta"] > 0
