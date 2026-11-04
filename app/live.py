"""Tab Đang chạy: GPU, khoá GPU dùng chung, tiến trình, huấn luyện, lượt đánh giá dở, server vLLM, và xếp hàng việc mới.

Đọc mọi worktree của repo (git worktree list), nên thấy cả việc chạy song song ở các worktree khác. Chỉ dừng được việc
do chính web-app khởi động (ghi trong logs/jobs.jsonl); việc khác chỉ xem, không đụng tới.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

STEP = re.compile(r"bước (\d+)/(\d+) loss ([\d.]+) lr \S+ (\d+) tok/s, còn (\d+) phút, bộ nhớ đỉnh ([\d.]+) GB")
NOTE = re.compile(r"^- (\d{4}-\d\d-\d\d \d\d:\d\d) ([^:]+): (.*)$")
SHELL_WRAPPER = re.compile(r"^(/usr)?(/bin/)?(ba)?sh -c ")
# mẫu trong dòng lệnh → loại việc; thứ tự quan trọng (dòng lệnh của hàng đợi có chứa tên các script con)
KINDS = [("train_sft.py", "Huấn luyện LLM"), ("finetune_", "Huấn luyện ASR"), ("gpu_job.sh", "Hàng đợi GPU"),
         ("run_queue.sh", "Hàng đợi config"), ("lifemate run", "Đánh giá"), ("build_sft.py", "Dựng dữ liệu SFT"),
         ("vllm", "Server vLLM"), ("streamlit", None)]


def worktrees(repo: Path) -> list[Path]:
    """Mọi worktree của repo, để gộp runs/ và logs/ của các việc chạy song song."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, timeout=5).stdout
        roots = [Path(ln[9:]) for ln in out.splitlines() if ln.startswith("worktree ")]
    except (OSError, subprocess.SubprocessError):
        roots = []
    return [r for r in roots if r.exists()] or [repo]


def shared_notes(repo: Path) -> tuple[Path, Path]:
    """(GPU.md, gpu.lock) dùng chung: biến GPU_LOG và GPU_LOCK, mặc định logs/ của repo như scripts/gpu_job.sh."""
    log = Path(os.environ.get("GPU_LOG", repo / "logs" / "GPU.md"))
    return log, Path(os.environ.get("GPU_LOCK", log.parent / "gpu.lock"))


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu() -> tuple[dict | None, dict[int, int]]:
    """Thông số card và bộ nhớ GPU (MiB) của từng pid."""
    q = _run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
              "--format=csv,noheader,nounits"]).strip().splitlines()
    if not q:
        return None, {}
    name, util, used, total, temp, power = (x.strip() for x in q[0].split(","))
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    mem = {}
    for ln in apps.strip().splitlines():
        pid, m = (x.strip() for x in ln.split(","))
        if pid.isdigit() and m.isdigit():
            mem[int(pid)] = int(m)
    return {"name": name, "util": util, "used": int(used), "total": int(total), "temp": temp, "power": power}, mem


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def descendants(pid: int) -> list[int]:
    try:
        kids = [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []
    return kids + [g for k in kids for g in descendants(k)]


def kind_of(args: str) -> str | None:
    low = args.lower()
    return next((k for pat, k in KINDS if pat in low), "")


def where(pid: int, roots: list[Path]) -> str:
    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return ""
    for r in sorted(roots, key=lambda p: -len(str(p))):
        if cwd == r or r in cwd.parents:
            return r.name
    return str(cwd)


def elapsed(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}" if sec >= 3600 else f"{sec // 60} phút"


def processes(roots: list[Path], gpu_mem: dict[int, int]) -> pd.DataFrame:
    rows = []
    for ln in _run(["ps", "-eo", "pid=,etimes=,args="]).splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3 or SHELL_WRAPPER.match(parts[2]):  # vỏ `bash -c`: tiến trình con mới là việc thật
            continue
        pid, et, args = int(parts[0]), int(parts[1]), parts[2]
        kind = kind_of(args)
        if gpu_mem.get(pid, 0) >= 500 and not kind:  # bỏ qua trình duyệt, IDE... chỉ giữ vài trăm MiB
            kind = "Dùng GPU"
        if kind:
            rows.append({"pid": pid, "loại": kind, "worktree": where(pid, roots), "đã chạy": elapsed(et),
                         "GPU MiB": gpu_mem.get(pid), "lệnh": args[:240]})
    df = pd.DataFrame(rows, columns=["pid", "loại", "worktree", "đã chạy", "GPU MiB", "lệnh"])
    return df.sort_values(["GPU MiB", "loại"], ascending=[False, True], na_position="last")


def lock_state(lock: Path) -> tuple[int | None, list[int]]:
    """(pid đang giữ khoá, các pid đang chờ) theo /proc/locks."""
    try:
        ino = str(lock.stat().st_ino)
        lines = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return None, []
    holder, waiting = None, []
    for ln in lines:  # "1: FLOCK  ADVISORY  WRITE 1234 103:02:5678 0 EOF", chờ thì có "->" sau số thứ tự
        parts = ln.split()
        blocked = "->" in parts
        parts = [p for p in parts if p != "->"]
        if len(parts) >= 6 and parts[1] == "FLOCK" and parts[5].rsplit(":", 1)[-1] == ino:
            if blocked:
                waiting.append(int(parts[4]))
            else:
                holder = int(parts[4])
    return holder, waiting


def job_of(pid: int) -> str:
    """Việc thật sự đứng sau một pid flock: tiến trình con có loại đã biết, không thì chính nó."""
    named = [(p, cmdline(p)) for p in descendants(pid)]
    known = [c for _, c in named if kind_of(c) not in ("", None, "Hàng đợi GPU", "Hàng đợi config")]
    return (known[0] if known else cmdline(pid))[:240]


def gpu_notes(path: Path) -> pd.DataFrame:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return pd.DataFrame(columns=["lúc", "ai", "việc"])
    rows = [m.groups() for ln in lines if (m := NOTE.match(ln))]
    return pd.DataFrame(rows[::-1], columns=["lúc", "ai", "việc"])


def tail(path: Path, n: int = 6) -> str:
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 4000))
        text = f.read().decode(errors="replace")
    lines = [x for x in re.split(r"[\r\n]+", text) if x.strip()]
    return "\n".join(lines[-n:])


def train_logs(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for p in (root / "logs").glob("train_*.log"):
            text = p.read_text(encoding="utf-8", errors="replace")
            steps = [tuple(float(x) for x in m.groups()) for m in STEP.finditer(text)]
            age = time.time() - p.stat().st_mtime
            status = ("lỗi" if "Traceback" in text else "xong" if ("đã gộp" in text or "adapter →" in text) else
                      "đang chạy" if age < 600 else "dừng giữa chừng")
            out.append({"worktree": root.name, "name": p.stem[len("train_"):], "path": p, "steps": steps,
                        "status": status, "age": age})
    return sorted(out, key=lambda r: r["age"])


def open_runs(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for cfg in (root / "runs").glob("*/config.yaml"):
            run = cfg.parent
            if (run / "metrics.json").exists():
                continue
            age = time.time() - max(p.stat().st_mtime for p in run.iterdir())
            prog = {}
            if (run / "progress.json").exists():
                prog = json.loads((run / "progress.json").read_text(encoding="utf-8"))
            rows.append({"worktree": root.name, "lượt chạy": run.name,
                         "tiến độ": f"{prog['done']}/{prog['total']}" if prog else "ASR / khởi động",
                         "điểm tạm": prog.get("score"), "cập nhật": f"{age / 60:.0f} phút trước",
                         "trạng thái": "đang chạy" if age < 600 else "bỏ dở"})
    return pd.DataFrame(rows)


def vllm_models(url: str = "http://localhost:8000/v1/models") -> list[str] | None:
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except (OSError, ValueError):
        return None


# -- việc do web-app khởi động ------------------------------------------------------------------------------
def _jobs_file(repo: Path) -> Path:
    return repo / "logs" / "jobs.jsonl"


def jobs(repo: Path) -> list[dict]:
    f = _jobs_file(repo)
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []


def alive(pid: int) -> bool:
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # con của web-app đã xong: dọn zombie
            return False
    except ChildProcessError:
        pass  # không phải con của tiến trình này (vd. web-app đã khởi động lại)
    return "gpu_job.sh" in cmdline(pid) or "flock" in cmdline(pid)


def launch(repo: Path, argv: list[str], label: str, env: dict[str, str]) -> None:
    (repo / "logs").mkdir(exist_ok=True)
    log = repo / "logs" / f"job_{datetime.now():%Y%m%d-%H%M%S}.log"
    notes, lock = shared_notes(repo)
    full_env = {**os.environ, "GPU_LOG": str(notes), "GPU_LOCK": str(lock), "WHO": f"web-app ({repo.name})", **env}
    with log.open("w") as out:
        p = subprocess.Popen([str(repo / "scripts" / "gpu_job.sh"), *argv], cwd=repo, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=full_env)
    with _jobs_file(repo).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"pid": p.pid, "lúc": f"{datetime.now():%F %H:%M}", "việc": label, "log": str(log)},
                           ensure_ascii=False) + "\n")


# -- giao diện ----------------------------------------------------------------------------------------------
def _monitor(repo: Path, roots: list[Path], colors: list[str]) -> None:
    info, mem = gpu()
    served = vllm_models()
    c1, c2, c3, c4 = st.columns(4)
    if info:
        c1.metric("GPU", f"{info['util']}%", help=info["name"])
        c2.metric("Bộ nhớ GPU", f"{info['used'] / 1024:.1f}/{info['total'] / 1024:.0f} GB")
        c3.metric("Nhiệt độ", f"{info['temp']}°C", help=f"Công suất {info['power']} W")
    else:
        c1.metric("GPU", "không đọc được")
    c4.metric("Server vLLM :8000", ", ".join(m.split("/")[-1] for m in served) if served else "tắt")

    notes, lock = shared_notes(repo)
    holder, waiting = lock_state(lock)
    st.markdown("**Khoá GPU dùng chung**")
    if holder:
        st.success(f"Đang giữ: pid {holder} · {where(holder, roots)} · `{job_of(holder)}`")
    else:
        st.info("Khoá đang trống.")
    if waiting:
        st.warning("Đang chờ khoá: " + " · ".join(f"pid {p} ({where(p, roots)}): `{job_of(p)[:120]}`" for p in waiting))
    st.caption(f"Khoá: `{lock}` · nhật ký: `{notes}`")

    st.markdown("**Tiến trình liên quan (mọi worktree)**")
    st.dataframe(processes(roots, mem), hide_index=True, width="stretch")

    st.markdown("**Huấn luyện**")
    logs = train_logs(roots)
    if not logs:
        st.caption("Chưa có logs/train_*.log nào.")
    for i, r in enumerate(logs[:4]):
        s = r["steps"]
        head = f"`{r['name']}` · {r['worktree']} · **{r['status']}**"
        if s:
            step, total, loss, tok, eta, peak = s[-1]
            head += f" · bước {step:.0f}/{total:.0f} · loss {loss:.4f} · {tok:.0f} token/s · đỉnh {peak} GB"
            if r["status"] == "đang chạy":
                head += f" · còn khoảng {eta:.0f} phút"
        st.markdown(head)
        if s:
            st.progress(min(1.0, s[-1][0] / s[-1][1]))
            df = pd.DataFrame(s, columns=["bước", "tổng", "loss", "token/s", "còn (phút)", "đỉnh GB"])
            line = alt.Chart(df).mark_line(point=True, color=colors[i % len(colors)]).encode(
                x=alt.X("bước:Q", scale=alt.Scale(domain=[0, s[-1][1]])), y=alt.Y("loss:Q", title="loss"),
                tooltip=["bước", "loss", "token/s", "đỉnh GB"])
            st.altair_chart(line.properties(height=160), width="stretch")
        else:
            st.code(tail(r["path"], 3), language=None)

    st.markdown("**Lượt đánh giá chưa xong**")
    opened = open_runs(roots)
    if len(opened):
        st.dataframe(opened, hide_index=True, width="stretch")
    else:
        st.caption("Không có lượt nào đang dở.")

    mine = jobs(repo)[-8:][::-1]
    if mine:
        st.markdown("**Việc do web-app xếp hàng**")
        for j in mine:
            on = alive(j["pid"])
            cols = st.columns([5, 1])
            cols[0].markdown(f"{'🟡 đang chạy hoặc chờ khoá' if on else '⚪ đã xong'} · {j['lúc']} · {j['việc']}")
            if on and cols[1].button("Dừng", key=f"stop_{j['pid']}"):
                try:
                    os.killpg(j["pid"], signal.SIGTERM)  # cả nhóm tiến trình của việc đó, kể cả server vLLM nó bật
                except OSError:
                    pass
            with st.expander("Nhật ký", expanded=False):
                p = Path(j["log"])
                st.code(tail(p, 12) if p.exists() else "(chưa có)", language=None)

    st.markdown("**Nhật ký khoá GPU**")
    st.dataframe(gpu_notes(notes).head(15), hide_index=True, width="stretch")
    with st.expander("Nhật ký khác trong logs/ (24 giờ gần nhất)"):
        others = sorted((p for r in roots for p in (r / "logs").glob("*.log")
                         if not p.name.startswith("train_") and time.time() - p.stat().st_mtime < 86400),
                        key=lambda p: -p.stat().st_mtime)
        for p in others[:12]:
            st.markdown(f"`{p.parent.parent.name}/logs/{p.name}` · {elapsed(time.time() - p.stat().st_mtime)} trước")
            st.code(tail(p, 4), language=None)


def _control(repo: Path, load_config) -> None:
    st.subheader("Xếp hàng việc GPU mới")
    st.caption("Việc chờ khoá GPU chung rồi mới chạy, nên không giẫm lên việc của người khác. Config đã có kết quả trong runs/ "
               "được bỏ qua (scripts/run_queue.sh). Chỉ nút Dừng ở trên mới tắt được việc, và chỉ việc do web-app xếp.")
    what = st.radio("Loại việc", ["Đánh giá config", "Huấn luyện SFT"], horizontal=True)
    cfg_dir = repo / "configs" / "experiments"
    if what == "Đánh giá config":
        names = st.multiselect("Config", [p.stem for p in sorted(cfg_dir.glob("*.yaml"))])
        sets = st.text_input("Ghi đè cho cả hàng (SET)", value="limit=100")
        models = set()
        for n in names:
            ag = load_config(cfg_dir / f"{n}.yaml").get("agent") or {}
            models.add(ag.get("llm", {}).get("model") if ag.get("name") in ("react_json", "native_tools") else None)
        if len(models) > 1:
            st.error(f"Các config dùng mô hình khác nhau ({', '.join(str(m) for m in models)}): xếp thành nhiều việc.")
        elif names:
            model = next(iter(models))
            missing = model and model.startswith("models/") and not (repo / model).exists()
            st.caption(f"Server: `{model}` (DETERMINISTIC=1)" if model else "Không cần vLLM.")
            if missing:
                st.error(f"Chưa có `{model}` trong worktree này.")
            elif st.button("Xếp hàng đánh giá", type="primary"):
                argv = (["eval", model] if model else ["run"]) + [str(cfg_dir / f"{n}.yaml") for n in names]
                launch(repo, argv, f"đánh giá {', '.join(names)} ({sets or 'đủ mẫu'})", {"SET": sets})
                st.success("Đã xếp hàng. Theo dõi ở phần trên.")
    else:
        data = sorted((repo / "sft").glob("*.jsonl"))
        if not data:
            st.info("Chưa có dữ liệu SFT: chạy `.venv/bin/python scripts/build_sft.py` trước.")
            return
        src = st.selectbox("Dữ liệu", data, format_func=lambda p: p.name)
        out = st.text_input("Thư mục mô hình ra", value=f"models/qwen3-4b-sft-{src.stem.replace('train', '').strip('_') or 'new'}")
        extra = st.text_input("Đối số thêm cho train_sft.py", value="", placeholder="--epochs 0.5 --lr 1e-4")
        st.caption("Khoảng 2 giờ cho 1 epoch 1.000 mẫu, giữ khoảng 13 GB GPU suốt thời gian đó.")
        if (repo / out / "config.json").exists():
            st.warning(f"`{out}` đã có, việc này sẽ ghi đè.")
        if st.button("Xếp hàng huấn luyện", type="primary"):
            launch(repo, ["train", str(src.relative_to(repo)), out, *extra.split()], f"huấn luyện {out} từ {src.name}", {})
            st.success("Đã xếp hàng. Theo dõi ở phần trên.")


def render(repo: Path, roots: list[Path], colors: list[str], load_config) -> None:
    auto = st.toggle("Tự làm mới mỗi 10 giây", value=True)
    st.fragment(run_every=10 if auto else None)(_monitor)(repo, roots, colors)
    st.divider()
    _control(repo, load_config)
