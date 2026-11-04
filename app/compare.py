"""Web-app theo dõi và so sánh kiến trúc: tổ hợp tốt nhất chọn theo từng tầng, việc đang chạy trên GPU, lộ trình đề
xuất, dữ liệu và nhãn, rồi đặt các lượt chạy cạnh nhau, xem vết từng mẫu, chạy thử một mẫu. Gộp runs/ và logs/ của mọi
worktree.

    .venv/bin/streamlit run app/compare.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "app"))

import best as best_arch  # noqa: E402
import dataset  # noqa: E402
import live  # noqa: E402
import roadmap  # noqa: E402
from lifemate import config as cfgmod  # noqa: E402
from lifemate import data, splits  # noqa: E402

st.set_page_config(page_title="LifeMate · theo dõi và so sánh", layout="wide")

# Bảng màu đã kiểm (dataviz skill): thứ tự cố định, màu đi theo run, không theo hạng.
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SERIES = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if DARK else
          ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
SEQ = ["#0d366b", "#86b6ef"] if DARK else ["#cde2fb", "#104281"]
MUTED = "#c3c2b7" if DARK else "#52514e"


# -- nạp các lượt chạy ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_runs(roots: tuple[str, ...], stamp: float) -> list[dict]:
    """Lượt chạy của mọi thư mục runs/ trong roots, mới trước; cùng tên thư mục (bản sao giữa worktree) chỉ lấy một."""
    found = {}
    for root in roots:
        for d in Path(root).glob("*/metrics.json"):
            found.setdefault(d.parent.name, d)
    out = []
    for name in sorted(found, reverse=True):
        run = found[name].parent
        m = json.loads(found[name].read_text(encoding="utf-8"))
        c = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        out.append({"dir": str(run), "label": f"{m['name']} · {run.name[6:8]}/{run.name[4:6]} {run.name[9:11]}:{run.name[11:13]}", "metrics": m, "config": c})
    return out


@st.cache_data(show_spinner=False)
def load_samples(run_dir: str) -> pd.DataFrame:
    rows = [json.loads(x) for x in (Path(run_dir) / "samples.jsonl").open(encoding="utf-8")]
    return pd.DataFrame(rows)


def stamp(roots: list[Path]) -> float:
    return max((p.stat().st_mtime for r in roots for p in r.glob("*/metrics.json")), default=0.0)


WORKTREES = live.worktrees(REPO)
with st.sidebar:
    every_wt = st.checkbox("Gộp runs/ của mọi worktree", value=True,
                           help="\n".join(f"- {w.name}" for w in WORKTREES))
RUN_DIRS = [w / "runs" for w in (WORKTREES if every_wt else [REPO])]
runs = load_runs(tuple(map(str, RUN_DIRS)), stamp(RUN_DIRS))

st.title("LifeMate · theo dõi và so sánh kiến trúc")
(t_best, t_live, t_road, t_over, t_arch, t_err, t_asr, t_sample, t_data, t_check, t_try) = st.tabs(
    ["Kiến trúc tốt nhất", "Đang chạy", "Lộ trình", "So sánh lượt chạy", "Cấu hình", "Lỗi", "ASR", "Từng mẫu", "Dữ liệu",
     "Checklist", "Chạy thử"])

with t_best:
    best_arch.render(REPO, runs, cfgmod.load)
with t_live:
    live.render(REPO, WORKTREES, SERIES, cfgmod.load)
with t_road:
    roadmap.render(REPO, runs, SERIES, MUTED)
with t_data:
    dataset.render()

if not runs:
    with t_over:
        st.info("Chưa có lượt chạy nào trong runs/. Chạy: `.venv/bin/lifemate run configs/experiments/01_oracle.yaml`")
    st.stop()

with st.sidebar:
    st.header("Chọn lượt chạy")
    splits_avail = sorted({r["metrics"]["split"] for r in runs})
    sp = st.selectbox("Split", splits_avail, index=splits_avail.index("train") if "train" in splits_avail else 0)
    latest_only = st.checkbox("Chỉ lần chạy mới nhất của mỗi config", value=True)
    pool = [r for r in runs if r["metrics"]["split"] == sp]
    if latest_only:
        seen, keep = set(), []
        for r in pool:
            if r["metrics"]["name"] not in seen:
                seen.add(r["metrics"]["name"])
                keep.append(r)
        pool = keep
    labels = [r["label"] for r in pool]
    # mặc định: các lượt agent có điểm, cùng tập con và cùng số mẫu phổ biến nhất, sắp theo tên config
    scored = [r for r in pool if r["metrics"].get("score") is not None and r["metrics"].get("repeats", 1) == 1]
    if scored:
        key = Counter((r["metrics"]["subset"], r["metrics"]["n"]) for r in scored).most_common(1)[0][0]
        default = [r["label"] for r in sorted(scored, key=lambda r: r["metrics"]["name"])
                   if (r["metrics"]["subset"], r["metrics"]["n"]) == key][:8]
    else:
        default = labels[:6]
    chosen = st.multiselect("Lượt chạy (tối đa 8)", labels, default=default, max_selections=8)
    sel = [r for r in pool if r["label"] in chosen]
    sel.sort(key=lambda r: chosen.index(r["label"]))
    st.caption("Màu đi theo thứ tự chọn ở đây và giữ nguyên ở mọi biểu đồ.")

COLOR = {r["label"]: SERIES[i] for i, r in enumerate(sel)}
color_scale = alt.Scale(domain=list(COLOR), range=list(COLOR.values()))


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    return out


def arch_line(c: dict) -> str:
    a, ag, g = c.get("asr", {}), c.get("agent", {}), c.get("guards", {})
    asr = a.get("name", "?") + (f"({a['model'].split('/')[-1]})" if a.get("model") else "")
    post = "+".join(p["name"] for p in c.get("asr_post") or []) or "—"
    llm = ag.get("llm", {}).get("model", "").split("/")[-1] if ag.get("name") in ("react_json", "native_tools") else ""
    agent = ag.get("name", "?") + (f"({llm})" if llm else "")
    guards = ", ".join(f"{k}={v}" for k, v in g.items() if k in ("id_guard", "dup_guard", "validate"))
    return f"A: {asr} → hậu xử lý: {post} → C: {agent} → D: {guards}"


# -- 1. So sánh lượt chạy ------------------------------------------------------------------------------------
with t_over:
    if not sel:
        st.info("Chọn ít nhất một lượt chạy ở thanh bên.")
    else:
        cols = ["score", "hash_acc", "f1_when_hash_ok", "n", "subset", "sec_p50", "sec_p95", "llm_calls_mean",
                "prompt_tokens_mean", "calls_mean", "rejected_calls", "guard_blocks", "stability", "run_errors"]
        tbl = pd.DataFrame([{"lượt chạy": r["label"], **{k: r["metrics"].get(k) for k in cols},
                             "WER": (r["metrics"].get("asr") or {}).get("wer"), "kiến trúc": arch_line(r["config"])} for r in sel])
        with_score = [r["label"] for r in sel if r["metrics"].get("score") is not None] or [r["label"] for r in sel]
        base = st.selectbox("So với lượt chạy gốc", with_score, index=0)
        b = tbl.set_index("lượt chạy").loc[base, "score"]
        if pd.notna(b):
            tbl.insert(2, "Δ điểm", (tbl["score"] - b).round(2))
        st.dataframe(tbl, hide_index=True, width="stretch")
        scored = tbl.dropna(subset=["score"])
        if len(scored):
            bars = alt.Chart(scored).mark_bar(cornerRadiusEnd=4, size=20).encode(
                y=alt.Y("lượt chạy:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                x=alt.X("score:Q", title="Điểm (F1 × đúng hash), %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("lượt chạy:N", scale=color_scale, legend=None),
                tooltip=["lượt chạy", "score", "hash_acc", "f1_when_hash_ok", "n", "subset"])
            text = bars.mark_text(align="left", dx=4, color=MUTED).encode(text=alt.Text("score:Q", format=".2f"), color=alt.value(MUTED))
            st.altair_chart((bars + text).properties(height=36 * len(scored) + 40), width="stretch")
        st.caption("Điểm chỉ so được giữa các lượt chạy cùng subset. val_app gần với app chưa thấy ở private nhất.")

# -- 2. Cấu hình --------------------------------------------------------------------------------------
with t_arch:
    if sel:
        for r in sel:
            st.markdown(f"<span style='color:{COLOR[r['label']]}'>■</span> **{r['label']}** — `{arch_line(r['config'])}`",
                        unsafe_allow_html=True)
        keys = ["asr", "asr_post", "agent", "guards", "subset", "limit", "repeats"]
        mats = {r["label"]: flat({k: r["config"].get(k) for k in keys}) for r in sel}
        df = pd.DataFrame(mats).drop(index=[i for i in pd.DataFrame(mats).index if i.startswith("agent.llm.base_url")], errors="ignore")
        only_diff = st.checkbox("Chỉ hiện dòng khác nhau giữa các lượt chạy", value=True)
        diff = df.astype(str).nunique(axis=1) > 1
        view = df[diff] if only_diff and diff.any() else df

        def hl(row):
            return ["background-color: rgba(42,120,214,.15)" if diff.get(row.name, False) else "" for _ in row]

        st.dataframe(view.astype(str).style.apply(hl, axis=1), width="stretch", height=min(700, 36 * len(view) + 40))

# -- 3. Lỗi --------------------------------------------------------------------------------------------
with t_err:
    scored = [r for r in sel if "errors" in r["metrics"]]
    if scored:
        long = pd.DataFrame([{"lượt chạy": r["label"], "lỗi": k, "số mẫu": v}
                             for r in scored for k, v in r["metrics"]["errors"].items() if k != "đúng"])
        if len(long):
            order = long.groupby("lỗi")["số mẫu"].sum().sort_values(ascending=False).index.tolist()
            xs = [r["label"] for r in scored if r["label"] in set(long["lượt chạy"])]
            heat = alt.Chart(long).mark_rect(stroke="#fcfcfb" if not DARK else "#1a1a19", strokeWidth=2).encode(
                x=alt.X("lượt chạy:N", sort=xs, title=None,
                        axis=alt.Axis(labelAngle=-25, labelLimit=200, labelExpr="split(datum.label, ' · ')[0]")),
                y=alt.Y("lỗi:N", sort=order, title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
                color=alt.Color("số mẫu:Q", scale=alt.Scale(range=SEQ), legend=alt.Legend(title="số mẫu")),
                tooltip=["lượt chạy", "lỗi", "số mẫu"])
            vmax = long["số mẫu"].max()
            # ô đậm (cuối thang) dùng chữ tương phản: sáng trên nền đậm ở light, tối trên nền sáng ở dark
            hi, lo = ("#0b0b0b", "#f0efec") if DARK else ("#ffffff", "#0b0b0b")
            txt = alt.Chart(long).mark_text(fontSize=12).encode(
                x=alt.X("lượt chạy:N", sort=xs), y=alt.Y("lỗi:N", sort=order), text="số mẫu:Q",
                color=alt.condition(alt.datum["số mẫu"] > vmax * 0.5, alt.value(hi), alt.value(lo)))
            st.subheader("Nhãn lỗi chính của mỗi mẫu chưa đạt 1 điểm")
            st.altair_chart((heat + txt).properties(height=36 * len(order) + 90), width="stretch")
            st.dataframe(long.pivot_table(index="lỗi", columns="lượt chạy", values="số mẫu", fill_value=0).loc[order], width="stretch")
        st.subheader("Kiểu phản ứng: gold → dự đoán")
        conf = pd.DataFrame([{"lượt chạy": r["label"], "gold → dự đoán": k, "số mẫu": v}
                             for r in scored for k, v in r["metrics"]["response_confusion"].items()])
        st.dataframe(conf.pivot_table(index="gold → dự đoán", columns="lượt chạy", values="số mẫu", fill_value=0), width="stretch")

# -- 4. ASR --------------------------------------------------------------------------------------------
with t_asr:
    asr_runs = [r for r in runs if r["metrics"].get("asr") and r["metrics"]["split"] == sp
                and (r["config"].get("asr") or {}).get("name") != "gold"]
    if not asr_runs:
        st.info("Chưa có lượt chạy ASR thật trên train. Chạy vd. `lifemate run configs/experiments/20_asr_whisper-turbo.yaml`.")
    else:
        latest = {}
        for r in asr_runs:
            k = json.dumps([r["config"].get("asr"), r["config"].get("asr_post"), r["metrics"]["subset"]], sort_keys=True)
            latest.setdefault(k, r)
        tbl = pd.DataFrame([{"ASR": r["label"], "subset": r["metrics"]["subset"], **r["metrics"]["asr"],
                             "giây ASR / mẫu": round(r["metrics"]["asr_total_sec"] / max(1, r["metrics"]["n"]), 3)}
                            for r in latest.values()])
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("entity_recall_folded: chuỗi trong tham số gold nghe ra đúng khi bỏ dấu (đủ cho tool tìm kiếm). "
                   "entity_recall_exact: đúng cả dấu (cần cho tham số khớp tuyệt đối).")
        mcols = [c for c in ("wer", "wer_num", "cer") if c in tbl]
        melt = tbl.melt(id_vars=["ASR"], value_vars=mcols, var_name="độ đo", value_name="%")
        ch = alt.Chart(melt).mark_bar(cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("ASR:N", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)), x=alt.X("%:Q", title="Tỉ lệ lỗi, % (thấp hơn là tốt)"),
            yOffset=alt.YOffset("độ đo:N", sort=mcols), color=alt.Color("độ đo:N", scale=alt.Scale(domain=mcols, range=SERIES[:len(mcols)]), legend=alt.Legend(orient="top")),
            tooltip=["ASR", "độ đo", "%"])
        st.altair_chart((ch + ch.mark_text(align="left", dx=4).encode(text=alt.Text("%:Q", format=".1f"), color=alt.value(MUTED)))
                        .properties(height=24 * len(mcols) * len(tbl) + 60), width="stretch")
        st.subheader("Nghe thử và so bản chép từng mẫu")
        ids = load_samples(asr_runs[0]["dir"])["id"].tolist()
        sid = st.selectbox("Mẫu", ids, key="asr_sid")
        s = data.by_id(sp)[sid]
        st.audio(str(data.audio_paths(sp, s)["final"]))
        rows = [{"nguồn": "audio_text (gold)", "bản chép": s.get("audio_text", "—")}]
        for r in latest.values():
            d = load_samples(r["dir"]).set_index("id")
            if sid in d.index:
                rows.append({"nguồn": r["label"], "bản chép": d.loc[sid, "transcript"], "WER": d.loc[sid].get("wer")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# -- 5. Từng mẫu ---------------------------------------------------------------------------------------
with t_sample:
    agent_runs = [r for r in sel if "score" in r["metrics"]]
    if len(agent_runs) >= 1:
        c1, c2, c3 = st.columns([2, 2, 2])
        la = c1.selectbox("Lượt chạy A", [r["label"] for r in agent_runs], index=0)
        lb = c2.selectbox("Lượt chạy B", [r["label"] for r in agent_runs], index=min(1, len(agent_runs) - 1))
        mode = c3.selectbox("Lọc", ["A khác B", "A sai", "B sai", "tất cả"])
        A = load_samples(next(r["dir"] for r in agent_runs if r["label"] == la)).set_index("id")
        B = load_samples(next(r["dir"] for r in agent_runs if r["label"] == lb)).set_index("id")
        common = A.index.intersection(B.index)
        comp = pd.DataFrame({"A": A.loc[common, "score"], "B": B.loc[common, "score"],
                             "lỗi A": A.loc[common, "error"], "lỗi B": B.loc[common, "error"]})
        if mode == "A khác B":
            comp = comp[comp["A"] != comp["B"]]
        elif mode == "A sai":
            comp = comp[comp["A"] < 1]
        elif mode == "B sai":
            comp = comp[comp["B"] < 1]
        st.caption(f"{len(comp)} mẫu · A tốt hơn ở {(comp['A'] > comp['B']).sum()} · B tốt hơn ở {(comp['B'] > comp['A']).sum()}")
        st.dataframe(comp.round(3), width="stretch", height=220)
        if len(comp):
            sid = st.selectbox("Xem mẫu", comp.index.tolist())
            s = data.by_id(sp)[sid]
            st.audio(str(data.audio_paths(sp, s)["final"]))
            st.markdown(f"**Gold:** {s.get('audio_text', '—')}  \n**Thời điểm:** {s['context']['current_time']} · "
                        f"**Lịch sử:** {len(s.get('history', []))} lượt · **Nhóm:** {splits.group(s) if 'label' in s else '—'}")
            with st.expander("Lịch sử hội thoại"):
                for t in s.get("history", []):
                    st.write(f"**{t['role']}**: {t.get('audio_text') or t.get('text', '')}")
                    for c in t.get("tool_calls") or []:
                        st.code(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}", language=None)
            st.markdown("**Call gold**")
            st.code("\n".join(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}" for c in s["label"]["tool_calls"]) or "[] (đứng yên)", language=None)
            ca, cb = st.columns(2)
            for col, lab, D in ((ca, la, A), (cb, lb, B)):
                row = D.loc[sid]
                with col:
                    st.markdown(f"<span style='color:{COLOR.get(lab, MUTED)}'>■</span> **{lab}** — điểm {row['score']:.2f} · "
                                f"hash {'đúng' if row['hash_ok'] else 'sai'} · {row['error']}", unsafe_allow_html=True)
                    st.caption(f"Bản chép: {row.get('transcript', '')}")
                    for i, stp in enumerate(row["steps"] or []):
                        head = f"{i + 1}. {stp.get('action')}"
                        if stp.get("name"):
                            head += f" · {stp['name']} {json.dumps(stp.get('arguments', {}), ensure_ascii=False)}"
                        with st.expander(head[:160]):
                            if stp.get("thought"):
                                st.write(f"Suy nghĩ: {stp['thought']}")
                            if "observation" in stp:
                                st.json(stp["observation"], expanded=False)
                            if stp.get("raw") and stp.get("action") == "parse_error":
                                st.code(stp["raw"], language=None)
                            if "sec" in stp:
                                st.caption(f"{stp['sec']:.2f}s · {stp.get('prompt_tokens', 0)} token vào")
                    blocked = [g for g in row["guard_log"] if g["decision"] != "chạy"]
                    if blocked:
                        st.warning("Middleware: " + "; ".join(f"{g['name']}: {g['decision']}" for g in blocked))

# -- 6. Checklist --------------------------------------------------------------------------------------
with t_check:
    from lifemate.registry import available

    st.subheader("Phương án theo tầng")
    used = {}
    for r in runs:
        c = r["config"]
        for stage, spec in (("asr", c.get("asr")), ("agent", c.get("agent"))):
            if spec:
                used.setdefault((stage, spec.get("name")), []).append(r)
        for p in c.get("asr_post") or []:
            used.setdefault(("asr_post", p["name"]), []).append(r)
    rows = []
    for stage, entries in available().items():
        for e in entries:
            rs = used.get((stage, e.name), [])
            best = max((r["metrics"].get("score") for r in rs if r["metrics"].get("score") is not None), default=None)
            wer = min(((r["metrics"].get("asr") or {}).get("wer") for r in rs if (r["metrics"].get("asr") or {}).get("wer") is not None), default=None)
            rows.append({"tầng": stage, "phương án": e.name, "trạng thái": e.status, "đã chạy": "✅" if rs else "⬜",
                         "số lượt": len(rs), "điểm tốt nhất": best, "WER tốt nhất": wer, "mô tả": e.doc})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Thí nghiệm trong configs/experiments")
    rows = []
    for p in sorted((REPO / "configs" / "experiments").glob("*.yaml")):
        first = next((ln[1:].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("#")), "")
        rs = [r for r in runs if r["metrics"]["name"] == p.stem]
        m = rs[0]["metrics"] if rs else {}
        rows.append({"": "✅" if rs else "⬜", "config": p.stem, "mục đích": first, "điểm": m.get("score"),
                     "WER": (m.get("asr") or {}).get("wer"), "n": m.get("n"), "lần chạy": len(rs),
                     "lệnh": f".venv/bin/lifemate run configs/experiments/{p.name}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Thêm một phương án: viết lớp mới với @register(tầng, tên) rồi tạo một file config dùng tên đó. Xem docs/BASELINE.md.")

# -- 7. Chạy thử ---------------------------------------------------------------------------------------
with t_try:
    st.write("Chạy một mẫu với một config bất kỳ (agent LLM cần server ở `agent.llm.base_url`).")
    cfgs = sorted((REPO / "configs" / "experiments").glob("*.yaml"))
    cpath = st.selectbox("Config", cfgs, format_func=lambda p: p.stem)
    sid = st.text_input("Mã mẫu", value="train_0001")
    extra = st.text_input("Ghi đè (cách nhau bởi dấu ;)", value="", placeholder="agent.max_steps=6; guards.id_guard=block")
    if st.button("Chạy", type="primary"):
        from lifemate.asr.base import run_asr
        from lifemate.registry import build
        from lifemate.runner import run_one

        c = cfgmod.load(cpath, [x.strip() for x in extra.split(";") if x.strip()])
        split = c.get("split", "train")
        s = data.by_id(split).get(sid)
        if s is None:
            st.error(f"Không có {sid} trong {split}")
        else:
            with st.spinner("Đang chạy…"):
                tr = run_asr(split, [s], c, log=lambda *_: None)[sid]
                row = run_one(c, build("agent", c["agent"]), split, s, tr)
            st.caption(arch_line(c))
            st.write(f"Bản chép: {tr['final']}")
            if "score" in row:
                st.metric("Điểm mẫu", f"{row['score']:.2f}", help=f"hash {'đúng' if row['hash_ok'] else 'sai'} · {row['error']}")
            if row.get("error_run"):
                st.error(row["error_run"])
            st.json({"tool_calls": row["tool_calls"], "steps": row["steps"], "guard_log": row["guard_log"]}, expanded=1)
