"""Tab Lộ trình: các nấc và ý tưởng của docs/PROPOSAL.md với trạng thái, cổng, bằng chứng tự lấy từ runs/,
khoảng cách app đã thấy và chưa thấy, và bảng đề xuất tiếp theo sửa được (docs/roadmap.yaml)."""
from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ICON = {"xong": "🟢", "đang làm": "🟡", "chưa làm": "⚪", "bỏ": "⛔"}
STATUSES = list(ICON)


def latest(runs: list[dict]) -> dict[str, dict]:
    """Lượt mới nhất của mỗi config (runs đã sắp mới trước)."""
    out = {}
    for r in runs:
        out.setdefault(r["metrics"]["name"], r)
    return out


def evidence(patterns: list[str], last: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in sorted(last.items()):
        if any(fnmatch(name, p) for p in patterns):
            m = r["metrics"]
            rows.append({"config": name, "tập": m.get("subset"), "n": m.get("n"), "điểm": m.get("score"),
                         "đúng hash": m.get("hash_acc"), "WER": (m.get("asr") or {}).get("wer_num", (m.get("asr") or {}).get("wer"))
                         if (r["config"].get("asr") or {}).get("name") != "gold" else None})
    return pd.DataFrame(rows)


def gap_pairs(last: dict[str, dict]) -> pd.DataFrame:
    """Cặp config cùng kiến trúc trên val_random và val_app, ghép theo tên bỏ số thứ tự và hậu tố _val-app."""
    by_key = {}
    for name, r in last.items():
        if r["metrics"].get("score") is None or r["metrics"].get("subset") not in ("val_random", "val_app"):
            continue
        key = re.sub(r"_val-app$", "", re.sub(r"^\d+_", "", name))
        by_key.setdefault(key, {})[r["metrics"]["subset"]] = (name, r["metrics"]["score"])
    rows = []
    for key, d in by_key.items():
        if len(d) == 2:
            for subset, (name, score) in d.items():
                rows.append({"kiến trúc": key, "tập": subset, "config": name, "điểm": score})
    return pd.DataFrame(rows)


def render(repo: Path, runs: list[dict], colors: list[str], muted: str) -> None:
    path = repo / "docs" / "roadmap.yaml"
    if not path.exists():
        st.info("Chưa có docs/roadmap.yaml.")
        return
    text = path.read_text(encoding="utf-8")
    rm = yaml.safe_load(text) or {}
    last = latest(runs)

    st.subheader("Câu hỏi chính: điểm trên app đã thấy và app chưa thấy")
    pairs = gap_pairs(last)
    if len(pairs):
        order = (pairs.pivot(index="kiến trúc", columns="tập", values="điểm")
                 .assign(gap=lambda d: d["val_random"] - d["val_app"]).sort_values("gap"))
        scale = alt.Scale(domain=["val_random", "val_app"], range=colors[:2])
        bars = alt.Chart(pairs).mark_bar(cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("kiến trúc:N", sort=order.index.tolist(), title=None, axis=alt.Axis(labelLimit=320)),
            yOffset=alt.YOffset("tập:N", sort=["val_random", "val_app"]),
            x=alt.X("điểm:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=list(range(0, 101, 20))), title="Điểm, %"),
            color=alt.Color("tập:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
            tooltip=["config", "tập", "điểm"])
        labels = bars.mark_text(align="left", dx=4).encode(text=alt.Text("điểm:Q", format=".1f"), color=alt.value(muted))
        st.altair_chart((bars + labels).properties(height=44 * len(order) + 50), width="stretch")
        st.dataframe(order.rename(columns={"gap": "khoảng cách"}).round(1), width="stretch")
        st.caption("Public có 11/36 app không có trong train, ở 65% số mẫu: val_app là con số gần private nhất. "
                   "Cặp được ghép theo tên config bỏ số thứ tự và hậu tố _val-app.")
    else:
        st.caption("Chưa có cặp config nào chạy trên cả val_random và val_app.")

    def section(title: str, items: list[dict], with_gate: bool) -> None:
        st.subheader(title)
        for it in items:
            head = f"{ICON.get(it.get('status'), '⚪')} **{it['id']}. {it['name']}** · {it.get('status', '?')}"
            with st.expander(head, expanded=it.get("status") == "đang làm"):
                if with_gate and it.get("gate"):
                    st.markdown(f"**Cổng:** {it['gate']}")
                if it.get("note"):
                    st.markdown(it["note"])
                ev = evidence(it.get("configs") or [], last)
                if len(ev):
                    st.dataframe(ev, hide_index=True, width="stretch")
                elif it.get("configs"):
                    st.caption("Chưa có lượt chạy khớp " + ", ".join(it["configs"]))

    section("Các nấc (PROPOSAL mục 6)", rm.get("rungs", []), True)
    section("Sáu ý tưởng khác biệt (PROPOSAL mục 5)", rm.get("ideas", []), False)

    st.subheader("Đề xuất tiếp theo")
    df = pd.DataFrame(rm.get("proposals", []), columns=["priority", "name", "why", "cost", "status"])
    edited = st.data_editor(
        df.sort_values("priority"), hide_index=True, width="stretch", num_rows="dynamic", key="proposals",
        column_config={"priority": st.column_config.NumberColumn("ưu tiên", step=1, width="small"),
                       "name": st.column_config.TextColumn("đề xuất", width="medium"),
                       "why": st.column_config.TextColumn("vì sao", width="large"),
                       "cost": st.column_config.TextColumn("chi phí", width="small"),
                       "status": st.column_config.SelectboxColumn("trạng thái", options=STATUSES, width="small")})
    if st.button("Lưu vào docs/roadmap.yaml"):
        header = "".join(ln + "\n" for ln in text.splitlines() if ln.startswith("#"))
        rows = [{k: v for k, v in r.items() if pd.notna(v) and v != ""} for r in edited.to_dict("records") if r.get("name")]
        for r in rows:
            if "priority" in r:
                r["priority"] = int(r["priority"])
        rm["proposals"] = sorted(rows, key=lambda r: r.get("priority", 99))
        path.write_text(header + yaml.safe_dump(rm, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
        st.success("Đã lưu.")
