"""Tab Dữ liệu: nhãn của BTC nằm ở đâu, nghe từng mẫu cùng bản chép lời và call gold, catalog app train."""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from lifemate import data, splits
from lifemate.splits import HELDOUT_APPS


@st.cache_data(show_spinner=False)
def overview() -> dict:
    root = data.data_root()
    tr = data.samples("train")
    return {"root": str(root), "train": len(tr),
            "turns": sum(1 for s in tr for t in s.get("history", []) if t.get("role") == "user"),
            "wav_train": sum(1 for _ in (root / "train" / "audio").rglob("*.wav"))}


@st.cache_data(show_spinner=False)
def app_table() -> pd.DataFrame:
    tools = Counter(n.split(".")[0] for n in data.tool_specs("train"))
    used = Counter(a for s in data.samples("train") for a in {c["name"].split(".")[0] for c in s["label"]["tool_calls"]})
    installed = Counter(a for s in data.samples("train") for a in s["context"]["installed_apps"])
    return pd.DataFrame([{
        "app": app, "loại": "giữ lại (val_app)" if app in HELDOUT_APPS else "train",
        "số tool": tools[app], "mẫu train gọi": used[app], "mẫu train cài": installed[app],
    } for app in sorted(tools)])


def _calls(calls: list[dict], results: list | None = None) -> str:
    lines = []
    for i, c in enumerate(calls):
        lines.append(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}")
        if results and i < len(results):
            lines.append(f"    → {json.dumps(results[i], ensure_ascii=False)[:300]}")
    return "\n".join(lines) or "[] (đứng yên)"


def render() -> None:
    try:
        ov = overview()
    except FileNotFoundError:
        st.info("Không tìm thấy gói dữ liệu BTC (đặt ở zalo-ai-2026/data/lifemate/ hoặc trỏ LIFEMATE_DATA).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Mẫu train có nhãn", f"{ov['train']:,}".replace(",", "."))
    c2.metric("Lượt user trong lịch sử có chép lời", f"{ov['turns']:,}".replace(",", "."))
    c3.metric("File wav train", f"{ov['wav_train']:,}".replace(",", "."))

    st.subheader("Nhãn nằm ở đâu")
    st.markdown(
        f"Gói dữ liệu: `{ov['root']}`. Mọi nhãn ở **`train/train.jsonl`**, mỗi dòng một mẫu:\n"
        "- `audio_text`: lời của yêu cầu cuối, ứng với `train/audio/<id>/<id>.wav`.\n"
        "- `history[i].audio_text`: lời của từng lượt user trước đó, ứng với `<id>_u1.wav`, `<id>_u2.wav`...\n"
        "- `label.tool_calls`, `label.tool_results`, `label.final_state`: đáp án call, kết quả từng call, state sau cùng.\n\n"
        "`public_test/public_test.jsonl` chỉ có đầu vào. Không có private test trước khi nộp Docker.")

    st.subheader("Xem từng mẫu")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    split = "train"
    c1.caption("Split: train")
    pool = data.samples(split)
    if split == "train":
        grp = c2.selectbox("Tập", ["tất cả", "val_random", "val_app", "train_rest"], key="ds_group")
        if grp != "tất cả":
            pool = [s for s in pool if splits.group(s) == grp]
    apps = sorted({a for s in pool for a in s["context"]["installed_apps"]})
    app = c3.selectbox("App có cài", ["(mọi app)", *apps], key="ds_app")
    if app != "(mọi app)":
        pool = [s for s in pool if app in s["context"]["installed_apps"]]
    q = c4.text_input("Tìm trong lời nói hoặc tên tool", key="ds_q").strip().lower()
    if q:
        pool = [s for s in pool if q in (s.get("audio_text") or "").lower()
                or any(q in c["name"] for c in (s.get("label") or {}).get("tool_calls", []))]
    st.caption(f"{len(pool)} mẫu khớp")
    if not pool:
        return
    s = data.by_id(split)[st.selectbox("Mẫu", [x["id"] for x in pool], key="ds_sid")]
    audio = data.audio_paths(split, s)
    ctx = s["context"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Yêu cầu cuối**")
        st.audio(str(audio["final"]))
        st.markdown(f"> {s.get('audio_text', '(public: không có lời gốc)')}")
        if s.get("history"):
            st.markdown("**Lịch sử hội thoại**")
            k = 0
            for t in s["history"]:
                if t["role"] == "user":
                    k += 1
                    st.markdown(f"🧑 *u{k}:* {t.get('audio_text', '')}")
                    if f"u{k}" in audio:
                        st.audio(str(audio[f"u{k}"]))
                else:
                    st.markdown(f"🤖 {t.get('text', '')}")
                    if t.get("tool_calls"):
                        st.code(_calls(t["tool_calls"]), language=None)
        if "label" in s:
            st.markdown(f"**Call gold** · nhóm `{splits.group(s)}`")
            st.code(_calls(s["label"]["tool_calls"], s["label"].get("tool_results")), language=None)
            with st.expander("final_state (state của các app bị đổi)"):
                st.json(s["label"]["final_state"], expanded=False)
    with right:
        st.markdown(f"**Thời điểm:** {ctx['current_time']}")
        st.markdown("**Người dùng:** " + ", ".join(f"{k}: {v}" for k, v in ctx.get("user_profile", {}).items()))
        st.markdown("**App đã cài**")
        st.dataframe(pd.DataFrame([{"app": a, "tool": ", ".join(t.split(".")[-1] for t in v["tools"]),
                                    "giữ lại": "✓" if a in HELDOUT_APPS else ""}
                                   for a, v in ctx["installed_apps"].items()]), hide_index=True, width="stretch")

    st.subheader("Catalog app: train")
    tbl = app_table()
    kinds = tbl["loại"].value_counts()
    st.caption(" · ".join(f"{k}: {v} app" for k, v in kinds.items()) +
               ". App giữ lại (val_app) vẫn được cài ở nhiều mẫu train_rest nhưng gold không bao giờ gọi.")
    pick = st.multiselect("Lọc theo loại", list(kinds.index), default=list(kinds.index), key="ds_kinds")
    st.dataframe(tbl[tbl["loại"].isin(pick)], hide_index=True, width="stretch")
