"""Tab Kiến trúc tốt nhất: chọn phương án từng tầng bằng độ đo riêng của tầng đó, theo thứ tự phụ thuộc
A (ASR) → C (LLM, prompt) → D (guards), rồi xem các tổ hợp đã ghép chạy đầu-cuối.

ASR không phụ thuộc agent nên xếp hạng bằng độ giữ thực thể, không cần LLM. Agent xếp hạng trên chữ đúng (asr: gold)
để lỗi nghe không lẫn vào. Guards phụ thuộc agent nên chỉ tin được khi đo cùng LLM đã chọn. Số lượt cần chạy là tổng
số phương án các tầng cộng vài lượt ghép, không phải tích. Cột "mất do ASR" kiểm giả định tách tầng: nếu ASR làm mất
gần như nhau với mọi agent thì chọn từng tầng riêng là đủ.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

SUBSETS = ("val_random", "val_app")
IGNORE = ("name", "split", "subset", "limit", "repeats", "workers", "agent.llm.base_url")
LICENSE = {"chunkformer": "CC BY-NC: hỏi BTC trước khi nộp"}


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    return out


def diff(c: dict, base: dict) -> dict:
    """Các khoá cấu hình khác base.yaml, bỏ khoá chỉ chọn mẫu."""
    fc, fb = flat(c), flat(base)
    return {k: v for k, v in fc.items() if not k.startswith(IGNORE) and fb.get(k) != v}


def short(d: dict) -> str:
    """Tóm tắt phần khác gốc cho người đọc: tên mô hình bỏ thư mục, khoá bỏ tiền tố tầng."""
    parts = []
    for k, v in d.items():
        v = v.split("/")[-1] if isinstance(v, str) and "/" in v else v
        key = re.sub(r"^(agent|asr)\.", "", k)
        parts.append(f"{key}={v}")
    return ", ".join(parts) or "như base.yaml"


def arch_key(name: str) -> str:
    return re.sub(r"_val-app$", "", re.sub(r"^\d+_", "", name))


def asr_label(c: dict) -> str:
    a = c.get("asr") or {}
    post = "+".join(p["name"] for p in c.get("asr_post") or [])
    return a.get("name", "?") + (f" ({a['model'].split('/')[-1]})" if a.get("model") else "") + (f" + {post}" if post else "")


def llm_label(c: dict) -> str:
    return ((c.get("agent") or {}).get("llm") or {}).get("model", "?").split("/")[-1]


def paired(last: dict[str, dict], keep) -> dict[str, dict]:
    """Ghép val_random và val_app của cùng kiến trúc: {kiến trúc: {subset: run}}."""
    out: dict[str, dict] = {}
    for name, r in last.items():
        m = r["metrics"]
        if m.get("split") == "train" and m.get("subset") in SUBSETS and keep(r):
            out.setdefault(arch_key(name), {})[m["subset"]] = r
    return out


def config_last(df: pd.DataFrame) -> pd.DataFrame:
    """Số liệu lên trước, tên config xuống cuối: khung hẹp vẫn thấy điểm mà không phải cuộn ngang."""
    return df[[c for c in df.columns if c != "config"] + ["config"]] if "config" in df else df


def mean2(a, b):
    return round((a + b) / 2, 2) if a is not None and b is not None else None


def render(repo: Path, runs: list[dict], load_cfg) -> None:
    base = load_cfg(repo / "configs" / "base.yaml")
    last = {}
    for r in runs:  # mới trước: giữ lượt mới nhất mỗi config
        last.setdefault(r["metrics"]["name"], r)

    def is_agent(r):
        return (r["config"].get("agent") or {}).get("name") in ("react_json", "native_tools") and r["metrics"].get("score") is not None \
            and r["metrics"].get("repeats", 1) == 1 and r["metrics"].get("n", 0) >= 100

    def on_gold(r):
        return (r["config"].get("asr") or {}).get("name") == "gold"

    # -- tầng A: ASR đo riêng --------------------------------------------------------------------------
    asr_rows = []
    for key, d in paired(last, lambda r: (r["config"].get("agent") or {}).get("name") == "none" and not on_gold(r)).items():
        cfg = next(iter(d.values()))["config"]
        row = {"phương án": asr_label(cfg), "config": key}
        for s in SUBSETS:
            a = (d[s]["metrics"].get("asr") or {}) if s in d else {}
            row[f"giữ thực thể {s}"] = a.get("entity_recall_exact")
            row[f"WER {s}"] = a.get("wer_num", a.get("wer"))
        r0 = d.get("val_random") or next(iter(d.values()))
        row["giây/mẫu"] = round(r0["metrics"].get("asr_total_sec", 0) / max(1, r0["metrics"]["n"]), 2)
        row["ghi chú"] = LICENSE.get((cfg.get("asr") or {}).get("name"), "")
        asr_rows.append(row)
    asr_df = pd.DataFrame(asr_rows)
    if len(asr_df):
        asr_df = asr_df.sort_values(["giữ thực thể val_random", "WER val_random"], ascending=[False, True], na_position="last")

    # -- tầng C, D: agent đo trên chữ đúng ---------------------------------------------------------------
    llm_rows, guard_rows = [], []
    for key, d in paired(last, lambda r: is_agent(r) and on_gold(r)).items():
        cfg = next(iter(d.values()))["config"]
        dd = diff(cfg, base)
        sc = {s: d[s]["metrics"]["score"] if s in d else None for s in SUBSETS}
        row = {"phương án": short(dd), "config": key, "val_random": sc["val_random"], "val_app": sc["val_app"],
               "trung bình": mean2(sc["val_random"], sc["val_app"]), "LLM": llm_label(cfg),
               "p95 giây": (d.get("val_random") or next(iter(d.values())))["metrics"].get("sec_p95")}
        (guard_rows if dd and all(k.startswith("guards.") for k in dd) else llm_rows).append(row)

    def rank(rows):
        df = pd.DataFrame(rows)
        return df.sort_values(["trung bình", "val_random"], ascending=False, na_position="last") if len(df) else df

    llm_df, guard_df = rank(llm_rows), rank(guard_rows)

    # -- ghép đầu-cuối ------------------------------------------------------------------------------------
    gold_by = {}  # cùng agent và guards, chỉ khác ASR → điểm trên chữ đúng
    for name, r in last.items():
        if is_agent(r) and on_gold(r) and r["metrics"].get("subset") in SUBSETS:
            k = json.dumps({x: v for x, v in diff(r["config"], base).items() if not x.startswith(("asr", "asr_post"))}, sort_keys=True)
            gold_by[(k, r["metrics"]["subset"])] = r["metrics"]["score"]
    e2e_rows = []
    for key, d in paired(last, lambda r: is_agent(r) and not on_gold(r)).items():
        cfg = next(iter(d.values()))["config"]
        k = json.dumps({x: v for x, v in diff(cfg, base).items() if not x.startswith(("asr", "asr_post"))}, sort_keys=True)
        row = {"ASR": asr_label(cfg), "LLM": llm_label(cfg), "config": key}
        for s in SUBSETS:
            row[s] = d[s]["metrics"]["score"] if s in d else None
            g = gold_by.get((k, s))
            row[f"mất do ASR {s}"] = round(row[s] - g, 1) if row[s] is not None and g is not None else None
        row["trung bình"] = mean2(row["val_random"], row["val_app"])
        e2e_rows.append(row)
    e2e_df = pd.DataFrame(e2e_rows)
    if len(e2e_df):
        e2e_df = e2e_df.sort_values(["trung bình", "val_random"], ascending=False, na_position="last")

    # ASR chọn bằng điểm ghép khi đã có: độ giữ thực thể chỉ để sàng lọc (nó không thấy chữ hoa, dấu câu).
    e2e_by_asr = e2e_df.groupby("ASR")["val_random"].max().to_dict() if len(e2e_df) else {}
    if len(asr_df):
        asr_df.insert(2, "đầu-cuối val_random", asr_df["phương án"].map(e2e_by_asr))
    best_llm = llm_df.iloc[0] if len(llm_df) else None
    best_e2e = e2e_df.iloc[0] if len(e2e_df) else None

    def score_pair(a, b):
        return f"**{a:.1f}** val_random · " + (f"**{b:.1f}** val_app" if pd.notna(b) else "val_app chưa đo")

    # -- đầu trang: tổ hợp đang chọn ----------------------------------------------------------------------
    st.caption("Mỗi tầng sàng lọc bằng độ đo rẻ của riêng nó, theo thứ tự phụ thuộc; vài phương án đứng đầu mới chạy ghép, "
               "và điểm ghép quyết định. Điểm là F1 nhân đúng state_hash, thang 0-100. val_random: app đã gặp khi học. "
               "val_app: 10 app giữ lại, gần với app mới của public và private.")
    with st.container(border=True):
        if best_e2e is not None:
            st.markdown(f"#### Tổ hợp tốt nhất đã chạy ghép\n`{best_e2e['ASR']}` → `{best_e2e['LLM']}` → guards như base.yaml\n\n"
                        + score_pair(best_e2e["val_random"], best_e2e["val_app"]))
        else:
            st.markdown("#### Chưa có lượt ghép nào\nChạy một config đầu-cuối (asr thật và agent react_json) để có điểm gần điểm thi.")

    rows = []
    if len(asr_df):
        pick = asr_df.dropna(subset=["đầu-cuối val_random"]).sort_values("đầu-cuối val_random", ascending=False)
        a = (pick if len(pick) else asr_df).iloc[0]
        rows.append({"tầng": "A · nghe (ASR)", "đang chọn": a["phương án"],
                     "vì sao": (f"điểm ghép cao nhất ({a['đầu-cuối val_random']:.1f})" if len(pick) else
                                f"giữ thực thể cao nhất ({a['giữ thực thể val_random']:.1f}%), chưa chạy ghép"),
                     "lưu ý": a["ghi chú"]})
    if best_llm is not None:
        rows.append({"tầng": "C · hiểu và gọi tool (LLM, prompt)", "đang chọn": best_llm["phương án"],
                     "vì sao": f"cao nhất trên chữ đúng: {best_llm['val_random']:.1f} / "
                               + (f"{best_llm['val_app']:.1f}" if pd.notna(best_llm["val_app"]) else "val_app chưa đo"),
                     "lưu ý": "val_app còn thấp hơn val_random nhiều: yếu với app mới" if pd.notna(best_llm["val_app"])
                              and best_llm["val_random"] - best_llm["val_app"] > 10 else ""})
    g = guard_df.iloc[0] if len(guard_df) else None
    rows.append({"tầng": "D · chặn call sai (guards)", "đang chọn": "như base.yaml",
                 "vì sao": "mặc định" if g is None else f"tốt nhất đã thử: {g['phương án']} ({g['val_random']:.1f}, với {g['LLM']})",
                 "lưu ý": "" if g is None or best_llm is None or g["LLM"] == best_llm["LLM"]
                          else f"guards phụ thuộc LLM: cần đo lại với {best_llm['LLM']} trước khi bật"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # -- chi tiết -------------------------------------------------------------------------------------------
    pct = st.column_config.NumberColumn(format="%.1f")
    st.subheader("Ghép đầu-cuối: nghe audio thật rồi gọi tool")
    st.caption("Mất do ASR = điểm ghép trừ điểm của cùng agent trên chữ đúng. Gần như nhau giữa các agent thì hai tầng gần độc lập "
               "và chọn riêng từng tầng là đủ; khác nhiều thì hai tầng tương tác, phải thử tổ hợp.")
    if len(e2e_df):
        st.dataframe(config_last(e2e_df), hide_index=True, width="stretch",
                     column_config={c: pct for c in e2e_df.columns if c in SUBSETS or c.startswith(("mất", "trung"))})
    else:
        st.info("Chưa có lượt ghép nào.")

    st.subheader("A · ASR một mình")
    st.caption("agent: none, không cần LLM. Giữ thực thể: tên riêng, số trong tham số gold nghe ra đúng cả dấu. Chỉ để sàng lọc: "
               "nó bỏ qua chữ hoa và dấu câu, mà agent đã SFT trên chữ có hoa, có dấu câu.")
    if len(asr_df):
        st.dataframe(config_last(asr_df), hide_index=True, width="stretch",
                     column_config={c: pct for c in asr_df.columns if c.startswith(("giữ", "WER", "đầu"))})
    else:
        st.info("Chưa có lượt ASR một mình.")

    st.subheader("C · LLM và prompt trên chữ đúng")
    st.caption("asr: gold, nên lỗi nghe không lẫn vào. Cột đầu ghi phần khác base.yaml. Xếp theo trung bình hai tập.")
    if len(llm_df):
        st.dataframe(config_last(llm_df), hide_index=True, width="stretch", column_config={c: pct for c in SUBSETS + ("trung bình",)})

    if len(guard_df):
        st.subheader("D · Guards: chỉ đổi middleware")
        st.caption("Cột LLM ghi mô hình lúc đo: guard có ích với mô hình chỉ dùng prompt có thể thừa với mô hình đã SFT.")
        st.dataframe(config_last(guard_df), hide_index=True, width="stretch", column_config={c: pct for c in SUBSETS + ("trung bình",)})
