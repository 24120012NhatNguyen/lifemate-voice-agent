"""Cặp audio–lời để fine-tune ASR: lượt cuối và mọi lượt user trong lịch sử có `audio_text`.

Chỉ lấy mẫu thuộc `train_rest` (splits.group), nên `val_random` và `val_app` không lọt vào huấn luyện và các số đo
ASR cũ vẫn so được. Trong `train_rest`, khoảng 1/12 số mẫu (băm id với muối riêng) để làm tập dev chọn epoch.
"""
from __future__ import annotations

import hashlib

from .. import data, splits
from .metrics import norm


def _is_dev(sample_id: str) -> bool:
    return int(hashlib.sha1(f"asr-dev:{sample_id}".encode()).hexdigest(), 16) % 12 == 0


def utterances(part: str = "train", groups: tuple[str, ...] = ("train_rest",)) -> list[dict]:
    """part: "train" | "dev" | "all". Mỗi phần tử: {"key", "path", "text", "group"}."""
    out = []
    for s in data.samples("train"):
        g = splits.group(s)
        if g not in groups:
            continue
        if part != "all" and _is_dev(s["id"]) != (part == "dev"):
            continue
        users = [t for t in s.get("history", []) if t.get("role") == "user" and t.get("audio")]
        for turn, path in data.audio_paths("train", s).items():
            text = s.get("audio_text") if turn == "final" else users[int(turn[1:]) - 1].get("audio_text")
            if text:
                out.append({"key": f"{s['id']}/{turn}", "path": str(path), "text": text, "group": g})
    return out


def ctc_text(text: str) -> str:
    """Nhãn cho mô hình CTC chỉ có chữ thường (ChunkFormer): đọc số thành chữ, bỏ dấu câu, giống `wer_num`."""
    return norm(text, numbers=True)
