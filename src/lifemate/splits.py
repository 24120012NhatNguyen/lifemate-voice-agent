"""Chia tập kiểm định (proposal mục 7). Tất định theo id, không cần file.

    all          mọi mẫu của split
    val_app      mẫu train có gold gọi một trong HELDOUT_APPS: đo khả năng đọc schema app chưa thấy
    val_random   20% mẫu còn lại, chọn theo băm id
    train_rest   phần còn lại, dành cho SFT
    ids:a,b,c    danh sách id cụ thể

`limit: N` lấy N mẫu ngẫu nhiên có seed cố định trong tập đã chọn.
"""
from __future__ import annotations

import hashlib
import random

from . import data

# 10 app chỉ có ở train (không có ở public), 301 mẫu, khoảng 19,5%: bỏ chúng khỏi SFT không mất gì cho public.
HELDOUT_APPS = ["appliances", "door_lock", "fitness", "learning", "library", "loyalty", "medication",
                "package_tracking", "subscriptions", "video_meeting"]


def _bucket(sample_id: str) -> int:
    return int(hashlib.sha1(sample_id.encode()).hexdigest(), 16) % 5


def _apps_used(s: dict) -> set[str]:
    return {c["name"].split(".", 1)[0] for c in s.get("label", {}).get("tool_calls", [])}


def group(s: dict) -> str:
    if _apps_used(s) & set(HELDOUT_APPS):
        return "val_app"
    return "val_random" if _bucket(s["id"]) == 0 else "train_rest"


def select(split: str, subset: str = "all", limit: int | None = None, seed: int = 2026) -> list[dict]:
    pool = data.samples(split)
    if subset.startswith("ids:"):
        want = subset[4:].split(",")
        idx = data.by_id(split)
        pool = [idx[i.strip()] for i in want]
    elif subset != "all":
        if split != "train":
            raise ValueError(f"tập con {subset} chỉ có cho train")
        pool = [s for s in pool if group(s) == subset]
    if limit and limit < len(pool):
        pool = sorted(random.Random(seed).sample(pool, limit), key=lambda s: s["id"])
    return pool
