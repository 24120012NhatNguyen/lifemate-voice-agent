"""Truy cập gói dữ liệu BTC: mẫu, catalog, audio, và `zaic_tools` riêng của từng split.

`zaic_tools` của train và public_test khác nhau (`utils.py`: LIVE_APPS, FREE_TEXT_KEYS), nên mỗi split nạp
đúng bản của nó dưới một tên module riêng. Dùng nhầm bản là state_hash lệch.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

from .config import REPO


@lru_cache(maxsize=1)
def data_root() -> Path:
    """Thư mục chứa train/ và public_test/: biến LIFEMATE_DATA, hoặc tìm `data/lifemate` ở các thư mục cha."""
    env = os.environ.get("LIFEMATE_DATA")
    if env:
        return Path(env).resolve()
    for base in [Path.cwd(), *Path.cwd().parents, REPO, *REPO.parents]:
        cand = base / "data" / "lifemate"
        if (cand / "VERSION").exists():
            return cand
    raise FileNotFoundError("không thấy data/lifemate; đặt biến LIFEMATE_DATA trỏ tới gói dữ liệu BTC")


@lru_cache(maxsize=None)
def samples(split: str) -> list[dict]:
    path = data_root() / split / f"{split}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@lru_cache(maxsize=None)
def by_id(split: str) -> dict[str, dict]:
    return {s["id"]: s for s in samples(split)}


@lru_cache(maxsize=None)
def catalog(split: str) -> dict:
    return json.loads((data_root() / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def tool_specs(split: str) -> dict[str, dict]:
    return {t["name"]: t for a in catalog(split)["apps"] for t in a["tools"]}


@lru_cache(maxsize=None)
def fuzzy_params(split: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, t in tool_specs(split).items():
        for p, s in t.get("parameters", {}).get("properties", {}).items():
            if s.get("x-match") == "fuzzy":
                out.setdefault(name, set()).add(p)
    return out


def installed_tools(sample: dict) -> list[str]:
    return [t for a in sample["context"]["installed_apps"].values() for t in a["tools"]] + ["sys.ask_missing_info"]


def audio_paths(split: str, sample: dict) -> dict[str, Path]:
    """{"final": lượt cuối, "u1": lượt user 1 trong history, ...}"""
    base = data_root() / split
    out = {"final": base / sample["audio"]}
    k = 0
    for turn in sample.get("history", []):
        if turn.get("role") == "user" and turn.get("audio"):
            k += 1
            out[f"u{k}"] = base / turn["audio"]
    return out


_IMPORT_LOCK = threading.Lock()


@lru_cache(maxsize=None)
def _load_zaic_tools(split: str):
    pkg = data_root() / split / "zaic_tools"
    name = f"zaic_tools_{split}"
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def zaic_tools(split: str):
    """Nạp gói zaic_tools của split dưới tên `zaic_tools_<split>` (một lần, an toàn khi chạy song song)."""
    with _IMPORT_LOCK:
        return _load_zaic_tools(split)


def tool_env(split: str, sample_id: str):
    """Một ToolEnv mới cho một mẫu. Luật thi: mỗi mẫu đúng một lần."""
    return zaic_tools(split).ToolEnv(split, sample_id, root=data_root())


def tool_kind(split: str, name: str) -> str:
    t = tool_specs(split).get(name)
    return t["kind"] if t else "write"
