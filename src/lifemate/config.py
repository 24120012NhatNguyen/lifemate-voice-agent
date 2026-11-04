"""Config thí nghiệm: YAML có `extends` (gộp sâu) và ghi đè từ dòng lệnh `--set a.b=c`."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    base = _load_raw(path.parent / parent) if parent else {}
    base.pop("name", None)
    return deep_merge(base, raw)


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = _load_raw(path)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        *head, last = key.strip().split(".")
        for k in head:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    cfg.setdefault("name", path.stem)
    return cfg


def stage_key(*parts) -> str:
    """Khoá bộ nhớ đệm cho một tầng: cùng config thì dùng lại kết quả (vd. transcript của cùng một ASR)."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
