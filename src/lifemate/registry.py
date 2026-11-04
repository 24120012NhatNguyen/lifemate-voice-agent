"""Sổ đăng ký thành phần: mỗi tầng (asr, asr_post, agent) có nhiều phương án, chọn bằng `name` trong config.

Thêm một phương án mới:

    @register("asr", "my_asr")
    class MyASR(ASR):
        \"\"\"Một dòng mô tả, hiện trong `lifemate list` và tab Checklist.\"\"\"
        def __init__(self, model="...", **kw): ...

rồi đặt `asr: {name: my_asr, model: ...}` trong file config.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

STAGES = ("asr", "asr_post", "agent")

# module chứa các phương án, nạp lười để không kéo torch khi chỉ chấm điểm
_MODULES = {
    "asr": ("lifemate.asr.adapters", "lifemate.asr.augment"),
    "asr_post": ("lifemate.asr.post",),
    "agent": ("lifemate.agent.agents",),
}


@dataclass
class Entry:
    stage: str
    name: str
    cls: type
    doc: str
    status: str  # ready | needs_deps | research


_REG: dict[str, dict[str, Entry]] = {s: {} for s in STAGES}


def register(stage: str, name: str, status: str = "ready"):
    def deco(cls):
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        _REG[stage][name] = Entry(stage, name, cls, doc, status)
        cls.registry_name = name
        return cls
    return deco


def _load(stage: str):
    for mod in _MODULES[stage]:
        importlib.import_module(mod)


def build(stage: str, spec: dict | None):
    """spec = {"name": ..., **tham_số}; trả về thể hiện của phương án."""
    spec = dict(spec or {})
    name = spec.pop("name", None)
    if name is None:
        raise ValueError(f"thiếu `name` cho tầng {stage}")
    _load(stage)
    if name not in _REG[stage]:
        raise KeyError(f"{stage}: không có phương án '{name}'. Có: {sorted(_REG[stage])}")
    return _REG[stage][name].cls(**spec)


def available() -> dict[str, list[Entry]]:
    for s in STAGES:
        _load(s)
    return {s: sorted(_REG[s].values(), key=lambda e: e.name) for s in STAGES}
