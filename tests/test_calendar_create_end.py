"""Synthetic unit fixtures; no BTC data required."""
from copy import deepcopy

import pytest

from lifemate import data
from lifemate.guards import DEFAULTS, GuardedEnv


@pytest.fixture
def guard(monkeypatch):
    iso = {"type": "string", "description": "ISO-8601 date and time"}
    spec = {"parameters": {"properties": {"start": iso, "end": iso}}}
    monkeypatch.setattr(data, "tool_specs", lambda _: {
        name: spec for name in ("calendar.create_event", "calendar.update_event", "other.create_event")
    })
    sample = {"context": {"installed_apps": {}, "current_time": "2026-01-01T00:00:00+07:00"}}
    return GuardedEnv(None, "train", sample, calendar_create_end=True)


@pytest.mark.parametrize("ending", [{}, {"end": None}, {"end": ""}])
@pytest.mark.parametrize("start, end", [
    ("2026-01-31T23:30", "2026-02-01T00:30:00+07:00"),
    ("2026-01-01T10:15:00", "2026-01-01T11:15:00+07:00"),
    ("2026-01-01T10:15:20+07:00", "2026-01-01T11:15:20+07:00"),
    ("2026-01-01T10:15:20+00:00", "2026-01-01T11:15:20+00:00"),
])
def test_fill_missing_end_after_normalize(guard, ending, start, end):
    args = {"start": start, **ending}
    before = deepcopy(args)
    result = guard.normalize("calendar.create_event", args)
    assert result["end"] == end
    assert args == before
    assert guard.normalize("calendar.create_event", result) == result


@pytest.mark.parametrize("end", ["2026-01-01T14:00:00+07:00", "not-ISO", 0, False])
def test_existing_end_is_preserved(guard, end):
    assert guard.normalize("calendar.create_event", {"start": "2026-01-01T10:00", "end": end})["end"] == end


@pytest.mark.parametrize("start", [None, "", 123, [], "tomorrow at ten", "2026-01-01", "20260101",
                                  "2026-02-30T10:00", "2026-01-01T25:00", "9999-12-31T23:30:00"])
def test_invalid_or_date_only_start_does_not_fill(guard, start):
    assert "end" not in guard.normalize("calendar.create_event", {"start": start})


def test_missing_start_does_not_fill(guard):
    assert guard.normalize("calendar.create_event", {}) == {}


@pytest.mark.parametrize("name", ["calendar.update_event", "other.create_event", "calendar.list_events"])
def test_other_tools_unchanged(guard, name):
    assert "end" not in guard.normalize(name, {"start": "2026-01-01T10:00"})


def test_disabled_by_default(guard):
    assert DEFAULTS["calendar_create_end"] is False
    guard.cfg["calendar_create_end"] = False
    assert "end" not in guard.normalize("calendar.create_event", {"start": "2026-01-01T10:00"})
