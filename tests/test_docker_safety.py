"""Các đường lỗi có thể làm hỏng lượt chạy Docker (review Argon 14/11, Pro lọc): fixture tổng hợp, không cần dữ liệu BTC."""
import numpy as np
import pytest
import soundfile as sf

from lifemate import data, fixes
from lifemate.asr import base
from lifemate.guards import GuardedEnv

SAMPLE = {"context": {"installed_apps": {}, "current_time": "2026-01-01T00:00:00+07:00"}}


def test_unreadable_audio_becomes_silence(tmp_path, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav file")
    for path in (bad, tmp_path / "missing.wav"):
        wav = base.load_audio(path)
        assert wav.dtype == np.float32 and len(wav) == base.SILENCE_SAMPLES and not wav.any()
    assert "CẢNH BÁO" in capsys.readouterr().out


def test_empty_audio_becomes_silence(tmp_path):
    for n, sr in ((0, 16000), (0, 44100), (100, 16000)):
        path = tmp_path / f"e{n}_{sr}.wav"
        sf.write(str(path), np.zeros(n, dtype="float32"), sr)
        assert len(base.load_audio(path)) == base.SILENCE_SAMPLES


def test_good_audio_is_unchanged(tmp_path):
    path = tmp_path / "ok.wav"
    wav = (np.sin(np.arange(16000) / 20) * 0.1).astype("float32")
    sf.write(str(path), np.stack([wav, wav], axis=1), 16000, subtype="FLOAT")
    np.testing.assert_allclose(base.load_audio(path), wav, atol=1e-6)


@pytest.fixture
def guard(monkeypatch):
    spec = {"kind": "write", "parameters": {"properties": {"n": {"type": "integer"},
                                                           "at": {"type": "string", "description": "ISO-8601"}}}}
    monkeypatch.setattr(data, "tool_specs", lambda _: {"x.set": spec})
    return GuardedEnv(None, "train", SAMPLE, clock_fix=True)


@pytest.mark.parametrize("raw, want", [("1.200.000", 1200000), ("1,200", 1200), (" 45 ", 45), ("-3", -3),
                                       ("15.0", 15), ("100,00", 100), ("1.000", 1000)])
def test_integer_strings_are_parsed(guard, raw, want):
    assert guard.normalize("x.set", {"n": raw})["n"] == want


@pytest.mark.parametrize("raw", [".", "-", " , ", "1-2", "1,5", "1.5", "2.50", "12,05"])
def test_integer_strings_without_a_number_are_left_for_validation(guard, raw):
    assert guard.normalize("x.set", {"n": raw})["n"] == raw


@pytest.mark.parametrize("error", [ValueError("day is out of range for month"),
                                   TypeError("can't subtract offset-naive and offset-aware datetimes")])
def test_a_failing_fix_rule_leaves_the_call_as_the_model_wrote_it(guard, monkeypatch, error):
    def boom(*a, **k):
        raise error
    monkeypatch.setattr(fixes, "fix_args", boom)
    assert guard.normalize("x.set", {"at": "2026-11-31T09:00"}) == {"at": "2026-11-31T09:00:00+07:00"}
    assert guard.fix_notes == []


def test_validate_keeps_the_caller_environment(monkeypatch, tmp_path):
    import subprocess
    import sys

    from lifemate import cli

    (tmp_path / "config.yaml").write_text("split: train\n", encoding="utf-8")
    seen = {}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/driver")
    monkeypatch.setattr(data, "data_root", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "call", lambda cmd, cwd, env: seen.update(env) or 0)
    monkeypatch.setattr(sys, "argv", ["lifemate", "validate", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0 and seen["LD_LIBRARY_PATH"] == "/opt/driver" and seen["PYTHONPATH"].endswith("train")


def test_asr_hint_survives_null_profile_and_null_assistant_text():
    sample = {"context": {"user_profile": None, "installed_apps": {}},
              "history": [{"role": "assistant", "text": None, "tool_results": [{"name": "Anh Tuấn"}]},
                          {"role": "assistant", "tool_results": None}]}
    assert isinstance(base.bias_hint(sample, "profile_history", "train"), str)
    assert base.bias_hint({"context": {"user_profile": {"name": None, "city": "Huế"}}, "history": None},
                          "profile", "train") == "Huế"


def test_missing_app_rank_file_is_a_warning_not_a_crash(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LIFEMATE_APP_RANK", str(tmp_path / "nope.json"))
    base._app_rank.cache_clear() if hasattr(base._app_rank, "cache_clear") else None
    assert base._app_rank() == {} and "CẢNH BÁO" in capsys.readouterr().out
    base._app_rank.cache_clear() if hasattr(base._app_rank, "cache_clear") else None


def test_a_failing_ask_rule_runs_the_call_instead_of_ending_the_sample(monkeypatch):
    class Env:
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            return {"ok": True}

    spec = {"kind": "write", "parameters": {"properties": {"at": {"type": "string"}}}}
    monkeypatch.setattr(data, "tool_specs", lambda _: {"x.set": spec})
    g = GuardedEnv(Env(), "train", SAMPLE, ask_missing=True, id_guard="off")

    def boom(*a, **k):
        raise ValueError("day is out of range for month")
    monkeypatch.setattr(fixes, "missing", boom)
    g.allowed = {"x.set"}
    assert g.call("x.set", {"at": "2026-11-31T09:00"}) == {"ok": True}
