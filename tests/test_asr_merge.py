"""asr.merge_with: bản chép lượt cuối của ASR thứ hai thành phương án đầu, đọc từ cache, không chạy model."""
import json

from lifemate.asr import base
from lifemate.config import stage_key

MAIN = {"name": "hf_whisper", "model": "models/a", "num_beams": 4, "n_best": 4, "batch_size": 2}
SECOND = {"model": "models/b", "num_beams": 1, "batch_size": 16}


def _write(cache, cfg, rows):
    key = stage_key("train", cfg, True, "none")
    path = cache / "asr" / "train" / f"{cfg['name']}_{key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _row(sid, final, alts=None):
    return {"id": sid, "final": final, "turns": {}, "alts": {"final": alts} if alts else {}, "sec": 0.1}


def test_merge_adds_new_hypothesis_first(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_CACHE", str(tmp_path))
    _write(tmp_path, MAIN, [_row("s1", "Tắt đèn bếp.", ["Tắt đèn bết."]), _row("s2", "Bật quạt.", ["Bật quát."]),
                            _row("s3", "Gọi mẹ.")])
    _write(tmp_path, {"name": "hf_whisper", **SECOND}, [_row("s1", "Tắt đèn phòng bếp."), _row("s2", "bật quạt"),
                                                        _row("s3", "Gọi mẹ nhé.")])
    samples = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]
    out = base.run_asr("train", samples, {"asr": {**MAIN, "merge_with": SECOND}}, log=lambda *_: None)
    assert out["s1"]["alts"]["final"] == ["Tắt đèn phòng bếp.", "Tắt đèn bết."]
    assert out["s2"]["alts"]["final"] == ["Bật quát."]  # chỉ khác hoa thường, dấu câu: không thêm
    assert out["s3"]["alts"]["final"] == ["Gọi mẹ nhé."]  # bản chính không có phương án nào
    assert out["s1"]["final"] == "Tắt đèn bếp."


def test_merge_keeps_main_cache_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_CACHE", str(tmp_path))
    _write(tmp_path, MAIN, [_row("s1", "Mở nhạc.")])
    _write(tmp_path, {"name": "hf_whisper", **SECOND}, [_row("s1", "Mở nhạc.")])
    plain = base.run_asr("train", [{"id": "s1"}], {"asr": dict(MAIN)}, log=lambda *_: None)
    merged = base.run_asr("train", [{"id": "s1"}], {"asr": {**MAIN, "merge_with": SECOND}}, log=lambda *_: None)
    assert plain == merged  # bản trùng không thêm gì; không đọc cache khác cho bản chính
