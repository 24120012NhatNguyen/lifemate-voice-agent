"""Tách adapter LoRA từ một Whisper đã gộp (scripts/whisper_lora_extract.py) rồi cộng lại vào bản gốc
(lifemate.asr.adapters.merge_lora_into): trên Whisper tí hon, trọng số gộp lại trùng bản gộp fp16 ban đầu, và PEFT đọc
được adapter. Chỉ CPU, không dữ liệu BTC."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("whisper_lora_extract", REPO / "scripts" / "whisper_lora_extract.py")
wle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wle)

from lifemate.asr.adapters import merge_lora_into  # noqa: E402


def tiny():
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(3)
    cfg = WhisperConfig(vocab_size=64, d_model=48, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=96, decoder_ffn_dim=96, num_mel_bins=8,
                        max_source_positions=20, max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        decoder_start_token_id=1)
    return WhisperForConditionalGeneration(cfg).float()


def test_extract_then_merge_reproduces_merged_fp16(tmp_path):
    from safetensors.torch import save_file

    rank, alpha = 4, 8
    base = tiny()
    start = {k: v.clone() for k, v in base.state_dict().items()}
    merged = {k: v.clone() for k, v in start.items()}
    n_target = 0
    for name, mod in base.named_modules():  # bản "đã train": cộng một delta hạng 4 vào mọi Linear đích, rồi lưu fp16
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in wle.TARGETS:
            out_f, in_f = mod.weight.shape
            merged[name + ".weight"] = start[name + ".weight"] + (torch.randn(out_f, rank) @ torch.randn(rank, in_f)) * 0.02
            n_target += 1
    merged16 = {k: v.to(torch.float16) for k, v in merged.items()}

    tensors, report = wle.extract(base, merged16, rank, alpha)
    assert report["modules"] == n_target == 2 * 6 + 2 * 10  # encoder 6 Linear đích mỗi lớp, decoder 10
    assert report["non_target_tensors_differing_from_base"] == 0
    assert report["sigma_next_over_sigma_1_max"] < 1e-2  # sau hạng 4 chỉ còn nhiễu làm tròn fp16

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file(tensors, adapter / "adapter_model.safetensors", metadata={"format": "pt"})
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": None, "base_model_name_or_path": "tiny", "r": rank, "lora_alpha": alpha,
        "lora_dropout": 0.0, "target_modules": wle.TARGETS, "bias": "none", "fan_in_fan_out": False, "inference_mode": True}))
    assert merge_lora_into(base, adapter) == n_target
    again = {k: v.to(torch.float16) for k, v in base.state_dict().items()}
    total = sum(v.numel() for v in merged16.values())
    differing = sum(int((again[k] != v).sum()) for k, v in merged16.items())
    assert differing / total < 0.5, f"{differing}/{total} phần tử khác bản gộp"  # khác thì chỉ một bậc làm tròn fp16
    worst = max((again[k].float() - v.float()).abs().max().item() for k, v in merged16.items())
    assert worst < 1e-3

    peft = pytest.importorskip("peft")
    fresh = tiny()
    fresh.load_state_dict(start)
    pm = peft.PeftModel.from_pretrained(fresh, str(adapter)).merge_and_unload().state_dict()
    assert all(torch.allclose(pm[k], v, atol=1e-6) for k, v in base.state_dict().items())


def test_merge_rejects_unpaired_adapter(tmp_path):
    from safetensors.torch import save_file

    base = tiny()
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    save_file({"base_model.model.model.encoder.layers.0.fc1.lora_B.weight": torch.zeros(96, 4)},
              tmp_path / "adapter_model.safetensors")
    with pytest.raises(ValueError):
        merge_lora_into(base, tmp_path)
