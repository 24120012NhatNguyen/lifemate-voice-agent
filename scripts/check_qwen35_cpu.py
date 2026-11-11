"""Validate the real exported checkpoint on CPU; no dataset or GPU is used."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export_text_only import EXPECTED_PARAMS, REVISION, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/qwen35-4b-text"))
    ap.add_argument("--report", type=Path, default=Path("reports/lm43/checkpoint-cpu.json"))
    ap.add_argument("--source-tokenizer", type=Path, default=Path.home() /
        ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION)
    a = ap.parse_args()
    torch.set_num_threads(2)
    receipt = json.loads((a.model / "export_receipt.json").read_text())
    for name, entry in receipt["output_files"].items():
        if sha256(a.model / name) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {name}")
    start = time.monotonic()
    model, info = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, device_map="cpu",
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(info)
    info = {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}
    count = sum(p.numel() for p in model.parameters())
    assert count == EXPECTED_PARAMS
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert all(p.device.type == "cpu" for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    original = AutoTokenizer.from_pretrained(a.source_tokenizer, local_files_only=True)
    samples = ["Xin chào Việt Nam!", "1.234.567₫ ngày 27/02/2026", "a\u0301 á \n 42", "<|im_start|>assistant\n"]
    assert tokenizer.chat_template == original.chat_template
    for sample in samples:
        assert tokenizer.encode(sample) == original.encode(sample)
    assert tokenizer.backend_tokenizer.to_str() == original.backend_tokenizer.to_str()
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": "Xin chào"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
    with torch.inference_mode():
        logits = model(**inputs, logits_to_keep=1, use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert not torch.cuda.is_initialized()
    report = {"architecture": type(model).__name__, "parameters": count,
              "loading_info": info, "device": "cpu", "dtype": str(model.dtype),
              "tied_head": True, "receipt_hashes_verified": True, "finite_forward": True,
              "tokenizer_backend_and_template_equal_source": True,
              "forward_tokens": inputs["input_ids"].shape[-1], "logits_shape": list(logits.shape),
              "elapsed_seconds": round(time.monotonic() - start, 2), "cuda_initialized": False,
              "vllm_engine_load": "NOT RUN: GPU is not authorized"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
