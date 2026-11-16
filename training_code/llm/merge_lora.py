"""Gộp một adapter LoRA (vd. tải về từ Kaggle) vào mô hình gốc bf16 để vLLM phục vụ trực tiếp.

Chạy trên CPU nên không cần khoá GPU: khoảng 2 phút và 16 GB RAM với Qwen3-4B.

    .venv/bin/python scripts/merge_lora.py models/qwen3-4b-sft-drop-lora models/qwen3-4b-sft-drop
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", help="thư mục có adapter_config.json và adapter_model.safetensors")
    ap.add_argument("out", help="thư mục mô hình đã gộp")
    ap.add_argument("--base", default=None, help="mặc định lấy base_model_name_or_path trong adapter_config.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter, out = REPO / a.adapter, REPO / a.out
    base = a.base or json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    if (adapter / "train_args.json").exists():
        shutil.copy(adapter / "train_args.json", out / "train_args.json")
    print(f"đã gộp {adapter} → {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
