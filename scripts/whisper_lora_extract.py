"""Tách lại adapter LoRA của một Whisper đã gộp, để image chỉ mang MỘT bản gốc + nhiều adapter (LM-50, brief mục 0000).

Hai ASR của bản cuối (phowhisper-large-ft và phowhisper-large-ft2-all) đều là LoRA hạng 32 trên vinai/PhoWhisper-large,
đã gộp rồi lưu fp16; adapter gốc không còn. Hai bản gộp là 2 × 1,54 tỷ tham số, cộng Qwen3.5-4B thì vượt 6 tỷ. Với mỗi
Linear đích (q/k/v/out_proj, fc1, fc2): delta = W_gộp − W_gốc có hạng ≤ 32 cộng nhiễu làm tròn fp16, nên SVD cắt ở hạng
32 trả lại B·A (và bỏ phần nhiễu). Adapter ghi theo định dạng PEFT; lifemate.asr.adapters.merge_lora_into cộng nó vào bản
gốc fp32 rồi mới về fp16, đúng thứ tự của finetune_whisper.merge. Trọng số ra không trùng từng bit bản gộp cũ: một phần
phần tử lệch một bậc làm tròn fp16 (--check in tỉ lệ), cùng cỡ với sai số fp16 vốn có.

    .venv/bin/python scripts/whisper_lora_extract.py models/phowhisper-large-ft2-all --out models/phowhisper-large-ft2-all-lora \
        --save-base models/phowhisper-large-base --check

Chỉ CPU (khoảng 10 GB RAM, vài phút). --check: gộp lại và so từng phần tử với bản gộp cũ; có peft thì so cả với PeftModel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # như scripts/finetune_whisper.py


def extract(base, merged_state: dict, rank: int, alpha: int) -> tuple[dict, dict]:
    """Trả (tensor adapter theo khoá PEFT, báo cáo). base là model fp32; merged_state là state dict của bản gộp."""
    scale = alpha / rank
    out, tails, resid, n_params = {}, [], [], 0
    targets = set()
    for name, mod in base.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS:
            targets.add(name + ".weight")
            d = merged_state[name + ".weight"].double() - mod.weight.detach().double()
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            root = s[:rank].sqrt()
            a = (root[:, None] * vh[:rank]).float()
            b = (u[:, :rank] * root / scale).float()
            out[f"base_model.model.{name}.lora_A.weight"] = a.contiguous()
            out[f"base_model.model.{name}.lora_B.weight"] = b.contiguous()
            n_params += a.numel() + b.numel()
            tails.append((s[rank] / s[0]).item() if len(s) > rank else 0.0)
            resid.append((d - (b.double() @ a.double()) * scale).abs().max().item())
    other = [k for k, v in base.state_dict().items()
             if k not in targets and k in merged_state and not torch.equal(v.to(merged_state[k].dtype), merged_state[k])]
    report = {"rank": rank, "alpha": alpha, "modules": len(tails), "adapter_params": n_params,
              "sigma_next_over_sigma_1_max": max(tails), "residual_abs_max": max(resid),
              "non_target_tensors_differing_from_base": len(other), "examples": other[:5]}
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged", help="thư mục Whisper đã gộp (fp16)")
    ap.add_argument("--base", default="vinai/PhoWhisper-large", help="id HF hoặc thư mục bản gốc (fp32)")
    ap.add_argument("--out", required=True, help="thư mục adapter ra")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=None, help="mặc định 2 × rank như finetune_whisper.py")
    ap.add_argument("--save-base", default=None, help="lưu bản gốc fp32 dạng safetensors + processor vào đây (để đóng vào image)")
    ap.add_argument("--check", action="store_true", help="gộp lại và so từng phần tử với bản gộp cũ")
    ap.add_argument("--check-peft", action="store_true", help="kèm --check: so thêm với PeftModel.merge_and_unload (cần peft)")
    a = ap.parse_args(argv)
    from safetensors.torch import load_file, save_file
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    from lifemate.asr.adapters import merge_lora_into

    torch.set_num_threads(8)
    path = lambda p: REPO / p if (REPO / p).is_dir() else Path(p)  # noqa: E731
    merged_dir, out = path(a.merged), path(a.out)
    alpha = a.alpha or 2 * a.rank
    base = WhisperForConditionalGeneration.from_pretrained(str(path(a.base)) if path(a.base).is_dir() else a.base,
                                                           torch_dtype=torch.float32)
    merged_state = load_file(merged_dir / "model.safetensors")
    tensors, report = extract(base, merged_state, a.rank, alpha)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "adapter_model.safetensors", metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": None, "base_model_name_or_path": a.base, "r": a.rank, "lora_alpha": alpha,
        "lora_dropout": 0.0, "target_modules": TARGETS, "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
    }, indent=1), encoding="utf-8")
    if (merged_dir / "train_info.json").exists():
        info = json.loads((merged_dir / "train_info.json").read_text(encoding="utf-8"))
        report["train_info"] = {k: info[k] for k in ("base", "train_utts", "best_epoch") if k in info}
    report["source_merged"] = str(a.merged)
    if a.save_base:
        bdir = path(a.save_base)
        if not (bdir / "model.safetensors").exists():
            base.save_pretrained(bdir, safe_serialization=True)
            AutoProcessor.from_pretrained(str(merged_dir)).save_pretrained(bdir)
        report["base_dir"] = str(a.save_base)
        report["base_params"] = sum(p.numel() for p in base.parameters())
    if a.check:
        n = merge_lora_into(base, out)
        again = {k: v.to(torch.float16) for k, v in base.state_dict().items()}
        total = diff = 0
        worst = 0.0
        for k, v in merged_state.items():
            if k in again and v.dtype == torch.float16:
                ne = again[k] != v
                total += v.numel()
                diff += int(ne.sum())
                if ne.any():
                    worst = max(worst, (again[k].float() - v.float()).abs().max().item())
        report["check"] = {"modules_merged": n, "elements": total, "elements_differing": diff,
                           "fraction_differing": diff / max(1, total), "abs_diff_max": worst}
        if a.check_peft:  # nạp thêm một bản gốc (6 GB RAM nữa): PeftModel đọc được adapter và gộp ra đúng cùng trọng số
            from peft import PeftModel

            fresh = WhisperForConditionalGeneration.from_pretrained(str(path(a.base)) if path(a.base).is_dir() else a.base,
                                                                    torch_dtype=torch.float32)
            pm = PeftModel.from_pretrained(fresh, str(out)).merge_and_unload().state_dict()
            report["check"]["same_as_peft_merge"] = all(torch.equal(pm[k], v) for k, v in base.state_dict().items())
    (out / "extract_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
