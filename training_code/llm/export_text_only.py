"""Export cached Qwen3.5 multimodal weights as a strict, offline text-only checkpoint.

No GPU, remote code, or network is used. One source shard is mapped at a time.
Receipt counts stored tensors (tied embedding/head counted once), not config estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import struct
from collections import Counter
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_PARAMS = 4_205_751_296
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "LICENSE")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 100_000_000:
            raise ValueError("Implausible safetensors header")
        return {k: v for k, v in json.loads(stream.read(size)).items() if k != "__metadata__"}


def tensor_group(name):
    if name.startswith("model.language_model."):
        return "language_model"
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    # This revision stores no separate lm_head: fail closed if that changes.
    raise ValueError(f"Unknown source tensor: {name}")


def target_name(name):
    return "model." + name.removeprefix("model.language_model.") if tensor_group(name) == "language_model" else None


def text_config(config):
    cfg = dict(config["text_config"])
    if cfg["model_type"] != "qwen3_5_text" or not cfg.get("tie_word_embeddings"):
        raise ValueError("Expected tied Qwen3.5 text architecture")
    cfg.update(architectures=["Qwen3_5ForCausalLM"], mtp_num_hidden_layers=0,
               transformers_version=importlib.metadata.version("transformers"))
    return cfg


def inspect_source(source):
    groups = Counter()
    headers = {}
    seen = set()
    for path in sorted(source.glob("*.safetensors")):
        headers[path.name] = read_header(path)
        for name, info in headers[path.name].items():
            if name in seen:
                raise ValueError(f"Duplicate tensor: {name}")
            seen.add(name)
            groups[tensor_group(name)] += math.prod(info["shape"])
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index) != seen or any(index[k] != shard for shard, h in headers.items() for k in h):
        raise ValueError("Source index does not match shard headers")
    return headers, dict(groups)


def export(source, out, revision=REVISION, expected=EXPECTED_PARAMS):
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, out = Path(source).resolve(), Path(out)
    if revision != REVISION or source.name != REVISION:
        raise ValueError("Source must be the pinned HF snapshot directory (full revision)")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    headers, groups = inspect_source(source)
    if groups.get("language_model") != expected:
        raise ValueError(f"Unexpected text parameter count: {groups}")
    cfg = text_config(json.loads((source / "config.json").read_text()))
    # Verify complete key/shape coverage against the actual Transformers architecture on meta.
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg))
    actual = {target_name(k): tuple(v["shape"]) for h in headers.values() for k, v in h.items() if target_name(k)}
    wanted = {k: tuple(v.shape) for k, v in model.state_dict().items() if k != "lm_head.weight"}
    if actual != wanted or sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError("Tensor coverage/shape/count differs from Qwen3_5ForCausalLM")
    out.mkdir(parents=True)
    try:
        weight_map, total_size = {}, 0
        for shard, header in headers.items():
            tensors = {}
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                for name in header:
                    target = target_name(name)
                    if target:
                        tensors[target] = handle.get_tensor(name)
            if tensors:
                save_file(tensors, out / shard, metadata={"format": "pt"})
                for name, tensor in tensors.items():
                    weight_map[name] = shard
                    total_size += tensor.numel() * tensor.element_size()
                del tensors
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n")
        for name in TOKENIZER_FILES:
            shutil.copyfile(source / name, out / name)
        # Explicit decoding defaults; the source snapshot has no generation_config.json.
        tok_cfg = json.loads((source / "tokenizer_config.json").read_text())
        # AutoTokenizer maps qwen3_5_text to Qwen2 unless this override is explicit.
        # Preserve Qwen3.5's combining-mark regex; copying the original Qwen2 class silently changes it.
        tok_cfg["tokenizer_class"] = "Qwen3_5Tokenizer"
        (out / "tokenizer_config.json").write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2) + "\n")
        special_ids = {v["content"]: int(k) for k, v in tok_cfg["added_tokens_decoder"].items()}
        eos_ids = sorted({cfg["eos_token_id"], special_ids[tok_cfg["eos_token"]]})
        (out / "generation_config.json").write_text(json.dumps({
            "_from_model_config": True, "bos_token_id": cfg.get("bos_token_id"),
            "eos_token_id": eos_ids, "pad_token_id": special_ids[tok_cfg["pad_token"]],
            "do_sample": False}, indent=2) + "\n")
        receipt = {
            "source_model": MODEL_ID, "source_revision": revision,
            "architecture": "Qwen3_5ForCausalLM", "model_type": cfg["model_type"],
            "tokenizer_class_override": "Qwen3_5Tokenizer (preserves the source pretokenizer regex)",
            "parameter_count_convention": "unique stored parameters; tied lm_head = model.embed_tokens, counted once",
            "source_parameter_groups": groups, "source_total_parameters": sum(groups.values()),
            "exported_parameters": expected, "removed_parameters": sum(groups.values()) - expected,
            "exported_tensors": len(weight_map), "tied_weights": {"lm_head.weight": "model.embed_tokens.weight"},
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
            "source_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(source.iterdir()) if p.is_file()},
            "output_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(out.iterdir()) if p.is_file()},
            "checks": {"meta_key_shape_coverage": True, "gpu_used": False,
                       "vllm_engine_load": "NOT RUN: GPU is not authorized"},
        }
        (out / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except BaseException:
        shutil.rmtree(out)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION)
    ap.add_argument("--out", type=Path, default=Path("models/qwen35-4b-text"))
    args = ap.parse_args()
    receipt = export(args.source, args.out)
    print(json.dumps({k: receipt[k] for k in ("source_revision", "source_parameter_groups", "exported_parameters", "exported_tensors")}))


if __name__ == "__main__":
    main()
