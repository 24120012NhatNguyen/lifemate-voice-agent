"""CPU/offline fp32 LoRA merge, SFT soup + DPO, or delta transfer; one bf16 cast.

See docs/MERGE_SCALED.md for report definitions and supported adapter format.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

# Bound native pools before importing torch (also when invoked as a CLI).
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

FLOAT_DTYPES = {"BF16", "F16", "F32"}
CHUNK = 1_048_576


class TensorStore:
    """Index headers only; map/read one tensor at a time, never a state_dict."""

    def __init__(self, root: Path, stem: str = "model"):
        self.root = root.resolve(strict=True)
        index = self.root / f"{stem}.safetensors.index.json"
        mapping = json.loads(index.read_text())["weight_map"] if index.exists() else None
        files = sorted(set(mapping.values())) if mapping is not None else [f"{stem}.safetensors"]
        self.entries = {}
        for filename in files:
            path = (self.root / filename).resolve(strict=True)
            if not path.is_relative_to(self.root):
                raise ValueError(f"shard outside model directory: {filename}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self.entries:
                        raise ValueError(f"duplicate tensor: {key}")
                    if mapping is not None and mapping.get(key) != filename:
                        raise ValueError(f"index/shard mismatch: {key}")
                    view = handle.get_slice(key)
                    self.entries[key] = (path, tuple(view.get_shape()), view.get_dtype())
        if not self.entries or (mapping is not None and set(mapping) != set(self.entries)):
            raise ValueError("empty or incomplete safetensors index")

    def get(self, name):
        with safe_open(self.entries[name][0], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def compatible(self, other, *, allow_float_dtype=False):
        if self.entries.keys() != other.entries.keys():
            raise ValueError(f"tensor set mismatch: {self.root} vs {other.root}")
        for key, (_, shape, dtype) in self.entries.items():
            _, other_shape, other_dtype = other.entries[key]
            dtype_ok = dtype == other_dtype or (allow_float_dtype and {dtype, other_dtype} <= FLOAT_DTYPES)
            if shape != other_shape or not dtype_ok:
                raise ValueError(f"shape/dtype mismatch: {key}")


class LoRA:
    """The plain Linear LoRA format saved by train_sft_q35r/train_dpo_q35."""

    def __init__(self, root, base):
        cfg = json.loads((root / "adapter_config.json").read_text())
        if cfg.get("peft_type") != "LORA" or cfg.get("bias", "none") != "none":
            raise ValueError("only plain LORA with bias=none is supported")
        for option in ("use_dora", "use_rslora", "fan_in_fan_out", "modules_to_save", "lora_bias",
                       "rank_pattern", "alpha_pattern", "layer_replication", "target_parameters",
                       "layers_to_transform", "exclude_modules", "trainable_token_indices",
                       "use_qalora", "alora_invocation_tokens", "megatron_config"):
            if cfg.get(option):
                raise ValueError(f"unsupported LoRA option: {option}")
        rank, scale = cfg["r"], cfg["lora_alpha"]
        if not isinstance(rank, int) or rank <= 0 or not math.isfinite(scale):
            raise ValueError("invalid LoRA rank/alpha")
        self.scale = scale / rank
        self.store = TensorStore(root, "adapter_model")
        pairs = {}
        for key in self.store.entries:
            match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
            if not match:
                raise ValueError(f"unsupported adapter tensor name: {key}")
            module, side = match.groups()
            pairs.setdefault(module + ".weight", {})[side] = key
        self.pairs = {}
        for weight, pair in pairs.items():
            if set(pair) != {"A", "B"} or weight not in base.entries:
                raise ValueError(f"unpaired LoRA or missing base tensor: {weight}")
            _, ashape, adtype = self.store.entries[pair["A"]]
            _, bshape, bdtype = self.store.entries[pair["B"]]
            _, wshape, wdtype = base.entries[weight]
            if (len(ashape) != 2 or len(bshape) != 2 or ashape[0] != rank or bshape[1] != rank
                    or wshape != (bshape[0], ashape[1])):
                raise ValueError(f"LoRA/base shape or rank mismatch: {weight}")
            if not {adtype, bdtype, wdtype} <= FLOAT_DTYPES:
                raise ValueError(f"unsupported LoRA dtype: {weight}")
            self.pairs[weight] = (pair["A"], pair["B"])
        targets = cfg.get("target_modules")
        if not targets:
            raise ValueError("explicit target_modules required")
        if not isinstance(targets, str):
            for target in targets:
                if not any(key == target + ".weight" or key.endswith("." + target + ".weight")
                           for key in base.entries):
                    raise ValueError(f"unknown target_modules entry: {target}")
        expected = set()
        for key in base.entries:
            if not key.endswith(".weight"):
                continue
            module = key.removesuffix(".weight")
            matches = (re.fullmatch(targets, module) if isinstance(targets, str)
                       else any(module == t or module.endswith("." + t) for t in targets))
            if matches:
                expected.add(key)
        if expected != set(self.pairs):
            raise ValueError("target_modules do not match adapter tensor set")

    def delta(self, key, alpha):
        a, b = (self.store.get(k).float() for k in self.pairs[key])
        # Same operation order as PEFT at alpha=1; all arithmetic is fp32.
        return (b @ a).mul_(self.scale).mul_(alpha)


@dataclass
class Stats:
    tensors: int = 0
    elements: int = 0
    delta_nonzero: int = 0
    retained_nonzero: int = 0
    delta_sq: float = 0.0
    retained_sq: float = 0.0

    def add(self, other):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def report(self):
        return {"tensors": self.tensors, "elements": self.elements,
                "delta_nonzero": self.delta_nonzero, "retained_nonzero": self.retained_nonzero,
                "nonzero_retention_ratio": self.retained_nonzero / self.delta_nonzero if self.delta_nonzero else None,
                "delta_l2": math.sqrt(self.delta_sq), "retained_l2": math.sqrt(self.retained_sq),
                "norm_retention_ratio": math.sqrt(self.retained_sq / self.delta_sq) if self.delta_sq else None}


def merge_tensor(base, delta):
    """Chunk arithmetic/statistics; only the final weight is rounded to bf16."""
    result = torch.empty(base.shape, dtype=torch.bfloat16)
    stats = Stats(tensors=1, elements=base.numel())
    original, output = base.reshape(-1), result.reshape(-1)
    flat_delta = delta.reshape(-1) if delta is not None else None
    for start in range(0, base.numel(), CHUNK):
        w = original[start:start + CHUNK].float()
        d = flat_delta[start:start + CHUNK] if flat_delta is not None else torch.zeros_like(w)
        if not torch.isfinite(w).all() or not torch.isfinite(d).all():
            raise ValueError("non-finite input weight or fp32 delta")
        new = (w + d).to(torch.bfloat16)
        if not torch.isfinite(new).all():
            raise ValueError("non-finite bf16 merged weight")
        output[start:start + CHUNK] = new
        retained = new.float() - w.to(torch.bfloat16).float()
        stats.delta_nonzero += torch.count_nonzero(d).item()
        stats.retained_nonzero += torch.count_nonzero(retained).item()
        stats.delta_sq += d.double().square().sum().item()
        stats.retained_sq += retained.double().square().sum().item()
    return result, stats


@dataclass
class Comparison:
    tensors: int = 0
    elements: int = 0
    reference_sq: float = 0.0
    difference_sq: float = 0.0

    def add(self, other):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def report(self):
        return {"tensors": self.tensors, "elements": self.elements,
                "reference_l2": math.sqrt(self.reference_sq),
                "difference_l2": math.sqrt(self.difference_sq),
                "relative_l2": math.sqrt(self.difference_sq / self.reference_sq) if self.reference_sq else None}


def compare_tensor(candidate, reference):
    """Compare stored bf16 candidate with reference; bounded fp64 norm reductions."""
    stats = Comparison(tensors=1, elements=candidate.numel())
    candidate, reference = candidate.reshape(-1), reference.reshape(-1)
    for start in range(0, candidate.numel(), CHUNK):
        c = candidate[start:start + CHUNK].to(torch.bfloat16).float()
        r = reference[start:start + CHUNK].float()
        if not torch.isfinite(c).all() or not torch.isfinite(r).all():
            raise ValueError("non-finite comparison weight")
        stats.reference_sq += r.double().square().sum().item()
        stats.difference_sq += (c.double() - r.double()).square().sum().item()
    return stats


def rebuild_soup(weight, key, adapters, weights):
    # copy=True also protects mmap storage when the input is already fp32.
    soup = weight.to(dtype=torch.float32, copy=True)
    for adapter, coefficient in zip(adapters, weights, strict=True):
        if key in adapter.pairs:
            delta = adapter.delta(key, coefficient)
            if not torch.isfinite(delta).all():
                raise ValueError(f"non-finite SFT delta: {key}")
            soup.add_(delta)
            del delta  # Never retain K dense updates.
    return soup


def module_type(key):
    parts = key.split(".")
    return parts[-2] if parts[-1] in {"weight", "bias"} and len(parts) > 1 else parts[-1]


def copy_assets(source, dest):
    # Explicit inference assets only; never copy source shards, adapters or training state.
    names = {"config.json", "generation_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "vocab.txt", "merges.txt", "spiece.model", "chat_template.json"}
    for file in sorted(source.iterdir()):
        if file.is_file() and (file.name in names or file.name.startswith("tokenizer")
                               or (file.name.startswith("chat_template") and file.suffix == ".jinja")):
            shutil.copyfile(file, dest / file.name)
    templates = source / "chat_templates"
    if templates.is_dir():
        shutil.copytree(templates, dest / "chat_templates")
    config_path = dest / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("quantization_config"):
        raise ValueError("quantized base/target is not supported")
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


@torch.no_grad()
def run(*, out, alpha=1.0, base=None, adapter=None, target=None, source_ref=None, source=None, threads=8,
        sft_adapters=None, sft_weights=None, dpo_adapter=None, compare=None):
    if not math.isfinite(alpha) or not 1 <= threads <= 8:
        raise ValueError("alpha must be finite and threads must be in [1, 8]")
    no_soup_args = all(x is None for x in (sft_adapters, sft_weights, dpo_adapter, compare))
    lora_mode = (base is not None and adapter is not None and no_soup_args
                 and all(x is None for x in (target, source_ref, source)))
    transfer_mode = (all(x is not None for x in (target, source_ref, source)) and no_soup_args
                     and base is None and adapter is None)
    soup_mode = (base is not None and sft_adapters is not None
                 and all(x is None for x in (adapter, target, source_ref, source)))
    if not (lora_mode or transfer_mode or soup_mode):
        raise ValueError("use --base + --adapter OR --target + --source-ref + --source "
                         "OR --base + --sft-adapters [--sft-weights ... --dpo-adapter ... --compare ...]")
    if soup_mode:
        sft_adapters = list(sft_adapters)
        if not sft_adapters:
            raise ValueError("at least one SFT adapter required")
        sft_weights = list(sft_weights) if sft_weights is not None else [1 / len(sft_adapters)] * len(sft_adapters)
        if len(sft_weights) != len(sft_adapters) or not all(math.isfinite(w) for w in sft_weights):
            raise ValueError("SFT weights must be finite, with one coefficient per adapter")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError(f"output must not exist: {out}")
    baseline = TensorStore(Path(target if transfer_mode else base))
    lora = LoRA(Path(adapter), baseline) if lora_mode else None
    sfts = [LoRA(Path(path), baseline) for path in sft_adapters] if soup_mode else []
    dpo = LoRA(Path(dpo_adapter), baseline) if dpo_adapter is not None else None
    comparison = TensorStore(Path(compare)) if compare is not None else None
    if comparison is not None:
        baseline.compatible(comparison, allow_float_dtype=True)
    src = TensorStore(Path(source)) if transfer_mode else None
    ref = TensorStore(Path(source_ref)) if transfer_mode else None
    if transfer_mode:
        baseline.compatible(src)
        baseline.compatible(ref)
    for store in (baseline, src, ref, comparison):
        if store is not None:
            config = json.loads((store.root / "config.json").read_text())
            if config.get("quantization_config"):
                raise ValueError("quantized checkpoints are not supported")
    report = {"schema_version": 2, "mode": "soup" if soup_mode else "lora" if lora_mode else "transfer", "alpha": alpha,
              "threads": threads, "compute_dtype": "float32", "output_float_dtype": "bfloat16",
              "inputs": {k: str(Path(v).resolve()) for k, v in
                         {"base": base, "adapter": adapter, "target": target,
                          "source_ref": source_ref, "source": source}.items() if v is not None},
              "definitions": {"delta": "alpha * (B_fp32 @ A_fp32) * (lora_alpha/r), or alpha * (source_fp32-ref_fp32)",
                              "retained": "output_bf16.float() - baseline.bfloat16().float()",
                              "nonzero_retention_ratio": "count_nonzero(retained) / count_nonzero(delta)",
                              "norm_retention_ratio": "sqrt(sum(retained**2) / sum(delta**2)); sums in fp64",
                              "zero_denominator": "null (undefined)",
                              "scope": "all floating tensors; module type = final module name; integer buffers unchanged"},
              "per_tensor": {}, "by_module_type": {}, "non_float_tensors": []}
    if soup_mode:
        report["inputs"]["sft_adapters"] = [str(ad.store.root) for ad in sfts]
        report["sft_weights"] = sft_weights
        report["inputs"]["dpo_adapter"] = str(dpo.store.root) if dpo is not None else None
        report["definitions"].update({
            "soup": "W_without = base_fp32 + sum(w_k * (B_k_fp32 @ A_k_fp32) * (lora_alpha_k/r_k)); "
                    "sequential fp32 additions in supplied order; missing modules contribute zero",
            "delta": "alpha * (B_D_fp32 @ A_D_fp32) * (lora_alpha_D/r_D); zero if DPO absent",
            "retained": "(W_without + delta).bfloat16().float() - W_without.bfloat16().float()",
            "scope": "DPO-only retention over all floating tensors, grouped by final module name; "
                     "integer buffers unchanged; diagnostic casts never feed the output computation"})
    compare_totals = {name: Comparison() for name in ("soup_without_dpo", "output")}
    if comparison is not None:
        report["comparison"] = {
            "reference": str(comparison.root),
            "definition": "relative_l2 = ||candidate_bf16.float() - reference.float()||_2 / ||reference.float()||_2; "
                          "soup_without_dpo = cast(W_without), output = cast(W_without + delta); "
                          "all floating tensors, fp64 sums; zero reference norm gives null",
            **{name: {"per_tensor": {}} for name in compare_totals}}
    groups, total, weight_map, total_size = {}, Stats(), {}, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        copy_assets(baseline.root, staging)
        keys = sorted(baseline.entries)
        for i, key in enumerate(keys, 1):
            # Scope each tensor to a function so no previous tensor survives into the next read.
            def process():
                weight = baseline.get(key)
                if not weight.is_floating_point():
                    if weight.is_complex():
                        raise ValueError(f"unsupported dtype: {key}")
                    if transfer_mode and not torch.equal(src.get(key), ref.get(key)):
                        raise ValueError(f"changed non-floating source buffer: {key}")
                    if comparison is not None and not torch.equal(weight, comparison.get(key)):
                        raise ValueError(f"changed non-floating comparison buffer: {key}")
                    return weight, None, None
                if baseline.entries[key][2] not in FLOAT_DTYPES:
                    raise ValueError(f"unsupported floating dtype: {key}")
                delta = None
                if lora_mode and key in lora.pairs:
                    delta = lora.delta(key, alpha)
                elif transfer_mode:
                    delta = src.get(key).float().sub_(ref.get(key).float()).mul_(alpha)
                elif soup_mode:
                    weight = rebuild_soup(weight, key, sfts, sft_weights)
                    if dpo is not None and key in dpo.pairs:
                        delta = dpo.delta(key, alpha)
                merged, stats = merge_tensor(weight, delta)
                comparisons = None
                if comparison is not None:
                    reference = comparison.get(key)
                    comparisons = {"soup_without_dpo": compare_tensor(weight, reference),
                                   "output": compare_tensor(merged, reference)}
                return merged, stats, comparisons

            merged, stats, comparisons = process()
            if stats is not None:
                report["per_tensor"][key] = stats.report()
                groups.setdefault(module_type(key), Stats()).add(stats)
                total.add(stats)
            else:
                report["non_float_tensors"].append(key)
            if comparisons is not None:
                for name, value in comparisons.items():
                    report["comparison"][name]["per_tensor"][key] = value.report()
                    compare_totals[name].add(value)
            filename = f"model-{i:05d}-of-{len(keys):05d}.safetensors"
            save_file({key: merged.contiguous()}, staging / filename, metadata={"format": "pt"})
            weight_map[key] = filename
            total_size += merged.numel() * merged.element_size()
            del merged
        report["by_module_type"] = {key: value.report() for key, value in sorted(groups.items())}
        report["total"] = total.report()
        if comparison is not None:
            for name, value in compare_totals.items():
                report["comparison"][name]["total"] = value.report()
        (staging / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n")
        (staging / "merge_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Fail without publishing a partial checkpoint, including on invalid late tensors.
        if out.exists() or out.is_symlink():
            raise ValueError(f"output appeared during merge: {out}")
        staging.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "target", "source-ref", "source", "dpo-adapter", "compare"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--sft-adapters", type=lambda value: csv_list(value, Path),
                        help="comma-separated local SFT adapter directories, in accumulation order")
    parser.add_argument("--sft-weights", type=lambda value: csv_list(value, float),
                        help="comma-separated finite coefficients (default: 1/K each; no normalization)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        report = run(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"merge failed: {error}\n")
    print(json.dumps({"out": str(args.out), "total": report["total"]}, indent=2))


def csv_list(value, convert):
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("comma-separated list must not contain empty entries")
    try:
        return [convert(part.strip()) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


if __name__ == "__main__":
    main()
