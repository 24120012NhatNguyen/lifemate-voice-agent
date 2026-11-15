# Scaled merge / SFT soup / delta transfer (LM-64, LM-66)

`scripts/merge_scaled.py` only opens local safetensors; no model instantiation,
GPU, network, training data, or PEFT dependency at runtime. Requires PyTorch and
safetensors already installed. Run from any directory; paths are relative to the
current directory. Output must **not exist**. Inputs are never modified.

## LoRA → base

```bash
python scripts/merge_scaled.py --base /models/soup211 \
  --adapter /models/step-dpo-lora --alpha 1 --out /models/dpo-a1 --threads 8
# Repeat with --alpha 2 / 3 and distinct outputs for extrapolation.
```

For each targeted Linear weight, compute `D = (B.float() @ A.float()) *
(lora_alpha / r) * alpha`, then `W_out = (W.float() + D).bfloat16()`.
No intermediate delta cast to bf16. Other floating tensors are converted to bf16;
integer buffers retain their original dtype/value. `alpha=0` returns tensor-exact
base weights **when the base is already bf16** (files/shard layout need not match).
For fp32/fp16 input, identity means `base.bfloat16()`, as required by output dtype.

Supports the plain Linear LoRA saved by `train_sft_q35r.py` and
`train_dpo_q35.py`: `base_model.model.<module>.lora_A.weight` / `lora_B.weight`,
global `r`/`lora_alpha`, suffix target names (Qwen2) or full names (Qwen3.5).
Checks paired A/B, target coverage, shapes, ranks and floating dtypes. All seven
attention/MLP and five DeltaNet projections are supported without renaming them.
Nonstandard variants (DoRA, rsLoRA, rank/alpha patterns, bias updates,
modules_to_save, embedding/Conv LoRA, etc.) are rejected rather than silently
merged incorrectly. Quantized checkpoints and fp64/float8 weights are rejected.

PEFT parity is measured against `peft_model.float().merge_and_unload().bfloat16()`.
At alpha=1, tests also match the installed PEFT default `merge_and_unload()`
on the bf16 base with fp32 adapters, exactly. PEFT with `safe_merge=True` on a
bf16 base, or bf16 adapters, can round the delta *before* addition; those paths
are not bit-exact oracles for the single-rounding algorithm. The regression suite
contains a value just above half an ulp where these two paths differ.

## Transfer an already merged delta

```bash
python scripts/merge_scaled.py --target /models/soup98f7 \
  --source-ref /models/soup211 --source /models/dpo-a1 \
  --alpha 1 --out /models/soup98f7-dpo-a1 --threads 8
```

Computes `(target.float() + alpha * (source.float() - ref.float())).bfloat16()`.
All three inputs must have exactly the same stored tensor keys, shapes and dtypes;
shard layouts may differ. A changed non-floating source buffer is rejected.
With `source=ref`, bf16 target tensors are preserved exactly.

**This cannot recover a delta already lost when source was merged to bf16.**
To retain the original adapter update while moving it to soup98f7, the first mode
can instead use `--base /models/soup98f7 --adapter /models/step-dpo-lora` after the
manager confirms architecture and training provenance. Matching tensor metadata
does not establish model lineage or prove transfer improves scores. The caller
must select the exact original reference checkpoint. `alpha` in transfer mode
scales the difference *already present* in source; do not inadvertently scale an
alpha-2 source twice.

## Rebuild an SFT soup in fp32, then add DPO

```bash
python scripts/merge_scaled.py --base /models/qwen35-4b-text \
  --sft-adapters /models/sft-seed0-lora,/models/sft-seed1-lora \
  --dpo-adapter /models/step-dpo-lora --alpha 1 \
  --compare /models/old-bf16-soup --out /models/rebuilt-soup-dpo --threads 8
# Omit --dpo-adapter for an SFT-only soup. --compare is optional.
# Optional literal coefficients: --sft-weights 0.25,0.75
```

Starting from the **original base**, compute, in the listed adapter order:

```text
W_without = W_base.float() + sum(w_k * (B_k.float() @ A_k.float()) * (lora_alpha_k / r_k))
D         = alpha * (B_D.float() @ A_D.float()) * (lora_alpha_D / r_D)
W_output  = (W_without + D).bfloat16()
```

All sums and matrix products use fp32. No intermediate soup or delta is rounded
to bf16 on the output computation path. The diagnostic cast of `W_without` does
not feed back into the computation. Missing DPO means D=0; DPO with alpha=0
returns exactly the SFT-only output (the adapter is still validated).

`--sft-adapters` requires at least one local directory, comma-separated. Each
adapter has its own rank, scale and target module set. A module absent from an
adapter contributes zero; **its weight is not redistributed** among adapters
that do target the module. All supplied names, A/B pairs, shapes and declared
target coverage are validated against the base, including zero-weight adapters.
There must be exactly K finite `--sft-weights`; the default is 1/K per adapter.
Explicit coefficients are used literally, without normalization; negative and
zero coefficients are supported. `alpha` scales DPO only, never the SFT soup.
Do not use an already merged SFT checkpoint as `--base`: this would add SFT twice.

In soup mode, the existing `per_tensor`, `by_module_type` and `total` sections
measure **DPO only**, with `delta = D` and
`retained = cast(W_without + D).float() - cast(W_without).float()`. Thus the
fractional part of the fp32 soup participates in the measurement. Without DPO,
counts/norms are zero and retention ratios are null. Provenance records the
ordered absolute adapter paths, coefficients, DPO path, base and alpha.

`--compare MERGED_DIR` is supported in soup mode. It requires identical tensor
keys/shapes, allows supported float dtypes to differ, and requires integer
buffers to match exactly. The `comparison` section contains two separate
`per_tensor`/`total` reports:

- `soup_without_dpo`: `cast(W_without)` versus the stored comparison checkpoint.
  This isolates reconstruction differences from the new DPO update.
- `output`: `cast(W_without + D)` versus that checkpoint.

Each reports `difference_l2`, `reference_l2` and
`relative_l2 = ||candidate - reference||₂ / ||reference||₂`. The reference is
the stored comparison tensor, promoted to fp32 without extra bf16 rounding.
Totals use fp64 sums of squared elements, not averages of tensor ratios. Zero
reference norm gives null even when both tensors are zero; use `difference_l2`
to distinguish equality. The report schema is now version 2; existing LoRA and
transfer statistics retain their previous meanings.

At base W=1, an SFT update 3/1024 gives an unrounded soup of 1+3/1024. Adding
DPO 2/1024 (less than half the upward bf16 ulp, 1/256) produces output 1+1/128.
Rounding the soup first produces 1, then adding the same DPO still produces 1.
In this fixture the new nonzero retention ratio is 1 and the norm ratio is 4;
the pre-rounded path retains zero. This demonstrates the rounding mechanism,
**not** a universal 90% retention guarantee. The norm can increase due to
quantization; retained nonzero counts/L2 do not measure direction fidelity,
expected amplitude or model quality. Actual checkpoint statistics and validation
scores require a separately authorized run with the chosen artifacts.

## Output and report

Produces Hugging Face-compatible sharded safetensors (one tensor per shard), an
index, and `merge_report.json`. Copies config, generation config, tokenizer files,
vocabulary/merges, and chat template files/directory **from base/target**, not from
source/adapter. Updates existing config dtype fields to bfloat16. No training
state or input weight files are copied. A failed merge removes its staging files;
the output directory is published only after the index and report are complete.

Report sections `per_tensor`, `by_module_type`, `total` cover all floating tensors.
Module type is the last module name (e.g. `q_proj`, `in_proj_qkv`, `down_proj`),
aggregated across layers. The report lists integer buffers separately.

- `delta_nonzero`: nonzero elements of the **scaled fp32** delta.
- `retained_nonzero`: nonzero elements of
  `R = W_out.float() - W_input.bfloat16().float()`; in soup mode `W_input`
  is the unrounded fp32 `W_without` defined above.
- `nonzero_retention_ratio = retained_nonzero / delta_nonzero`.
- `delta_l2 = sqrt(sum(D²))`, `retained_l2 = sqrt(sum(R²))`.
- `norm_retention_ratio = retained_l2 / delta_l2`.

Sums of squares are accumulated in fp64 in bounded chunks; tensor arithmetic is
fp32. Aggregate norm ratios come from aggregate sums of squares, not an average
of per-tensor ratios. Zero denominators produce JSON `null`. Rounding may increase
the norm ratio above 1; the ratio is not a quality score or directional cosine.

At W=1, bf16 upward ulp is 1/128. The fixture with delta rows `[1/1024, 1/128]`
retains 2 of 4 elements: nonzero ratio 0.5 and norm ratio `sqrt(64/65)`.

## Resource bounds and validation

Headers are indexed first, tensors are read individually, and no full model
state_dict is resident. Memory grows with the largest tensor plus its LoRA factors,
not total checkpoint size; matmul still materializes one full fp32 delta. Transfer
also holds source/ref intermediates for one tensor. Statistics use chunks of
1,048,576 elements. Safetensors serialization can temporarily copy the current
output tensor. This is not a fixed RAM cap or an in-place merge.
Soup mode retains one fp32 accumulator and at most one dense SFT delta at a
time, then the current DPO delta/output. Comparison loads only the matching
reference tensor; diagnostic reductions/casts are chunked. K increases header
metadata and computation time, not the number of resident dense deltas. No full
checkpoint or full adapter state_dict is loaded.

CLI defaults to 8 intra-op threads (accepts 1–8), one inter-op thread, capped native
pool environment variables, and no CUDA visibility. Processing is sequential.
Many small output shards trade loading/file-count overhead for bounded memory.
No full 4B runtime/RSS or vLLM compatibility benchmark has been run in this task.
Tiny Qwen2 and Qwen3.5 reload through Transformers with exact reference weights.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LIFEMATE_CPU_ONLY=1 PYTHONPATH=src OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_merge_scaled.py
ruff check scripts/merge_scaled.py tests/test_merge_scaled.py
```

All test models/tokenizers are synthetic. Tests cover alpha 0/1/2/3, FP32 PEFT
parity, Qwen3.5 DeltaNet naming, transfer identity/formula, differing shard layouts,
tied embeddings reload, config/template copying, small-delta loss, chunked norms,
unsupported formats, mismatched metadata, CLI errors and output cleanup.
Soup tests also cover K=1 parity/reload, K=2 hand-calculated equal/custom/signed
weights, DPO alpha=0/1/3/-1, overlapping and disjoint SFT/DPO targets, rank/scale
differences, sub-half-ulp dithering, comparison norms/zero denominators,
non-finite inputs, invalid coefficients and malformed comma-separated CLI lists.
