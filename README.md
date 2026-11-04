# ZAIC 2026 · LifeMate

Team solution for the **LifeMate** track of the Zalo AI Challenge 2026: a voice assistant that hears a Vietnamese
request and answers it by calling tools in the organisers' `zaic_tools` environment. Each sample is scored by the F1 of
the call list, gated by whether the final environment state (`state_hash`) matches the reference.

```
audio ──► ASR ──► ASR post-processing ──► agent (LLM, ReAct JSON) ──► guards ──► ToolEnv ──► tool_calls + state_hash
         PhoWhisper-large (LoRA)                Qwen3-4B (LoRA SFT)
```

Every stage is swappable from a YAML config, so an experiment is one file in `configs/experiments/` and one command.
The first public submission (config `90_submit_public`, fine-tuned PhoWhisper-large then the SFT agent) scored
**42.15**, in line with the local validation mean of 42.2.

## Requirements

- Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with 16 GB for vLLM, ASR and LoRA training. `pyproject.toml` pins CUDA 12.8 wheels for torch, which
  RTX 50xx cards need.
- The organisers' data package (see [Data](#data)). It is licensed to registered teams only: never commit or share it.

## Setup

```bash
uv sync --extra asr --extra app --extra dev --extra chunkformer --extra sft --extra train
```

```bash
uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm
```

```bash
.venv/bin/python scripts/download_models.py
```

```bash
.venv/bin/pre-commit install
```

## Data

The data is not in the repository. Put the organisers' package next to it, or point `LIFEMATE_DATA` at it:

```
zalo-ai-2026/
├── data/lifemate/
│   ├── train/          train.jsonl, audio/, train_environment/, zaic_tools/
│   └── public_test/    public_test.jsonl, audio/, public_test_environment/, zaic_tools/
└── lifemate/           this repository
```

Do not modify `zaic_tools/` or `*_environment/`: the organisers score with the original copies.

## Usage

Run an experiment (results go to `runs/<timestamp>_<config>/`):

```bash
.venv/bin/lifemate run configs/experiments/01_oracle.yaml
```

Serve the agent LLM for configs that need it (OpenAI-compatible API on port 8000):

```bash
DETERMINISTIC=1 MODEL=models/qwen3-4b-sft scripts/serve_vllm.sh
```

Queue GPU work so several people or worktrees can share one card (`GPU_LOCK` and `GPU_LOG` default to `logs/`):

```bash
SET="limit=100" scripts/gpu_job.sh eval models/qwen3-4b-sft configs/experiments/40_*.yaml
```

Build SFT data, train the LoRA agent and merge an adapter trained elsewhere (for example on Kaggle):

```bash
.venv/bin/python scripts/build_sft.py --out sft/train.jsonl
```

```bash
.venv/bin/python scripts/train_sft.py --data sft/train.jsonl --out models/qwen3-4b-sft
```

```bash
.venv/bin/python scripts/merge_lora.py models/qwen3-4b-sft-drop-lora models/qwen3-4b-sft-drop
```

Fine-tune ASR:

```bash
.venv/bin/python scripts/finetune_whisper.py --model vinai/PhoWhisper-large --out models/phowhisper-large-ft
```

Compare runs, pick the best stage combination and watch running jobs:

```bash
.venv/bin/streamlit run app/compare.py
```

Produce and check a submission file:

```bash
.venv/bin/lifemate run configs/experiments/90_submit_public.yaml
```

```bash
.venv/bin/lifemate validate runs/<timestamp>_90_submit_public
```

Run the tests (data-dependent tests skip when the data package is missing):

```bash
.venv/bin/pytest -q
```

## Repository layout

```
app/          Streamlit app: best architecture, run comparison, live GPU jobs, roadmap, dataset browser
configs/      base.yaml and one file per experiment (experiments/NN_name.yaml)
docs/         PROPOSAL.md (architecture), BASELINE.md (how to run, every result), roadmap.yaml
kaggle/       runner for LoRA SFT on Kaggle T4 x2 notebooks
prompts/      agent prompt templates and tool-use rules
scripts/      training, data building, serving, GPU queue and packaging scripts
src/lifemate/ the package: data, splits, ASR, agent, guards, scoring, runner, CLI
tests/        pytest suite
```

Local only, ignored by git:

```
models/       fine-tuned and merged weights
runs/         one folder per experiment run (predictions, per-sample traces, metrics)
cache/        ASR transcripts keyed by ASR config
sft/          generated SFT datasets and Kaggle packs
logs/         training and serving logs, GPU.md, gpu.lock
artifacts/    downloads such as Kaggle outputs
```

## Competition constraints

- Under 6B parameters in total, under 5B for the LLM, 16K context.
- 30 seconds per sample on one A30 GPU, offline, inside Docker.
- One `ToolEnv` per sample; every call counts towards the submission.

## Documentation

The detailed documents are in Vietnamese:

- [docs/PROPOSAL.md](docs/PROPOSAL.md): pipeline architecture and plan.
- [docs/BASELINE.md](docs/BASELINE.md): installation, running, adding a variant, and every experiment result.
- [docs/roadmap.yaml](docs/roadmap.yaml): milestones and next proposals (also shown in the app).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Copyright © 2026 the LifeMate team. All rights reserved; see [LICENSE](LICENSE). Third-party models keep their own
licences, listed in [docs/BASELINE.md](docs/BASELINE.md) section 9.
