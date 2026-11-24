# ZAIC 2026 · LifeMate — team solution

A Vietnamese voice assistant that hears a request (plus the audio of earlier user turns) and answers it by calling
tools in the organisers' `zaic_tools` environment. Each sample is scored by the F1 of the call list, gated by whether
the final environment state (`state_hash`) matches the reference.

```
audio ─► ASR (2 passes) ──────────────────► agent (LLM, ReAct JSON) ─► guards ─► zaic_tools.ToolEnv ─► tool_calls + state_hash
         PhoWhisper-large base                Qwen3.5-4B (text only)
         + LoRA "ft-all"  (beam 4, 4-best)    LoRA SFT x 3 seeds, weights averaged (soup)
         + LoRA "ft2-all" (greedy, name/app prompt)
```

## Results

Team **zAIo_Astra**, small-model track (< 6 B parameters).

| Milestone | Public leaderboard score |
|---|---:|
| First submission (PhoWhisper-large LoRA + Qwen3-4B SFT agent) | 42.15 |
| Final model (config 270: Qwen3.5-4B, 3-seed soup with guard-recovery traces) | **72.86** |
| Output of the submitted Docker image on the public test | 72.83 |

The Docker image was submitted on 16/11/2026 and its private-test predictions were submitted after the private release;
the final ranking is announced by the organisers.

## Running the image (what the organisers run)

```bash
docker run --gpus '"device=0"' -v <data>:/data:ro -v <result>:/result IMAGE /bin/bash /code/predict.sh
```

- Input: `/data/private_test/` (`private_test.jsonl`, `private_test_tools_catalog.json`, `private_test_environment/`,
  `audio/<id>/...`, `zaic_tools/`). The split is found automatically (`private_test` when present). The split's own
  `zaic_tools` is imported for every `ToolEnv` (one per test case); no copy of `zaic_tools` from the image is used.
- Output: `/result/submission.jsonl` (the folder is created if missing), one line per test case
  `{"id", "tool_calls", "state_hash", "time"}`, `time` in milliseconds. stdout prints the model-load time and the
  prediction time (`load model: ... ms`, `predict: N samples, total ... ms, mean ... ms/sample`). Logs, the vLLM log and
  per-sample traces go to `/result/predict_logs/`.
- No network is needed: every model (ASR, LLM, tokenizers) is inside the image and HF offline mode is forced.
- All hyper-parameters are fixed in `configs/docker-final.yaml` and `predict.py`; no argument is needed.
  `docker run ... IMAGE` without a command runs the same `predict.sh` (the image has no ENTRYPOINT).
- Timing notebook: `docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <result>:/result IMAGE
  /bin/bash /code/start_jupyter.sh`, open port 9777, password/token `zac2026`, open `predict_notebook.ipynb`, Run All.
  It writes `/result/jupyter_submission.jsonl` (same format, with `time`) and logs to `/result/jupyter_logs/`.

### Why the notebook gives the same predictions as `predict.sh`

`predict.sh` only runs `predict.py`. The notebook imports the same `predict.py` and calls the same functions in
separate cells (seed → test cases → ASR → load LLM → agent on every test case → write). Both start the same
subprocesses with the same config, models, flags and environment (PYTHONHASHSEED=0, HF offline, a fresh ASR cache),
and decoding is deterministic: ASR beam search/greedy in fp16 over the same batches, LLM temperature 0 and seed 0,
vLLM serving one sequence at a time (`--max-num-seqs 1`) while the agent runs the test cases one after another.
Only the measured `time` differs between the two files.

### Per-sample time

`time` = the test case's ASR seconds (its share of the 32-sample ASR chunk it was transcribed in, both passes, model
loading excluded) + its agent wall time (from its first LLM call to its final state, `ToolEnv` included). Model loading
(PhoWhisper, vLLM start-up) is printed separately: the FAQ says it is not counted. A 25-second budget per sample (ASR
included) stops the agent and keeps the calls already made, so a slow GPU cannot push a sample past 30 seconds.

## Models

| Part | Model | Parameters (safetensors headers) |
|---|---|---:|
| ASR base | `vinai/PhoWhisper-large` (BSD-3-Clause), revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` | 1,543,304,960 |
| ASR adapter 1 | LoRA r=32 "ft-all": all 5,036 train utterances, 3 epochs | 57,671,680 |
| ASR adapter 2 | LoRA r=32 "ft2-all": all train utterances, speed 0.9/1.0/1.1, name/app prompt, 6 epochs | 57,671,680 |
| LLM | `Qwen/Qwen3.5-4B` (Apache-2.0), text-only export (vision tower and MTP head removed), LoRA SFT merged, 3-seed soup | 4,205,751,296 |
| **Total** | | **5,864,399,616** (< 6 B) |

- The PhoWhisper-large base ships in the image because inference applies the two LoRA adapters on top of it at load
  time (its weights are bit-identical to the pinned revision); the Qwen3.5-4B training base is not in the image, only
  the merged, fine-tuned LLM. How the adapters were trained and extracted: `training_code/README.md`, section 4.
- ASR: each adapter is added to the fp32 base when it is loaded, then the model is cast to fp16
  (`src/lifemate/asr/adapters.py`). Pass 1: beam 4, 4-best, batch 2. Pass 2: greedy, batch 16, prompted with the
  profile/history names and the installed apps' vocabulary (apps rare in train first; ranking in
  `models/app_rank_train.json`, computed from train only). Pass 2's text joins the alternatives shown to the agent.
  History turns are transcribed too.
- LLM: Qwen3.5-4B text-only (`qwen3_5_text`), bf16, served by vLLM 0.33.0 with `--max-model-len 16384` (context 16,384
  tokens; the longest single call we measured is about 7k prompt tokens) and at most 400 output tokens per step, with the
  empty-think chat template `prompts/qwen35_scaffold.jinja` (thinking disabled), temperature 0, seed 0.

## Agent loop

- Prompt (`src/lifemate/context.py`, `prompts/react_json.md`, `prompts/rules/verified.md`): user profile, current time and
  a two-week calendar table, a compact schema of the installed apps (tool name, read/write, required parameters, `x-ref`
  ID sources, free-text fields), the conversation history with ASR text for user turns, a ledger of earlier calls and
  results, the ASR alternatives, and tool-use rules distilled from the train data.
- Each step the LLM returns one JSON object `{"thought", "action": "call" | "stop", "name", "arguments"}`. Decoding is
  constrained by a JSON Schema (`action` is call or stop, `name` an installed tool or `sys.ask_missing_info`). The call
  runs on the real `ToolEnv` and its result is appended; at most 8 steps.
- Guards (`src/lifemate/guards.py`, `fixes.py`) check every call before it reaches `ToolEnv`: schema validation and type
  normalisation (numbers, ISO datetimes with timezone), calendar end from a spoken duration, clock and relative-day
  fixes, spoken money amounts, unknown-ID and duplicate checks, asking for missing information instead of inventing it.
- A decision layer with fixed confidence thresholds (`agent.decision` in the config) can turn a low-confidence action
  into a clarifying question or a stop, mainly for apps never seen in train.

## Data and synthetic data (LLM SFT)

Training file of the submitted model: `training_code/data/train_full_para_v1_asrfold_val_guard.jsonl`, 6,928 multi-turn
conversations built exactly like the inference prompts (system and user turns built by the agent above, one assistant
turn per gold call followed by the tool result from `label.tool_results`, then stop; loss only on assistant turns):

| Part | Conversations | How it was made |
|---|---:|---|
| Train gold traces | 1,542 | every BTC train sample (3 dropped because a guard flagged their gold call) |
| Synthetic scenarios | 600 | generated by code from the train tool catalogue (`src/lifemate/synth/`, fixed seeds) |
| D1 paraphrases | 2,991 | train requests re-phrased by a teacher LLM (Claude family), gold trajectory kept, filtered by an independent teacher solve |
| ASR-fold copies | 948 | user turns replaced by 5-fold ASR transcripts (each fold's ASR never trained on it), so the agent learns real ASR errors |
| Validation ASR copies | 480 | the same for the samples once held out for validation |
| Guard-recovery traces | 367 | traces where a guard note corrects a wrong call (`scripts/build_guard_traces.py`) |

Only the train split is used to build training data; the public test, including its tool catalogue, is never used.
Teacher models are named by family only (Claude / GPT / Gemini); the paraphrase set used here was written by Claude.

Training recipe (LoRA r=16, α=32, dropout 0, lr 2e-4, 1 epoch, grad-accum 16, max length 12,288, fp16, seeds 0/1/2),
merging and the 3-seed average: [`training_code/README.md`](training_code/README.md).

## Repository layout (= `/code` in the image)

| Path | Content |
|---|---|
| `predict.sh`, `predict.py` | entry point and driver (ASR → vLLM → agent → `submission.jsonl`) |
| `predict_notebook.ipynb`, `start_jupyter.sh` | timing notebook (same driver) and JupyterLab launcher |
| `configs/docker-final.yaml` | the submitted configuration |
| `src/lifemate/` | pipeline: ASR, agent, guards, runner |
| `prompts/` | agent prompt, rules, chat template |
| `models/` | ASR base + 2 adapters, merged LLM, app ranking (in the image only) |
| `training_code/` | training code, training data, receipts, how to reproduce |
| `scripts/` | data builders, ASR fine-tuning, merge/soup, Docker tooling |
| `requirements.txt` | pipeline/ASR environment; `docker/requirements-vllm.txt` and `docker/requirements-jupyter.txt` for the others |

The image has three Python environments: `.venv` (pipeline + ASR: torch 2.11 cu128, transformers 4.57.6, from
`uv.lock`), `.venv-vllm` (vLLM 0.33.0, torch cu130) and `.venv-jupyter` (JupyterLab 4.4.10, also runs `predict.py`).
Base image `nvidia/cuda:13.0.3-base-ubuntu24.04` (CUDA 13.0 ≤ 13.2). In the image `/code` is a link to `/app`.

## Developer notes

Local runs: `uv sync --extra asr`, a vLLM venv from `docker/requirements-vllm.txt`, data under `../data/lifemate` (or
`LIFEMATE_DATA`), then `.venv/bin/lifemate run configs/<config>.yaml`. Build the image with
`scripts/docker_swap.sh <merged Qwen3.5 dir> && TAG=<tag> scripts/docker_build.sh`; check it the way the organisers will
with `scripts/btc_sim.sh <image.tar.gz> [data_dir] [reference.jsonl]`. Licence: see `LICENSE`.

## Acknowledgements

This project was built with AI coding assistants (Anthropic Claude, OpenAI Codex and Google Gemini), used under the
author's direction to write and review code, run experiment pipelines and draft documentation. Design choices,
experiments and submissions were decided by the author.
