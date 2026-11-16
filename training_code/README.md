# Training code and data

Everything needed to rebuild the weights used at inference. The training base models are public on HuggingFace and are
downloaded at a pinned revision. The Qwen3.5-4B base is **not** in the image (only the fine-tuned, merged LLM is). The
PhoWhisper-large weights are in the image (`models/phowhisper-large-base`) only because inference adds the two LoRA
adapters on top of them at load time; they are bit-identical to the revision below (all 1,259 tensors compared).

| Base | HuggingFace id | Revision |
|---|---|---|
| LLM | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (pinned in `llm/run_sft.py`, `llm/export_text_only.py`) |
| ASR | `vinai/PhoWhisper-large` | `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` |

Seeds are fixed everywhere (LLM SFT seeds 0/1/2, ASR seed 2026, data builders' seeds in their commands).
Training needs internet (base models, pip packages); inference does not.

```
training_code/
├── data/                          LLM SFT data (sha256 in DATA_SHA256SUMS)
│   ├── train_full_para_v1_asrfold_val_guard.jsonl   6,928 conversations: data of the SUBMITTED model (soup fg3)
│   ├── train_full_para_v1_asrfold_val.jsonl         6,561: the same without the 367 guard-recovery traces
│   └── train_full_para_v1_asrfold_val_ren2.jsonl    8,201: the 6,561 + 1,640 entity-renamed copies (alternative model)
├── llm/                           LLM SFT: the flat package run on Kaggle/Colab
│   ├── run_sft.py                 runner: pins libraries, downloads + exports the text-only base, trains, zips the adapter
│   ├── train_sft.py               LoRA SFT trainer (Qwen3.5 text-only; loss on assistant turns only; resumable)
│   ├── export_text_only.py        Qwen3.5-4B → text-only checkpoint (drops vision tower and MTP head): 4,205,751,296 params
│   ├── qwen35_scaffold.jinja      chat template (empty think block), identical to the one served at inference
│   ├── merge_lora.py              merges a LoRA adapter into the bf16 base
│   ├── soup.py                    averages the weights of several merged checkpoints
│   ├── receipts/                  train_args.json of the 3 seeds, soup.json of the submitted model
│   └── colab/                     Colab notebook and rented-GPU script running the same runner
└── asr/receipts/                  train_info.json of both PhoWhisper LoRA runs, adapter extraction reports
```

The code that builds the data and fine-tunes the ASR lives next to the pipeline it imports (`/code/src/lifemate`):

| What | Code (in `/code/scripts/`) |
|---|---|
| SFT data: gold traces + synthetic scenarios | `build_sft.py` (+ `src/lifemate/synth/`, replay orders `docs/repro/`) |
| SFT data: ASR-fold and validation ASR copies, TTS variants | `build_v2.py` |
| SFT data: guard-recovery traces | `build_guard_traces.py` |
| SFT data: entity-renamed copies (ren2 file) | `build_rename_v2.py` |
| Synthetic v3 scenarios (not in the submitted data) | `synth_v3/` |
| ASR fine-tuning, merge | `finetune_whisper.py` |
| ASR adapter extraction (what the image ships) | `whisper_lora_extract.py` |
| App ranking for the ASR prompt | `export_app_rank.py` |
 Step-by-step commands with the hashes of the
intermediate files (teacher paraphrases, 5-fold ASR, synthetic replay orders in `/code/docs/repro/`):
`/code/docs/REPRODUCE_98F_ARCHIVE.md` (in Vietnamese; it describes the 6,081-line base set: items 1–3 below without the 480 validation ASR copies).

## 1. LLM data

`train_full_para_v1_asrfold_val_guard.jsonl` is the concatenation of (built from the BTC **train** split only):

1. 1,542 gold traces of the train samples + 600 code-generated synthetic scenarios
   (`scripts/build_sft.py --subset all --seed 2026 --synth-replay docs/repro/synth_full_v2_orders.json`, 2,142 lines,
   sha256 `eb9ed4d7…7be067`);
2. 2,991 paraphrases of train requests written by a teacher LLM (Claude family), gold trajectories kept, kept only when
   an independent teacher solve agreed (→ `train_full_para_v1.jsonl`);
3. 948 copies whose user turns are 5-fold ASR transcripts (one PhoWhisper LoRA per fold, each never trained on the fold
   it transcribes) and 480 ASR copies of the samples once held out for validation
   (→ `train_full_para_v1_asrfold_val.jsonl`, 6,561 lines);
4. 367 guard-recovery traces (`scripts/build_guard_traces.py --subset all`), where a guard note fixes a wrong call
   (→ 6,928 lines).

Each line is `{"id", "renamed", "asr", "messages": [...]}` in the exact format of the inference prompt; assistant turns
marked `"train": false` stay in the context but carry no loss.

## 2. LLM training (one run per seed; submitted model = seeds 0, 1, 2)

Hyper-parameters (`llm/receipts/*.train_args.json`): LoRA r=16, α=32, dropout 0, targets q/k/v/o/gate/up/down
projections + the 5 Gated-DeltaNet projections, lr 2e-4, warm-up 5 %, 1 epoch, 16 samples per optimiser step, max
length 12,288 tokens, fp16 with GradScaler (Kaggle T4). Libraries pinned by the runner: transformers 5.20.2,
peft 0.21.2, fla-core 0.6.0, flash-linear-attention 0.6.0 (torch of the host image).

```bash
mkdir pkg && cp training_code/llm/{run_sft.py,train_sft.py,export_text_only.py,qwen35_scaffold.jinja} pkg/
cp training_code/data/train_full_para_v1_asrfold_val_guard.jsonl pkg/
cd pkg
export WORK=$PWD/work KAGGLE_INPUT=$PWD/none Q35_SCRATCH=$PWD/scratch
for s in 0 1 2; do   # 2 GPUs (Kaggle T4 x2): add --ddp. Re-run the same command to resume after an interruption.
  python -u run_sft.py full-guard-s$s=train_full_para_v1_asrfold_val_guard.jsonl -- --seed $s
done
# → work/qwen35-4b-sft-full-guard-s{0,1,2}-lora.zip ; text-only base exported once to scratch/qwen35-4b-text
```

## 3. LLM merge and soup

```bash
for s in 0 1 2; do
  unzip -q work/qwen35-4b-sft-full-guard-s$s-lora.zip -d work/
  python ../training_code/llm/merge_lora.py $PWD/work/qwen35-4b-sft-full-guard-s$s-lora \
    $PWD/qwen35-4b-sft-full-guard-s$s --base $PWD/scratch/qwen35-4b-text
done
python ../training_code/llm/soup.py $PWD/qwen35-4b-soup-fg3 \
  $PWD/qwen35-4b-sft-full-guard-s0 $PWD/qwen35-4b-sft-full-guard-s1 $PWD/qwen35-4b-sft-full-guard-s2   # weights 1/3 each
```

The result is `/code/models/qwen35-4b-soup-fg3` (its `soup.json` is in `llm/receipts/`). GPU training is not bit-exact
across GPU types and library builds: the same data, seeds and recipe give a model of the same quality, not identical
bytes.

## 4. ASR adapters

From `/code` with the pipeline env (`.venv`, plus `peft`), BTC train data under `$LIFEMATE_DATA/train`. Both runs use
the 5,036 train utterances (final requests and user history turns); the training receipts are in `asr/receipts/`.

```bash
# 0. base at the pinned revision
huggingface-cli download vinai/PhoWhisper-large --revision b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb \
  --local-dir models/phowhisper-large-hf
B=models/phowhisper-large-hf

# 1. LoRA fine-tuning (adapter only: --no-merge writes <out>/adapter and <out>/adapter-ep<N>)
.venv/bin/python scripts/finetune_whisper.py --model $B --data all --epochs 3 --lr 2e-4 --rank 32 \
  --batch-size 8 --grad-accum 2 --warmup 0.1 --seed 2026 --speed 1.0 --hint none --dtype float16 --no-merge \
  --save-every-epoch --out runs-asr/ft-all                         # "ft-all": final epoch 3 kept
.venv/bin/python scripts/finetune_whisper.py --model $B --data all --epochs 6 --lr 2e-4 --rank 32 \
  --batch-size 8 --grad-accum 2 --warmup 0.1 --seed 2026 --speed 0.9,1.0,1.1 --hint profile_history_apps \
  --hint-prob 0.5 --dtype float16 --no-merge --save-every-epoch --out runs-asr/ft2-all   # "ft2-all": final epoch 6 kept

# 2. merge each adapter into the fp32 base, saved in fp16 (CPU, ~13 GB RAM): the checkpoints used in experiments
.venv/bin/python scripts/finetune_whisper.py --model $B --merge runs-asr/ft-all/adapter --out models/phowhisper-large-ft-all
.venv/bin/python scripts/finetune_whisper.py --model $B --merge runs-asr/ft2-all/adapter --out models/phowhisper-large-ft2-all

# 3. what the image ships: one fp32 base + two rank-32 adapters re-extracted from the merged checkpoints
#    (two merged Whisper-large would be 7.29 B parameters with the LLM, over the 6 B limit)
.venv/bin/python scripts/whisper_lora_extract.py models/phowhisper-large-ft-all --base $B \
  --out models/phowhisper-large-ft-all-lora --save-base models/phowhisper-large-base --check
.venv/bin/python scripts/whisper_lora_extract.py models/phowhisper-large-ft2-all --base $B \
  --out models/phowhisper-large-ft2-all-lora --check

# 4. app ranking for the ASR prompt (train only)
.venv/bin/python scripts/export_app_rank.py models/app_rank_train.json
```

Step 3 takes, for every LoRA target layer, the rank-32 SVD of (merged − base). The fine-tuning delta has rank 32, so
this recovers B·A up to fp16 rounding of the merged checkpoint (largest discarded singular value ≤ 0.19 % of the
first; reports `asr/receipts/*-lora.extract_report.json`). At inference `src/lifemate/asr/adapters.py` adds
B·A·(α/r) to the fp32 base and casts to fp16, in the same order as step 2. The resulting weights are not bit-identical to
the step-2 checkpoints (12.7 % of elements differ by one fp16 rounding step), so transcripts can differ on a small
fraction of utterances.
