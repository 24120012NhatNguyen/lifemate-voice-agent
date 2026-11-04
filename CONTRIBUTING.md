# Contributing

## Workflow

1. Branch from an up-to-date `main`: `<type>/<short-description>`, for example `feat/asr-entity-bias` or
   `exp/sft-rename-all`. Use the same types as commit messages (below), plus `exp/` for experiment-only branches.
2. Keep each pull request to one change: one feature, one fix, or one experiment with its configs and results.
3. Open a pull request into `main` using the template, and get one review before merging. Prefer squash merge so
   `main` keeps one commit per change.
4. Delete the branch after merging.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), in English:

```
<type>(<optional scope>): <imperative summary, 72 characters at most>

<body: what changed and why, wrapped at 72 characters>
```

Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`. Mark breaking changes with `!`
after the type or a `BREAKING CHANGE:` footer.

## Experiments

- One experiment is one file `configs/experiments/NN_name.yaml` that `extends: ../base.yaml` and changes as little as
  possible. Take the next free number and say in its header comment what it is compared against.
- Compare variants on the same samples and report paired differences with their standard error; the app's
  comparison tabs do this from `runs/`.
- Record every result that changes a decision in [docs/BASELINE.md](docs/BASELINE.md) and update
  [docs/roadmap.yaml](docs/roadmap.yaml).

## Shared GPU

- Start GPU work through `scripts/gpu_job.sh` so jobs queue on one lock instead of running out of memory together.
  Set `GPU_LOCK` and `GPU_LOG` to a shared path when several worktrees or people use the same card.
- Check `nvidia-smi` before starting anything. Never stop a job you did not start.

## Data, weights and secrets

- Never commit competition data, model weights, run outputs, submissions or credentials. `.gitignore` covers the usual
  paths and the pre-commit hooks reject large files and private keys.
- Datasets uploaded to Kaggle or anywhere else must stay private: the organisers' licence covers registered teams only.

## Code

- Python 3.11 and 3.12 must both work. Lint with ruff; the configuration is in `pyproject.toml`.
- Add or update tests in `tests/` with every behaviour change. Tests that need the data package skip when it is absent.
- Identifiers and commit messages are in English. Docstrings, comments and the detailed docs may be in Vietnamese.

Before opening a pull request:

```bash
.venv/bin/pre-commit run --all-files
```

```bash
.venv/bin/pytest -q
```
