#!/usr/bin/env python3
"""LifeMate inference driver for the Zalo AI Challenge 2026 grading container (offline, audio -> submission.jsonl).

Used by BOTH entry points, so they run the same code path:
  /code/predict.sh             -> `python predict.py`  -> main(): writes /result/submission.jsonl
  /code/predict_notebook.ipynb -> imports this module, calls the same functions cell by cell and writes
                                  /result/jupyter_submission.jsonl

Pipeline (three phases, one process each, so a phase gives the GPU back before the next one starts):
  1. ASR: PhoWhisper-large base + LoRA adapter "ft-all" (beam 4, 4-best), then a second pass with adapter "ft2-all"
     (greedy, prompted with profile/history names and app vocabulary). Batched over the split, 32 samples per chunk.
  2. LLM: vLLM serves the merged Qwen3.5-4B agent on 127.0.0.1 only (temperature 0, seed 0, one sequence at a time).
  3. Agent: ReAct JSON loop per sample, sequentially, through the split's own zaic_tools.ToolEnv and our guards.

Per-sample `time` (ms) = that sample's share of its ASR chunk (both passes, model loading excluded) + its agent wall
time (first LLM call to final state). Model loading (PhoWhisper, vLLM) is reported separately on stdout.

Only the Python standard library is used here, so the Jupyter kernel needs no extra packages; the actual work runs in
/code/.venv (pipeline, ASR) and /code/.venv-vllm (vLLM), exactly as predict.sh always did.

Defaults are baked into the image (Dockerfile ENV). Environment overrides, for testing only:
  DATA (default /data: either /data/<split>/... or the split's files directly in /data), OUT (/result),
  SPLIT (auto: private_test if present, else the only split besides train),
  CONFIG, LLM_MODEL, ASR_MODEL, LIMIT (run on a few samples), PORT, VLLM_WAIT, GPU_UTIL, EXTRA_SET.
"""
from __future__ import annotations

import atexit
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parent  # /app in the image (/code is a symlink to it)
SEED = 0  # vLLM --seed 0, request seed 0, temperature 0, PYTHONHASHSEED=0 in every subprocess

# Defaults of the submitted image. The Dockerfile sets the same values as ENV; they are repeated here so a missing
# variable can never silently change the model or the config.
DEFAULTS = {
    "DATA": "/data",
    "OUT": "/result",
    "SPLIT": "auto",
    "SUBSET": "all",
    "CONFIG": "configs/docker-final.yaml",
    "ASR_MODEL": "models/phowhisper-large-base",
    "PORT": "8000",
    "VLLM_WAIT": "900",
    "LIFEMATE_APP_RANK": str(CODE / "models/app_rank_train.json"),
}
# vLLM servers started by this Python process (a notebook kernel keeps them across "Run all"): stopped before a new
# server starts and when the process exits, so an interrupted run never leaves a server holding the GPU or the port.
_SERVERS: list[subprocess.Popen] = []


def _stop_servers() -> None:
    for p in _SERVERS:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except OSError:
                pass
    _SERVERS.clear()


atexit.register(_stop_servers)


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def seed_everything(seed: int = SEED) -> None:
    """Seeds of the calling process (kernel or driver). The pipeline subprocesses get PYTHONHASHSEED via env()."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # not installed in the Jupyter venv; the pipeline seeds itself

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class Run:
    """State of one inference run: paths, environment, child processes, timings."""

    def __init__(self, out: str | None = None, work_name: str = "predict_logs", submission: str = "submission.jsonl",
                 log_name: str = "predict.log"):
        get = lambda k: os.environ.get(k) or DEFAULTS.get(k, "")  # noqa: E731
        self.data = Path(get("DATA"))
        self.out = Path(out or get("OUT"))
        self.work = self.out / work_name
        self.submission = self.out / submission
        self.split = get("SPLIT")
        self.subset = get("SUBSET")
        self.config = get("CONFIG")
        self.llm = os.environ.get("LLM_MODEL") or _find_llm()
        self.asr = get("ASR_MODEL")
        self.port = int(get("PORT"))
        self.vllm_wait = int(get("VLLM_WAIT"))
        self.limit = os.environ.get("LIMIT") or ""
        self.extra_set = os.environ.get("EXTRA_SET", "").split()
        self.py = str(CODE / ".venv/bin/python")
        self.vllm = None
        self.smi_stop = threading.Event()
        self.t = {}
        self.out.mkdir(parents=True, exist_ok=True)
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        self.logf = open(self.work / log_name, "a", encoding="utf-8")

    # ---------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def _stream(self, cmd: list[str]) -> int:
        """Run a pipeline subprocess, mirroring its output to stdout (notebook cell or container log) and the log file."""
        p = subprocess.Popen(cmd, cwd=CODE, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            self.logf.write(line)
        self.logf.flush()
        return p.wait()

    # ---------------------------------------------------------------- setup
    def prepare(self) -> list[dict]:
        """Resolve the split, build the subprocess environment, return the test cases in file order."""
        data, split = self.data, self.split
        if (data / f"{data.name}.jsonl").is_file():  # the split folder itself was mounted
            split, data = data.name, data.parent
        else:  # the split's contents mounted directly: /data/<split>.jsonl + /data/<split>_tools_catalog.json
            flat = sorted(f.name[:-len("_tools_catalog.json")] for f in data.glob("*_tools_catalog.json")
                          if (data / f"{f.name[:-len('_tools_catalog.json')]}.jsonl").is_file())
            if flat and (split == "auto" or split in flat):
                split = split if split in flat else ("private_test" if "private_test" in flat else flat[0])
                # the pipeline and zaic_tools expect <root>/<split>/...: a link to the mounted folder gives that layout
                root = self._root = Path(tempfile.mkdtemp(prefix="lifemate-data-"))
                (root / split).symlink_to(data.resolve(), target_is_directory=True)
                self.log(f"split contents mounted directly in {data}: using {root / split} -> {data}")
                data = root
        if split == "auto":
            found = sorted(d.name for d in data.iterdir()
                           if d.is_dir() and d.name != "train" and (d / f"{d.name}.jsonl").is_file())
            if "private_test" in found:
                split = "private_test"
            elif len(found) == 1:
                split = found[0]
            else:
                raise SystemExit(f"cannot choose a split in {data} (found: {found or 'none'}); set SPLIT")
        self.data, self.split = data, split
        cases = [json.loads(x) for x in (data / split / f"{split}.jsonl").open(encoding="utf-8") if x.strip()]

        env = {k: v for k, v in os.environ.items() if not k.startswith(("JPY_", "PYDEVD", "MPLBACKEND"))}
        env.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "PYTHONHASHSEED": str(SEED), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
            # PyTorch sampler, as in every public submission (FlashInfer would JIT-compile kernels at run time).
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTHONPATH": str(CODE / "src"),
            "LIFEMATE_DATA": str(data),
            "LIFEMATE_CACHE": tempfile.mkdtemp(prefix="lifemate-cache-"),  # always transcribe from scratch
            "LIFEMATE_APP_RANK": os.environ.get("LIFEMATE_APP_RANK") or DEFAULTS["LIFEMATE_APP_RANK"],
        })
        for k in ("VIRTUAL_ENV", "FORCE_COLOR", "CLICOLOR", "CLICOLOR_FORCE"):  # set by venv activation / ipykernel
            env.pop(k, None)
        # the pipeline never runs from the Jupyter venv: same PATH as the image's old entry point
        env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if p and ".venv-jupyter" not in p)
        # Qwen3.5 text-only serving (config.json model_type qwen3_5_text): no batch-invariant kernels for Gated
        # DeltaNet, so reproducibility comes from serving one sequence at a time; empty-think chat template.
        flavor = os.environ.get("LLM_FLAVOR") or ("qwen35" if _is_qwen35(CODE / self.llm) else "qwen3")
        if flavor == "qwen35":
            self.deterministic = os.environ.get("LLM_DETERMINISTIC", "0")
            mns = os.environ.get("MAX_NUM_SEQS", "1")
            env["GPU_UTIL"] = os.environ.get("GPU_UTIL", "0.80")
            env["VLLM_EXTRA"] = os.environ.get("VLLM_EXTRA") or (
                f"--gdn-prefill-backend triton --max-num-seqs {mns} --max-num-batched-tokens 8192 "
                f"--chat-template {CODE}/prompts/qwen35_scaffold.jinja")
        else:
            self.deterministic = os.environ.get("LLM_DETERMINISTIC", "1")
        self.flavor = flavor
        drv = _smi("driver_version")
        # torch of the vLLM venv is built for CUDA 13 (driver >= 580); older datacenter drivers use the image's compat libs
        if drv and int(drv.split(".")[0]) < 580 and Path("/usr/local/cuda/compat/libcuda.so.1").exists():
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/compat" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            self.log(f"driver {drv} < 580: using /usr/local/cuda/compat")
        self.env = env
        self.cases = cases
        self.sets = ["workers=1", "agent.time_budget=25", f"split={split}", f"subset={self.subset}",
                     f"asr.model={self.asr}", f"agent.llm.model={self.llm}",
                     f"agent.llm.base_url=http://127.0.0.1:{self.port}/v1"]
        if self.limit:
            self.sets.append(f"limit={self.limit}")
        self.sets += self.extra_set
        self.log(f"split {split} ({len(cases)} samples) in {data}, config {self.config}"
                 + (f", LIMIT {self.limit}" if self.limit else ""))
        self.log(f"LLM {self.llm} ({flavor}, DETERMINISTIC={self.deterministic}, vLLM extra: {env.get('VLLM_EXTRA', '')}),"
                 f" ASR {self.asr}, app rank {env['LIFEMATE_APP_RANK']}")
        self.log(f"GPU {_smi('name,memory.total') or '?'}, driver {drv or '?'}")
        self._start_smi()
        return cases

    def _check_port(self) -> None:
        """Never talk to a server this run did not start: stop our own leftovers, refuse anything else on the port."""
        _stop_servers()
        if _port_busy(self.port):
            raise RuntimeError(f"port {self.port} is already in use by a process this run did not start (an orphan vLLM "
                               f"from an interrupted run?). Restart the container / Jupyter kernel, or set PORT.")

    def _cli(self, *extra: str) -> list[str]:
        cmd = [self.py, "-m", "lifemate.cli", "run", self.config]
        for kv in self.sets + list(extra):
            cmd += ["--set", kv]
        return cmd + ["--runs-dir", str(self.work / "runs")]

    def _start_smi(self) -> None:
        if not shutil.which("nvidia-smi"):
            return

        def loop():
            with open(self.work / "gpu_mem.csv", "w") as f:
                while not self.smi_stop.wait(0.5):
                    f.write(f"{time.time()},{_smi('memory.used', units=False)}\n")
                    f.flush()

        threading.Thread(target=loop, daemon=True).start()

    # ---------------------------------------------------------------- phases
    def run_asr(self) -> None:
        """Phase 1: both ASR passes over every sample of the split (loads PhoWhisper, transcribes, frees the GPU)."""
        self.log("1/3 ASR")
        self._check_port()  # an orphan vLLM would also hold the GPU memory ASR needs
        self.t["asr"] = [time.time(), None]
        rc = self._stream(self._cli("name=asr", "agent.name=none"))
        self.t["asr"][1] = time.time()
        if rc:
            raise RuntimeError(f"ASR phase failed (exit {rc}); see {self.work / 'predict.log'}")

    def start_llm(self) -> None:
        """Phase 2: start vLLM with the agent model on 127.0.0.1 and wait until it serves the model."""
        self.log("2/3 vLLM")
        self._check_port()
        self.t["vllm_start"] = [time.time(), None]
        env = {**self.env, "DETERMINISTIC": self.deterministic, "MODEL": self.llm, "PORT": str(self.port)}
        with open(self.work / "vllm.log", "w") as f:
            self.vllm = subprocess.Popen([str(CODE / "scripts/serve_vllm.sh"), "--host", "127.0.0.1"], cwd=CODE,
                                         env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        _SERVERS.append(self.vllm)
        url = f"http://127.0.0.1:{self.port}/v1/models"
        ok = False
        for _ in range(self.vllm_wait):
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    ok = f'"{self.llm}"' in r.read().decode()
            except OSError:
                pass
            if ok or self.vllm.poll() is not None:
                break
            time.sleep(1)
        self.t["vllm_start"][1] = time.time()
        if not ok:
            tail = (self.work / "vllm.log").read_text(errors="replace").splitlines()[-30:]
            self.log("vLLM did not come up:\n" + "\n".join(tail))
            self.stop_llm()
            raise RuntimeError("vLLM did not start")
        self.log(f"vLLM ready in {self.t['vllm_start'][1] - self.t['vllm_start'][0]:.1f} s")

    def run_agent(self) -> Path:
        """Phase 3: the agent on every sample, sequentially; returns the run folder (samples.jsonl, predictions.jsonl)."""
        self.log("3/3 agent")
        self.t["agent"] = [time.time(), None]
        rc = self._stream(self._cli("name=submit"))
        self.t["agent"][1] = time.time()
        runs = sorted((self.work / "runs").glob("*_submit"))
        if rc or not runs or not (runs[-1] / "predictions.jsonl").exists():
            raise RuntimeError(f"agent phase failed (exit {rc}); see {self.work / 'predict.log'}")
        return runs[-1]

    def stop_llm(self) -> None:
        if self.vllm is None:
            return
        try:
            os.killpg(self.vllm.pid, signal.SIGTERM)
            self.vllm.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.vllm.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except OSError:
            pass
        self.vllm = None

    # ---------------------------------------------------------------- output
    def results(self, run: Path | None) -> list[dict]:
        """One row per test case, in input order: {id, tool_calls, state_hash, time}."""
        want = [c["id"] for c in self.cases]
        if self.limit and int(self.limit) < len(want):  # the subset lifemate.splits.select picks for `limit`
            want = sorted(random.Random(2026).sample(want, int(self.limit)))
        preds, secs = {}, {}
        if run is not None:
            preds = {r["id"]: r for r in map(json.loads, (run / "predictions.jsonl").open(encoding="utf-8"))}
            for r in map(json.loads, (run / "samples.jsonl").open(encoding="utf-8")):
                secs[r["id"]] = r["sec_asr"] + r["sec_agent"]
        missing = [i for i in want if i not in preds]
        if missing:  # never leave a test case out: empty call list with the untouched environment's hash
            self.log(f"WARNING: {len(missing)} samples without a prediction, writing empty tool_calls for them")
            preds.update(self._empty(missing))
        return [{"id": i, "tool_calls": preds[i]["tool_calls"], "state_hash": preds[i]["state_hash"],
                 "time": int(round(1000 * secs.get(i, 0.0)))} for i in want]

    def _empty(self, ids: list[str]) -> dict:
        code = ("import json,sys\nfrom lifemate import data\nfor i in sys.argv[2:]:\n"
                " e=data.tool_env(sys.argv[1],i)\n"
                " print(json.dumps({'id':i,'tool_calls':[],'state_hash':e.get_state_hash()}))")
        rows = {}
        try:
            out = subprocess.run([self.py, "-c", code, self.split, *ids], cwd=CODE, env=self.env, capture_output=True,
                                 text=True, errors="replace", timeout=600).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():  # other output (warnings, logs) may be mixed in: keep only our JSON rows
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("id") in ids and isinstance(r.get("state_hash"), str):
                rows[r["id"]] = r
        return {i: rows.get(i, {"id": i, "tool_calls": [], "state_hash": ""}) for i in ids}

    def write(self, rows: list[dict], path: Path | None = None) -> Path:
        path = path or self.submission
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return path

    def summary(self, rows: list[dict]) -> None:
        ms = sorted(r["time"] for r in rows) or [0]
        asr_phase = _span(self.t.get("asr"))
        sec_asr = 0.0
        tr = list((self.work / "runs").glob("*_asr/transcripts.jsonl"))
        if tr:
            sec_asr = sum(json.loads(x).get("sec", 0.0) for x in tr[0].open(encoding="utf-8"))
        load = max(0.0, asr_phase - sec_asr) + _span(self.t.get("vllm_start"))
        self.log(f"load model: {1000 * load:.0f} ms (PhoWhisper x2 load ~{max(0.0, asr_phase - sec_asr):.1f} s, "
                 f"vLLM Qwen3.5-4B start {_span(self.t.get('vllm_start')):.1f} s)")
        self.log(f"predict: {len(rows)} samples, total {sum(ms)} ms, mean {sum(ms) / len(ms):.0f} ms/sample, "
                 f"p50 {ms[len(ms) // 2]} ms, p95 {ms[int(0.95 * (len(ms) - 1))]} ms, max {ms[-1]} ms")
        self.log(f"phases: ASR {asr_phase:.1f} s, vLLM start {_span(self.t.get('vllm_start')):.1f} s, "
                 f"agent {_span(self.t.get('agent')):.1f} s")
        (self.work / "timing.json").write_text(json.dumps({k: v for k, v in self.t.items() if v and v[1]}) + "\n")

    def close(self) -> None:
        self.stop_llm()
        self.smi_stop.set()
        if getattr(self, "env", None):
            shutil.rmtree(self.env["LIFEMATE_CACHE"], ignore_errors=True)
        if getattr(self, "_root", None):  # removes only the link, never the mounted data
            shutil.rmtree(self._root, ignore_errors=True)
        try:  # the container runs as root: hand the outputs back to the owner of the output folder
            st = self.out.stat()
            for p in [self.out, *self.out.rglob("*")]:
                os.lchown(p, st.st_uid, st.st_gid)
        except OSError:
            pass
        self.logf.flush()


def _span(t) -> float:
    return (t[1] - t[0]) if t and t[1] else 0.0


def _smi(query: str, units: bool = True) -> str:
    try:
        fmt = "csv,noheader" if units else "csv,noheader,nounits"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"], capture_output=True,
                             text=True, timeout=10).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _is_qwen35(model_dir: Path) -> bool:
    try:
        return re.search(r'"model_type"\s*:\s*"qwen3_5_text"', (model_dir / "config.json").read_text()) is not None
    except OSError:
        return False


def _find_llm() -> str:
    """The merged Qwen3.5 checkpoint shipped in models/ (used only if LLM_MODEL is unset)."""
    found = sorted(p.parent.name for p in (CODE / "models").glob("*/config.json") if _is_qwen35(p.parent))
    if len(found) != 1:
        raise SystemExit(f"set LLM_MODEL (Qwen3.5 checkpoints in models/: {found})")
    return f"models/{found[0]}"


def predict(out: str | None = None, **kw) -> Path:
    """Full run, as used by predict.sh. Returns the submission path."""
    seed_everything(SEED)
    run = Run(out, **kw)
    t0 = time.time()
    agent_run = None
    try:
        run.prepare()
        try:
            run.run_asr()
            run.start_llm()
            agent_run = run.run_agent()
        except RuntimeError as e:
            run.log(f"ERROR: {e}")
        finally:
            run.stop_llm()
        rows = run.results(agent_run)
        path = run.write(rows)
        run.log(f"{path}: {len(rows)} lines (input {len(run.cases)} samples"
                + (f", LIMIT {run.limit}" if run.limit else "") + f"), total wall {time.time() - t0:.1f} s")
        run.summary(rows)
        if agent_run is not None:
            v = subprocess.run([run.py, "-m", "lifemate.cli", "validate", str(agent_run)], cwd=CODE, env=run.env,
                               capture_output=True, text=True)
            run.log(f"zaic_tools --validate exit {v.returncode}: {(v.stdout + v.stderr).strip()[-300:]}")
        else:
            run.log("WARNING: degraded output (pipeline error above); every sample still has a line")
        return path
    finally:
        run.close()


if __name__ == "__main__":
    predict()
