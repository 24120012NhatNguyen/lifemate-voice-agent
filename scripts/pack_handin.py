"""Package the reviewed code and allowlisted derived training data; never scan BTC data directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import os

# Workspace that holds this repo, fable-eval/ and fable-work/ (override with ZAIC_WORKSPACE).
WORKSPACE = Path(os.environ.get("ZAIC_WORKSPACE", Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(commit: str, name: str, out: Path) -> dict:
    sha = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    # Audit even historical archives: a source archive may not carry competition data or weights.
    paths = git("ls-tree", "-rz", "--name-only", sha).decode().split("\0")
    forbidden = {".jsonl", ".wav", ".mp3", ".flac", ".zip", ".safetensors", ".bin", ".pt", ".pth"}
    for path in filter(None, paths):
        p = PurePosixPath(path)
        if p.suffix.lower() in forbidden or p.parts[0] in {"data", "sft", "models", "cache", "runs"}:
            raise ValueError(f"source archive contains data/weights: {path}")
    filename = f"code-{name}-{sha[:12]}.tar.gz"
    git("archive", "--format=tar.gz", f"--prefix={name}/", "-o", str(out / filename), sha)
    return {"commit": sha, "path": filename, "sha256": digest(out / filename)}


def build(out: Path, roots: dict[str, Path]) -> None:
    # Reject dirty tracked changes and untracked source: git archive must contain the reviewed script itself.
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("commit source changes before packaging (git archive uses HEAD)")
    if out.exists():
        raise ValueError("destination already exists; choose a new directory")
    plan = json.loads((REPO / "docs/handin_data.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    # A failed run never leaves a destination looking like a completed hand-in.
    with tempfile.TemporaryDirectory(prefix=".handin-", dir=out.parent) as temp:
        stage = Path(temp)
        code = [archive("HEAD", "lifemate", stage)]
        for historical in plan["historical_code"]:
            code.append({**archive(historical["commit"], historical["name"], stage),
                         "source": historical["purpose"]})
        files = []
        for entry in plan["files"]:
            # Names and destinations come from committed metadata, never from a directory walk.
            name, dest = PurePosixPath(entry["file"]), PurePosixPath(entry["destination"])
            if len(name.parts) != 1 or name.name in {"", ".", ".."} or dest.is_absolute() or ".." in dest.parts:
                raise ValueError("invalid allowlist path")
            src = roots[entry["group"]] / name.name
            target = stage / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            if digest(target) != entry["sha256"] or target.stat().st_size != entry["bytes"]:
                raise ValueError(f"hash/size mismatch: {src}")
            if target.suffix == ".jsonl":
                count = 0
                with target.open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        rid = row.get("id", "")
                        if not isinstance(rid, str) or not rid.startswith(("train_", "synth_", "teach_")):
                            raise ValueError(f"unexpected sample ID in {src}")
                        count += 1
                if count != entry["lines"]:
                    raise ValueError(f"line count mismatch: {src}")
            files.append(entry)
        shutil.copyfile(REPO / "docs/REPRODUCE.md", stage / "REPRODUCE.md")
        manifest = {"schema": 1, "code": code, "files": files, "data_recipe": plan["data_recipe"],
                    "excluded": ["original BTC data", "audio", "public/private test data", "model weights"],
                    "teacher_provenance": "Original D1 MANIFEST retained byte for byte; families in REPRODUCE.md"}
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        sums = [f"{digest(p)}  {p.relative_to(stage).as_posix()}\n"
                for p in sorted(stage.rglob("*")) if p.is_file()]
        (stage / "SHA256SUMS").write_text("".join(sums))
        stage.rename(out)
    print(f"Packaged {len(files)} verified derived-data files and {len(code)} code archives in {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    p.add_argument("--sft-root", type=Path,
                   default=WORKSPACE / "fable-eval/pro-threshold/sft")
    p.add_argument("--d1-root", type=Path,
                   default=WORKSPACE / "fable-work/out/LM-41/paraphrase")
    p.add_argument("--fold-root", type=Path,
                   default=WORKSPACE / "fable-work/out/LM-51/sft")
    a = p.parse_args()
    try:
        build(a.destination.expanduser().absolute(), {"sft": a.sft_root, "d1": a.d1_root, "fold": a.fold_root})
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        p.exit(1, f"DỪNG: {error}\n")


if __name__ == "__main__":
    main()
