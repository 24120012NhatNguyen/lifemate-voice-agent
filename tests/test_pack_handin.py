"""Exercise packaging with fake derived data and a disposable Git repository; no BTC files or models."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def command(repo, *args):
    return subprocess.run(args, cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def package_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docs").mkdir()
    for name in ("pack_handin.sh", "pack_handin.py"):
        shutil.copyfile(SOURCE / "scripts" / name, repo / "scripts" / name)
    (repo / "docs/REPRODUCE.md").write_text("Fake reproduction instructions\n")
    source = tmp_path / "derived"
    source.mkdir()
    content = b'{"id":"synth_00001","messages":[]}\n'
    (source / "derived.jsonl").write_bytes(content)
    # A directory scan would accidentally copy these: the allowlist must leave them alone.
    (source / "train.jsonl").write_text('{"id":"train_1","original":"BTC"}\n')
    (source / "public_test.jsonl").write_text('{"id":"public_1"}\n')
    plan = {"historical_code": [], "data_recipe": "fake derived fixture", "files": [{
        "group": "sft", "file": "derived.jsonl", "destination": "data/sft/derived.jsonl",
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "lines": 1,
        "source": "generated test fixture",
    }]}
    (repo / "docs/handin_data.json").write_text(json.dumps(plan))
    command(repo, "git", "init", "-q")
    command(repo, "git", "add", ".")
    command(repo, "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    return repo, source, tmp_path / "output"


def pack(repo, source, out):
    return subprocess.run(["bash", "scripts/pack_handin.sh", str(out), "--sft-root", str(source)],
                          cwd=repo, capture_output=True, text=True)


def test_archive_manifest_checksums_and_exclusion(package_repo):
    repo, source, out = package_repo
    result = pack(repo, source, out)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["source"] == "generated test fixture"
    assert (out / "data/sft/derived.jsonl").read_bytes() == (source / "derived.jsonl").read_bytes()
    assert not list(out.rglob("train.jsonl")) and not list(out.rglob("public_test.jsonl"))
    with tarfile.open(out / manifest["code"][0]["path"]) as archive:
        assert "lifemate/scripts/pack_handin.sh" in archive.getnames()
        assert not any(name.endswith(".jsonl") for name in archive.getnames())
    for line in (out / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert pack(repo, source, out).returncode != 0  # no overwrite


@pytest.mark.parametrize("failure", ["changed", "missing", "dirty", "untracked", "raw_in_git", "missing_commit"])
def test_failures_leave_no_partial_destination(package_repo, failure):
    repo, source, out = package_repo
    if failure == "changed":
        (source / "derived.jsonl").write_text('{"id":"public_1"}\n')
    elif failure == "missing":
        (source / "derived.jsonl").unlink()
    elif failure == "dirty":
        (repo / "docs/REPRODUCE.md").write_text("not committed")
    elif failure == "untracked":
        (repo / "untracked.py").write_text("print('missing from archive')")
    else:
        if failure == "raw_in_git":
            (repo / "train.jsonl").write_text("private raw fixture")
        else:
            path = repo / "docs/handin_data.json"
            plan = json.loads(path.read_text())
            plan["historical_code"] = [{"name": "teacher", "commit": "f" * 40, "purpose": "missing"}]
            path.write_text(json.dumps(plan))
        command(repo, "git", "add", ".")
        command(repo, "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "failure")
    assert pack(repo, source, out).returncode != 0
    assert not out.exists()
    assert not list(out.parent.glob(".handin-*"))


def test_synth_catalog_access_is_train_only_without_btc(monkeypatch):
    from lifemate import data
    from lifemate.synth.scenarios import real_app_ids

    accessed = []

    def catalog(split):
        accessed.append(split)
        assert split == "train"
        return {"apps": [{"id": "fake_train_app"}]}

    monkeypatch.setattr(data, "catalog", catalog)
    assert real_app_ids() == {"fake_train_app"}
    assert accessed == ["train"]
