"""Synth × N-best: các config mới chỉ khác config gốc ở checkpoint agent; bản nộp 98s và Docker của nó chỉ khác 98n ở đó."""
import pytest

from lifemate import config, data
from lifemate.config import REPO

try:
    data.data_root()
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)


def load(name):
    return config.load(REPO / "configs" / name)


def only_model_differs(a, b, model):
    assert a["agent"] == {**b["agent"], "llm": {**b["agent"]["llm"], "model": model}}
    assert {k: v for k, v in a.items() if k not in ("agent", "name")} == {k: v for k, v in b.items() if k not in ("agent", "name")}


def test_99n_is_54n_55n_with_the_synth_checkpoint():
    only_model_differs(load("experiments/99n_qwen3-4b-sft-synth_phowhisper-large-ft-nbest4_react.yaml"),
                       load("experiments/54n_qwen3-4b-sft_phowhisper-large-ft-nbest4_react.yaml"), "models/qwen3-4b-sft-synth")
    only_model_differs(load("experiments/99n_qwen3-4b-sft-synth_phowhisper-large-ft-nbest4_react_val-app.yaml"),
                       load("experiments/55n_qwen3-4b-sft_phowhisper-large-ft-nbest4_react_val-app.yaml"), "models/qwen3-4b-sft-synth")


def test_98s_and_its_docker_config_are_98n_with_the_full_synth_checkpoint():
    only_model_differs(load("experiments/98s_submit_public_full-synth_fixes_nbest4.yaml"),
                       load("experiments/98n_submit_public_full_fixes_nbest4.yaml"), "models/qwen3-4b-sft-full-synth")
    only_model_differs(load("docker-synth-nbest.yaml"), load("docker-nbest.yaml"), "models/qwen3-4b-sft-full-synth")
    assert load("docker-synth-nbest.yaml")["asr"]["n_best"] == 4 and load("docker-synth-nbest.yaml")["agent"]["time_budget"] == 25


def test_kaggle_has_the_full_synth_experiment():
    text = (REPO / "kaggle/run_sft.py").read_text(encoding="utf-8")
    assert '"full-synth": ("train_full_synth.jsonl", [])' in text
