"""N-best trong đường suy luận và Docker của config98: bật bằng asr.n_best; tắt (n_best 1, không có alts) thì config,
prompt và cách gọi Whisper y như 98."""
import pytest

from lifemate import config, data
from lifemate.agent.agents import ReactJSON
from lifemate.asr.adapters import _distinct
from lifemate.config import REPO

try:
    data.data_root()
    SAMPLE = data.samples("train")[0]
except FileNotFoundError:
    pytest.skip("không có gói dữ liệu BTC", allow_module_level=True)


def strip(cfg, keys=("asr", "agent", "name")):
    return {k: v for k, v in cfg.items() if k not in keys}


def test_98n_differs_from_98_only_in_asr_decoding_and_alts():
    base = config.load(REPO / "configs/experiments/98_submit_public_full_fixes.yaml")
    nb = config.load(REPO / "configs/experiments/98n_submit_public_full_fixes_nbest4.yaml")
    assert nb["asr"] == {**base["asr"], "num_beams": 4, "n_best": 4, "batch_size": 2}
    assert nb["agent"] == {**base["agent"], "show_alts": True}
    assert strip(nb) == strip(base)
    assert nb["guards"] == base["guards"] and nb["agent"]["llm"]["model"] == "models/qwen3-4b-sft-full"


def test_docker_nbest_is_docker_yaml_plus_nbest():
    d = config.load(REPO / "configs/docker.yaml")
    dn = config.load(REPO / "configs/docker-nbest.yaml")
    assert strip(dn) == strip(d) and dn["workers"] == 1 and dn["subset"] == "all"
    assert dn["agent"] == {**d["agent"], "show_alts": True} and dn["agent"]["time_budget"] == 25
    assert dn["asr"]["n_best"] == 4 and d["asr"].get("n_best", 1) == 1


def test_dockerfile_and_build_script_take_config():
    assert "ARG CONFIG=configs/docker.yaml" in (REPO / "docker/Dockerfile").read_text(encoding="utf-8")
    assert 'CONFIG=${CONFIG:-configs/docker.yaml}' in (REPO / "scripts/docker_build.sh").read_text(encoding="utf-8")
    assert 'CONFIG=${CONFIG:-configs/docker.yaml}' in (REPO / "docker/predict.sh").read_text(encoding="utf-8")


def test_alts_reach_the_prompt_only_when_asr_returned_them():
    agent = ReactJSON()
    tr = {"final": "Tắt đèn bếp.", "alts": {"final": ["Tắt đèn bớt.", "Tắt đèn bếp nhé."]}}
    assert 'Bản nghe khác của cùng câu: "Tắt đèn bớt." | "Tắt đèn bếp nhé."' in agent.user_prompt(SAMPLE, tr)
    greedy = agent.user_prompt(SAMPLE, {"final": "Tắt đèn bếp."})  # n_best 1: không có alts, prompt như 98
    assert "Bản nghe khác" not in greedy and greedy == agent.user_prompt(SAMPLE, {"final": "Tắt đèn bếp.", "alts": {}})
    assert "Bản nghe khác" not in ReactJSON(show_alts=False).user_prompt(SAMPLE, tr)


def test_nbest_alts_are_distinct_in_words_and_capped_at_three():
    assert _distinct("Giao hàng nhé.", ["giao hàng nhé", "Sao hàng nhé.", "Sao hàng, nhé", "Giao hàng nha."]) == \
        ["Sao hàng nhé.", "Giao hàng nha."]
    assert _distinct("a", ["b", "c", "d", "e"]) == ["b", "c", "d"]
    assert _distinct("a", []) == []  # greedy: Result(text, []) như trước khi gộp
