"""FAQ compares only gold keys; all other scoring rules remain unchanged."""
import pytest

from lifemate import data
from lifemate.scoring import score_faq, score_sample


@pytest.fixture(autouse=True)
def specs(monkeypatch):
    monkeypatch.setattr(data, "fuzzy_params", lambda _: {"demo": {"title"}})
    monkeypatch.setattr(data, "tool_kind", lambda *_: "write")


def call(**args):
    return {"name": "demo", "arguments": args}


def test_extra_keys_only_ignored_by_faq():
    gold, pred = [call(start="10")], [call(start="10", end="11")]
    assert score_sample("train", pred, "h", gold, "h")["score"] == 0
    assert score_faq("train", pred, "h", gold, "h")["score"] == 1
    assert score_faq("train", pred, "wrong", gold, "h")["score"] == 0


@pytest.mark.parametrize("pred", [call(), call(start="11"), call(start=10),
                                  {"name": "other", "arguments": {"start": "10"}}])
def test_gold_parameters_and_tool_still_required(pred):
    assert score_faq("train", [pred], "h", [call(start="10")], "h")["score"] == 0


def test_missing_null_is_not_present_null():
    assert score_faq("train", [call()], "h", [call(start=None)], "h")["score"] == 0
    assert score_faq("train", [call(start=None, end="11")], "h", [call(start=None)], "h")["score"] == 1


def test_fuzzy_boundary_is_strict():
    gold = [call(title="a b c d e")]
    assert score_faq("train", [call(title="a b c x y", extra=1)], "h", gold, "h")["score"] == 0
    assert score_faq("train", [call(title="a b c d x", extra=1)], "h", gold, "h")["score"] == 1


def test_one_to_one_matching_and_empty_gold():
    gold = [call(start="10")]
    pred = [call(start="10", end="11")] * 2
    assert score_faq("train", pred, "h", gold, "h")["score"] == pytest.approx(2 / 3)
    assert score_faq("train", [], "h", [], "h")["score"] == 1
    assert score_faq("train", pred, "h", [], "h")["score"] == 0
