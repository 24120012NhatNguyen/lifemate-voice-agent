"""LM-97 CPU contract tests with scripted LLM; no data/model/network required."""
import copy
import json
from types import SimpleNamespace

import pytest
from openai import APIError

from lifemate import data, splits
from lifemate.agent import agents
from lifemate.agent.llm import LLMTimeout
from lifemate.agent.review import ASK, InferenceReview
from lifemate.config import REPO, load

SAMPLE = {'context': {'installed_apps': {'new': {'tools': ['new.read', 'new.write']}}}}
SPECS = {'new.read': {'kind': 'read'}, 'new.write': {'kind': 'write'}, ASK: {'kind': 'meta'}}
STOP = {'action': 'stop', 'thought': ''}
READ = {'action': 'call', 'name': 'new.read', 'arguments': {}, 'thought': ''}
WRITE = {'action': 'call', 'name': 'new.write', 'arguments': {'id': 'one'}, 'thought': ''}
ASKING = {'action': 'call', 'name': ASK, 'arguments': {'tool_name': 'new.write', 'params': ['id']}}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, **kw):
        self.requests.append(copy.deepcopy({'messages': messages, **kw}))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        raw = json.dumps(reply, ensure_ascii=False) if isinstance(reply, dict) else reply
        return SimpleNamespace(content=raw), {'sec': 1., 'prompt_tokens': 10, 'completion_tokens': 5}


class Env:
    def __init__(self, retry=False):
        self.calls, self.retry_hit, self.retry = [], False, retry

    def call(self, name, args):
        self.calls.append((name, args))
        self.retry_hit = self.retry and len(self.calls) == 1
        return {'ok': True, 'id': 'one'}


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    monkeypatch.setattr(agents.LLM, '__init__', lambda self, **kw: None)
    monkeypatch.setattr(agents.LLMAgent, 'system_prompt', lambda *a: 'schema')
    monkeypatch.setattr(agents.LLMAgent, 'user_prompt', lambda *a: 'request')
    monkeypatch.setattr(data, 'tool_specs', lambda split: SPECS)
    monkeypatch.setattr(splits, 'known_apps', lambda subset: {'known'})


def execute(replies, cfg=None, env=None, **kw):
    a = agents.ReactJSON(inference_review=cfg, **kw)
    a.llm = ScriptedLLM(replies)
    env = env or Env()
    trace = a.run('train', SAMPLE, {'final': 'request'}, env)
    assert not a.llm.replies
    return trace, env.calls, a.llm.requests


def enabled(mode='source', **kw):
    return {'enabled': True, 'mode': mode, **kw}


def test_absent_empty_disabled_have_identical_trace_calls_and_requests(monkeypatch):
    # Off must not even resolve training membership.
    monkeypatch.setattr(splits, 'known_apps', lambda *_: pytest.fail('off read known apps'))
    replies = [READ, WRITE, ASKING, STOP]
    expected = execute(replies)
    for cfg in ({}, {'enabled': False}, {'enabled': False, 'mode': 'verify'}):
        assert execute(replies, cfg) == expected
    assert not any('inference_review' in s for s in expected[0]['steps'])


def test_source_fixes_initial_routing_before_env_and_later_ask_uses_observations():
    trace, calls, req = execute([WRITE, READ, ASKING, WRITE, STOP], enabled())
    assert calls == [('new.read', {}), ('new.write', {'id': 'one'})]
    assert len(req) == 5
    assert req[1]['messages'][-2]['content'] == json.dumps(WRITE, ensure_ascii=False)
    assert 'KẾT QUẢ new.read' in req[3]['messages'][-3]['content']
    assert all('CHƯA' not in m['content'] for m in req[-1]['messages'])
    assert trace['steps'][0]['inference_review']['changed']
    assert trace['steps'][1]['sec'] == 2
    assert trace['steps'][1]['prompt_tokens'] == 20


def test_verify_does_not_review_read_and_can_reject_wrong_conditional_write():
    trace, calls, req = execute([READ, WRITE, STOP], enabled('verify'))
    assert calls == [('new.read', {})] and len(req) == 3
    assert 'inference_review' not in trace['steps'][0]
    assert trace['steps'][1]['action'] == 'stop'
    assert trace['steps'][1]['inference_review']['proposal_raw'] == json.dumps(WRITE, ensure_ascii=False)


def test_verify_corrects_value_and_leaves_proposal_unexecuted():
    corrected = {**WRITE, 'arguments': {'id': 'two'}}
    trace, calls, _ = execute([WRITE, corrected, STOP, STOP], enabled('verify'))
    assert calls == [('new.write', {'id': 'two'})]
    assert trace['steps'][0]['inference_review']['changed']
    assert not trace['steps'][1]['inference_review']['changed']


@pytest.mark.parametrize('mode', ['source', 'verify'])
def test_stop_can_be_kept_or_replaced_without_forcing_action(mode):
    t, calls, req = execute([STOP, STOP], enabled(mode))
    assert not calls and t['steps'][0]['inference_review']['status'] == 'accepted'
    assert 'stop' in req[1]['response_format']['json_schema']['schema']['properties']['action']['enum']
    _, calls, _ = execute([STOP, READ, STOP, STOP], enabled(mode))
    assert calls == [('new.read', {})]


def test_kept_ask_is_not_reviewed_again_and_unchanged_action_keeps_original_bytes():
    new_thought = {**ASKING, 'thought': 'confirmed'}
    trace, calls, req = execute([ASKING, new_thought, STOP], enabled())
    assert len(req) == 3 and calls[0][0] == ASK
    assert trace['steps'][0]['raw'] == json.dumps(ASKING)
    assert not trace['steps'][0]['inference_review']['changed']
    assert 'inference_review' not in trace['steps'][1]


@pytest.mark.parametrize('bad', ['invalid', [], {'action': 'call', 'name': 'new.unknown', 'arguments': {}},
                               {'action': 'call', 'name': 'new.write', 'arguments': []},
                               {'action': 'call', 'name': 'new.write'}, {'action': 'bad'}])
def test_invalid_review_falls_back_and_counts_attempt(bad):
    trace, calls, _ = execute([WRITE, bad, STOP], enabled(max_reviews=1))
    assert calls == [('new.write', {'id': 'one'})]
    assert trace['steps'][0]['inference_review']['status'] == 'invalid'
    assert trace['steps'][0]['prompt_tokens'] == 20
    assert 'inference_review' not in trace['steps'][1]


@pytest.mark.parametrize('exc,status', [(LLMTimeout('timeout'), 'timeout'),
    (APIError('failed', request=SimpleNamespace(method='POST', url='http://test'), body=None), 'api_error')])
def test_review_exception_falls_back_without_retry_or_side_effect(exc, status):
    trace, calls, req = execute([WRITE, exc, STOP], enabled(max_reviews=1))
    assert len(req) == 3 and calls == [('new.write', {'id': 'one'})]
    event = trace['steps'][0]['inference_review']
    assert event['status'] == status and event['attempted'] and not event['changed']


def test_existing_force_and_retry_masks_cannot_be_bypassed():
    trace, calls, req = execute([WRITE, ASKING, STOP], enabled(max_reviews=1),
                               decision={'force_act_first': True})
    assert calls == [('new.write', {'id': 'one'})]
    assert trace['steps'][0]['inference_review']['status'] == 'invalid'
    assert req[1]['response_format'] == req[0]['response_format']
    trace, calls, req = execute([READ, WRITE, STOP, STOP, STOP], enabled('verify'),
                               env=Env(retry=True), act_after_retry=True)
    assert calls == [('new.read', {}), ('new.write', {'id': 'one'})]
    assert trace['steps'][1]['inference_review']['status'] == 'invalid'
    assert req[2]['response_format'] == req[1]['response_format']


def test_budget_skip_and_fresh_remaining_timeout(monkeypatch):
    # t0=0, initial left=9; review left=1 < min_step_sec, original stop preserved.
    ticks = iter([0, 1, 9])
    monkeypatch.setattr(agents.time, 'perf_counter', lambda: next(ticks))
    trace, calls, req = execute([STOP], enabled(), time_budget=10)
    assert len(req) == 1 and not calls
    assert trace['steps'][0]['inference_review']['status'] == 'budget_skip'
    assert not trace['steps'][0]['inference_review']['attempted']
    ticks = iter([0, 1, 3, 3])
    trace, _, req = execute([STOP, STOP], enabled(), time_budget=10)
    assert req[0]['timeout'] == 9 and req[1]['timeout'] == 7


def test_known_and_empty_apps_do_not_activate_and_state_is_per_run(monkeypatch):
    monkeypatch.setattr(splits, 'known_apps', lambda _: {'new'})
    assert execute([STOP], enabled())[0]['steps'] == execute([STOP])[0]['steps']
    assert not InferenceReview(enabled(known=[])).sample_enabled(
        {'context': {'installed_apps': {'sys': {'tools': [ASK]}, 'empty': {'tools': []}}}})
    a = agents.ReactJSON(inference_review=enabled(known=[]))
    traces = []
    for _ in range(2):
        a.llm = ScriptedLLM([STOP, STOP])
        traces.append(a.run('train', SAMPLE, {}, Env()))
    assert traces[0] == traces[1]


@pytest.mark.parametrize('cfg', [True, {'enabled': 'false'}, {'mode': 'typo'}, {'max_reviews': 3},
                                {'max_reviews': -1}, {'max_reviews': True}, {'known': 'val_app'}])
def test_bad_config_is_rejected(cfg):
    with pytest.raises(ValueError):
        agents.ReactJSON(inference_review=cfg)


def test_enabled_requires_schema_and_zero_budget_is_off():
    with pytest.raises(ValueError):
        agents.ReactJSON(inference_review=enabled(), json_mode='off')
    assert execute([STOP], enabled(max_reviews=0)) == execute([STOP])


@pytest.mark.parametrize('mode', ['source', 'verify'])
@pytest.mark.parametrize('suffix', ['', '_val-app'])
def test_265_resolves_to_260_plus_only_review_flag(mode, suffix):
    root = REPO / 'configs/experiments'
    base = load(root / f'260_base_pub{suffix}.yaml')
    arm = load(root / f'265_{mode}_pub{suffix}.yaml')
    assert arm.pop('name').startswith('265_')
    base.pop('name')
    assert arm['agent'].pop('inference_review') == enabled(mode, max_reviews=2, known='train_rest')
    assert base == arm and arm['split'] == 'train'
    assert 'inference_review' not in base['agent']


def test_threshold_override_mask_survives_review(monkeypatch):
    original = ScriptedLLM.chat

    def with_probs(self, messages, **kw):
        msg, meta = original(self, messages, **kw)
        if kw.get('logprobs'):
            meta['logprobs'] = ['stub']
        return msg, meta

    monkeypatch.setattr(ScriptedLLM, 'chat', with_probs)
    monkeypatch.setattr(agents, 'decision_probs', lambda *a: {'p_stop': 0.1, 'p_ask': 0.2})
    trace, calls, req = execute([ASKING, READ, STOP, STOP], enabled(max_reviews=1),
                               decision={'mode': 'threshold', 'ask_min': 0.7})
    assert calls == [('new.read', {})]
    assert trace['steps'][0]['decision_override']['from'] == 'ask'
    assert trace['steps'][0]['inference_review']['status'] == 'invalid'
    assert req[1]['response_format'] == req[2]['response_format']
    assert trace['steps'][0]['prompt_tokens'] == 30
