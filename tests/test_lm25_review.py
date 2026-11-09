"""LM-38: CPU-only regressions; fake LLM/env, no model or contest data needed."""
import importlib.util
import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from lifemate.agent import agents
from lifemate.agent.agents import ASK, decision_probs
from lifemate.agent.llm import LLMTimeout


def lp(p):
    return math.log(p)


def dist(text, key, selected, alternatives, selected_p=0.8):
    marker = json.dumps(key) + ': "'
    start = text.index(marker) + len(marker)
    assert text[start:].startswith(selected)
    return [(text[:start], -0.01, []),
            (selected, lp(selected_p), [(t, lp(p)) for t, p in alternatives]),
            (text[start + len(selected):], -0.01, [])]


def action(name=None):
    return json.dumps({'thought': '', 'action': 'call' if name else 'stop',
                       **({'name': name, 'arguments': {}} if name else {})})


def test_sampled_token_is_included_even_if_absent_from_top_logprobs():
    text = action()
    p = decision_probs(dist(text, 'action', 'stop', [('call', 0.3)], 0.2), text, [ASK, 'app.write'])
    assert p['p_stop'] == pytest.approx(0.4)


def test_invalid_suffix_is_not_a_valid_schema_option_before_masking():
    text = action()
    p = decision_probs(dist(text, 'action', 'stop', [('stop', 0.2), ('call', 0.3),
                                                   ('calling', 0.5)], 0.2), text, [ASK, 'app.write'])
    assert p['p_stop'] == pytest.approx(0.4)


def test_ambiguous_sys_prefix_must_not_turn_a_chosen_ask_into_zero_probability():
    text = action(ASK)
    # 'sys' can continue to either ask_missing_info or calc; it cannot all be discarded
    # while the remaining 0.2 is renormalized to certain act.
    a = text.index('call')
    n = text.index('sys')
    toks = [(text[:a], -0.01, []), ('call', lp(1.0), [('call', lp(1.0))]),
            (text[a + 4:n], -0.01, []), ('sys', lp(0.8), [('sys', lp(0.8)), ('app', lp(0.2))]),
            (text[n + 3:], -0.01, [])]
    p = decision_probs(toks, text, [ASK, 'sys.calc', 'app.write'])
    assert p is not None and 'p_ask' not in p  # no fabricated confidence; threshold keeps original


def test_misaligned_decoded_tokens_do_not_trigger_a_decision():
    text = action()
    toks = dist(text, 'action', 'stop', [('stop', 0.8), ('call', 0.2)])
    # A byte-fallback token was decoded differently from message.content.
    toks[0] = ('x' + toks[0][0][1:], -0.01, [])
    assert decision_probs(toks, text, [ASK, 'app.write']) is None


class FakeLLM:
    def __init__(self, replies, env, clock=None, sec=0.0):
        self.replies, self.env = list(replies), env
        self.requests, self.clock, self.sec = [], clock, sec

    def chat(self, msgs, **kw):
        self.requests.append({'messages': list(msgs), 'calls_before': list(self.env.calls), **kw})
        if self.clock:
            self.clock.t += self.sec
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        meta = {'sec': self.sec, 'prompt_tokens': 1, 'completion_tokens': 1}
        if result == action() and kw.get('logprobs'):
            meta['logprobs'] = dist(result, 'action', 'stop', [('stop', 0.4), ('call', 0.6)], 0.4)
        elif result == action(ASK) and kw.get('logprobs'):
            a, n = result.index('call'), result.index('sys')
            meta['logprobs'] = [(result[:a], -0.01, []), ('call', lp(0.9), [('call', lp(0.9)), ('stop', lp(0.1))]),
                                (result[a + 4:n], -0.01, []),
                                ('sys', lp(0.3), [('sys', lp(0.3)), ('app', lp(0.7))]),
                                (result[n + 3:], -0.01, [])]
        return SimpleNamespace(content=result), meta


class FakeEnv:
    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return {'ok': True}


@pytest.fixture
def run_agent(monkeypatch):
    monkeypatch.setattr(agents, 'LLM', lambda **_: None)
    monkeypatch.setattr(agents.ReactJSON, 'system_prompt', lambda *a: 'system')
    monkeypatch.setattr(agents.ReactJSON, 'user_prompt', lambda *a: 'user')
    monkeypatch.setattr(agents.data, 'tool_specs', lambda split: {
        'app.read': {'kind': 'read'}, 'app.write': {'kind': 'write'}})
    sample = {'context': {'installed_apps': {'app': {'tools': ['app.read', 'app.write']}}}}

    def run(replies, transcript=None, clock=None, sec=0.0, **opts):
        a = agents.ReactJSON(**opts)
        env = FakeEnv()
        a.llm = FakeLLM(replies, env, clock, sec)
        result = a.run('public_test', sample, transcript or {'final': 'x'}, env)
        return a.llm.requests, env.calls, result['steps']
    return run


def test_regeneration_connection_error_keeps_original_instead_of_aborting(run_agent):
    err = openai.APIConnectionError(request=SimpleNamespace())
    reqs, calls, steps = run_agent([action(), err], decision={'mode': 'threshold', 'stop_min': 0.7})
    assert len(reqs) == 2 and not calls and steps[0]['action'] == 'stop'


@pytest.mark.parametrize('error', [LLMTimeout('fake'), openai.APIConnectionError(request=SimpleNamespace()), 'bad'])
def test_failed_ask_regeneration_calls_original_ask_only_after_retry(run_agent, error):
    reqs, calls, steps = run_agent([action(ASK), error, action()],
                                   decision={'mode': 'threshold', 'ask_min': 0.7})
    assert len(reqs) == 3 and not reqs[1]['calls_before']
    assert calls == [(ASK, {})] and steps[0]['raw'] == action(ASK)


@pytest.mark.parametrize('reply', [LLMTimeout('fake timeout'), 'invalid JSON', action('unknown.tool')])
def test_failed_regeneration_falls_back_once(run_agent, reply):
    reqs, calls, steps = run_agent([action(), reply], decision={'mode': 'threshold', 'stop_min': 0.7})
    assert len(reqs) == 2 and not calls and steps[0]['raw'] == action()


def test_regeneration_updates_raw_messages_and_precedes_all_env_calls(run_agent):
    reqs, calls, steps = run_agent([action(), action('app.write'), action()],
                                   decision={'mode': 'threshold', 'stop_min': 0.7})
    assert not reqs[1]['calls_before'] and reqs[1]['messages'] == reqs[0]['messages']
    assert calls == [('app.write', {})] and steps[0]['raw'] == action('app.write')
    assert reqs[2]['messages'][-2]['content'] == action('app.write')


def test_force_first_survives_parse_error_and_then_expires(run_agent):
    reqs, calls, steps = run_agent(['bad', action('app.read'), action()],
                                   decision={'force_act_first': True})
    enums = [r['response_format']['json_schema']['schema']['properties']['action']['enum'] for r in reqs]
    assert enums == [['call'], ['call'], ['call', 'stop']]
    assert steps[0]['action'] == 'parse_error' and len(calls) == 1


def test_force_first_does_not_replace_recheck_once_after_first_stop(run_agent):
    # Deliberately return stop despite the first forced schema, to exercise interaction.
    reqs, _, steps = run_agent([action(), action('app.read'), action()],
                               decision={'force_act_first': True},
                               recheck_stop={'mode': 'force', 'known': []})
    props = [r['response_format']['json_schema']['schema']['properties'] for r in reqs]
    assert ASK not in props[0]['name']['enum'] and ASK in props[1]['name']['enum']
    assert props[1]['action']['enum'] == ['call'] and props[2]['action']['enum'] == ['call', 'stop']
    assert steps[0]['recheck'] == 'force'


def test_regeneration_uses_remaining_budget_including_asr(run_agent, monkeypatch):
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(agents.time, 'perf_counter', lambda: clock.t)
    reqs, calls, _ = run_agent([action(), action('app.write')], clock=clock, sec=3,
                               transcript={'final': 'x', 'sec': 2}, max_steps=1,
                               time_budget=10, decision={'mode': 'threshold', 'stop_min': 0.7})
    assert [r['timeout'] for r in reqs] == [8, 5] and len(calls) == 1


def test_insufficient_regeneration_budget_keeps_original(run_agent, monkeypatch):
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(agents.time, 'perf_counter', lambda: clock.t)
    reqs, calls, steps = run_agent([action()], clock=clock, sec=7,
                                   transcript={'final': 'x', 'sec': 2}, time_budget=10,
                                   decision={'mode': 'threshold', 'stop_min': 0.7})
    assert len(reqs) == 1 and not calls and steps[0]['action'] == 'stop'


def sweep_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'decision_sweep.py'
    spec = importlib.util.spec_from_file_location('lm25_sweep', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_run(path, rows):
    path.mkdir()
    (path / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def test_sweep_rejects_unpaired_ids_instead_of_silently_changing_the_subset(tmp_path):
    mod = sweep_module()
    rec = write_run(tmp_path / 'rec', [{'id': 'a'}, {'id': 'b'}])
    frc = write_run(tmp_path / 'frc', [{'id': 'a'}])
    with pytest.raises(ValueError, match='id'):
        mod.Pair('val_app', rec, frc)


def test_sweep_rejects_duplicate_ids(tmp_path):
    mod = sweep_module()
    path = write_run(tmp_path / 'rec', [{'id': 'a'}, {'id': 'a'}])
    with pytest.raises(ValueError, match='id'):
        mod.load(path)


def test_sweep_app_folds_keep_all_apps_of_a_sample_together(tmp_path):
    rows = {
        'mixed': {'gold_calls': [{'name': 'video_meeting.create'}, {'name': 'medication.add'}]},
        'v': {'gold_calls': [{'name': 'video_meeting.create'}]},
        'm': {'gold_calls': [{'name': 'medication.add'}]},
        'a': {'gold_calls': [{'name': 'appliances.on'}]},
        'd': {'gold_calls': [{'name': 'door_lock.lock'}]},
    }
    samples = [{'id': i, 'score': 0.0, 'steps': [{'action': 'stop', 'p_stop': 0.9}], **r}
               for i, r in rows.items()]
    rec, frc = write_run(tmp_path / 'rec', samples), write_run(tmp_path / 'frc', samples)
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'decision_sweep.py'
    out = subprocess.run([sys.executable, str(script), '--app', rec, frc],
                         check=True, capture_output=True, text=True).stdout
    line = next(line for line in out.splitlines() if line.startswith('2-fold'))
    left, right = line.split('nửa A ', 1)[1].split(', nửa B ')
    halves = [ast.literal_eval(left), ast.literal_eval(right)]
    assert not set(halves[0]) & set(halves[1])
    for r in rows.values():
        apps = {c['name'].split('.')[0] for c in r['gold_calls']}
        assert any(apps <= set(half) for half in halves)


def test_no_decision_stays_record_and_se_is_paired(tmp_path):
    mod = sweep_module()
    rows = [{'id': 'none', 'score': 0.25, 'steps': [{'action': 'parse_error'}]},
            {'id': 'stop', 'score': 0.0, 'steps': [{'action': 'stop', 'p_stop': 0.4}]}]
    force = [{**r, 'score': 1.0} for r in rows]
    pair = mod.Pair('val_app', write_run(tmp_path / 'rec', rows), write_run(tmp_path / 'frc', force))
    assert pair.scores(0.7, 0.7) == [0.25, 1.0]
    assert pair.delta(0.7, 0.7) == pytest.approx((50, 50, 1, 1, 0))
