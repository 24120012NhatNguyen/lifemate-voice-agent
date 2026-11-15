"""Pairing, repeat averaging, trigger attribution and input integrity for LM-91."""
import importlib.util
import json

import pytest
import yaml

from lifemate import splits
from lifemate.config import REPO, load

spec = importlib.util.spec_from_file_location('arm_compare', REPO / 'scripts/arm_compare.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def row(score=0, events=()):
    return dict(score=score, steps=list(events), gold_calls=[], tool_calls=[], state_hash='x')


def run(rows, path='run', subset='val_app'):
    return dict(rows=rows, path=path, subset=subset)


def test_pair_by_id_average_repeats_and_use_union_of_touched():
    stop = {'decision_override': {'from': 'stop', 'p': 0.4}}
    b1 = run({'a': row(0), 'b': row(1)})
    b2 = run({'b': row(0), 'a': row(0)})
    a1 = run({'b': row(0), 'a': row(1, [stop])})
    a2 = run({'a': row(0), 'b': row(0)})
    result = ac.compare([b1, b2], [a1, a2], 'stop')
    assert result['groups']['all'] == dict(n=2, base=25, arm=25, delta_pp=0, se_pp=50,
                                          wins=1, losses=1, ties=0)
    assert result['groups']['touched']['delta_pp'] == 50
    assert result['groups']['touched']['n'] == result['groups']['arm_triggered']['n'] == 1
    assert result['groups']['untouched']['delta_pp'] == -50
    assert result['base_repeat_noise']['delta_pp'] == -50
    assert result['per_repeat'][1]['touched']['n'] == 1  # union, even though not triggered on repeat 2


def test_shared_ask_budget_and_new_ask_range_are_not_confused_with_stop():
    ask = lambda p: {'decision_override': {'from': 'ask', 'p': p}}
    b = run({'a': row(1, [ask(0.3), ask(0.5)]), 'b': row(0), 'c': row(0)})
    a = run({'a': row(1, [ask(0.3)]), 'b': row(1, [ask(0.8)]), 'c': row(1, [ask(0.7)])})
    r = ac.compare([b], [a], 'stop')
    assert r['groups']['touched']['n'] == 3
    assert r['groups']['arm_triggered']['n'] == 0
    assert r['groups']['arm_triggered']['delta_pp'] is None
    assert r['groups']['budget_exposed']['n'] == 1
    assert ac.compare([b], [a], 'ask85')['groups']['arm_triggered']['n'] == 2
    assert not ac.arm_triggered(row(1, [ask(0.85)]), 'ask85')
    assert ac.arm_triggered(row(1, [{'act_after_retry': True}]), 'retry')


@pytest.mark.parametrize('arms', [[run({'wrong': row()})], [run({'a': row()}, subset='val_random')],
                                  [run({'a': row()}), run({'a': row()})]])
def test_mismatches_are_not_silently_intersected(arms):
    with pytest.raises(ValueError):
        ac.compare([run({'a': row()})], arms, 'stop')


@pytest.fixture
def run_file(tmp_path, monkeypatch):
    gold = [{'id': f'train_{i}', 'label': {'tool_calls': []}} for i in range(301)]
    monkeypatch.setattr(splits, 'select', lambda split, subset: gold)

    def make(arm='base'):
        path = tmp_path / arm
        path.mkdir(exist_ok=True)
        cfg = load(REPO / f'configs/experiments/{ac.ARMS[arm]}_pub_val-app.yaml')
        (path / 'config.yaml').write_text(yaml.safe_dump(cfg))
        (path / 'metrics.json').write_text(json.dumps(dict(n=301, run_errors=0)))
        rows = [dict(id=s['id'], **row()) for s in gold]
        (path / 'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path
    return make


def test_cli_accepts_one_or_two_complete_repeats(run_file, tmp_path, capsys):
    base, arm = run_file(), run_file('stop')
    out = tmp_path / 'result.json'
    r = ac.main(['--base', str(base), '--stop', str(arm), '--score', 'strict', '--out', str(out)])
    assert r['comparisons'][0]['groups']['all']['n'] == 301
    assert json.loads(out.read_text()) == r
    assert 'Only one repeat' in capsys.readouterr().out
    import shutil
    base2, arm2 = tmp_path / 'base2', tmp_path / 'arm2'
    shutil.copytree(base, base2)
    shutil.copytree(arm, arm2)
    r = ac.main(['--base', str(base), str(base2), '--stop', str(arm), str(arm2), '--score', 'strict'])
    assert r['comparisons'][0]['repeats'] == 2


@pytest.mark.parametrize('corruption', ['ids', 'duplicate', 'missing', 'errors', 'config', 'repeats', 'score',
                                        'gold', 'steps', 'public'])
def test_load_rejects_bad_runs_before_reporting_scores(run_file, corruption):
    path = run_file()
    samples = path / 'samples.jsonl'
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    cfg = yaml.safe_load((path / 'config.yaml').read_text())
    if corruption == 'ids':
        rows[0]['id'] = 'wrong'
    elif corruption == 'duplicate':
        rows.append(rows[0])
    elif corruption == 'missing':
        rows.pop()
    elif corruption == 'errors':
        rows[0]['error_run'] = 'timeout'
    elif corruption == 'config':
        cfg['asr'].pop('merge_with')
    elif corruption == 'repeats':
        cfg['repeats'] = 2
    elif corruption == 'score':
        rows[0]['score'] = float('nan')
    elif corruption == 'gold':
        rows[0]['gold_calls'] = [{'name': 'changed'}]
    elif corruption == 'steps':
        rows[0].pop('steps')
    else:
        cfg['split'] = 'public_test'
    (path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        ac.load_run(path, 'base', 'strict')


def test_faq_rescores_instead_of_reusing_strict_score(run_file, monkeypatch):
    from lifemate import runner, scoring
    path = run_file()
    monkeypatch.setattr(runner, 'gold_hash', lambda split, sample: 'gold')
    calls = []

    def score(*args):
        calls.append(args)
        return {'score': 0.5}
    monkeypatch.setattr(scoring, 'score_faq', score)
    r = ac.load_run(path, 'base')
    assert all(s['score'] == 0.5 for s in r['rows'].values()) and len(calls) == 301
    assert calls[0] == ('train', [], 'x', [], 'gold')


def test_cli_rejects_duplicate_paths_and_missing_repeat(tmp_path):
    with pytest.raises(SystemExit):
        ac.main(['--base', str(tmp_path), '--stop', str(tmp_path)])
    with pytest.raises(SystemExit):
        ac.main(['--base', 'one', 'two', '--stop', 'three'])


@pytest.mark.parametrize('arm', ['source', 'verify'])
def test_review_arms_load_compare_and_report_attempts_not_just_changes(run_file, arm, tmp_path):
    base, treatment = run_file(), run_file(arm)
    samples = treatment / 'samples.jsonl'
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    rows[0]['steps'] = [{'inference_review': {'mode': arm, 'attempted': True, 'changed': False,
                                            'status': 'accepted', 'prompt_tokens': 100}}]
    rows[1]['steps'] = [{'inference_review': {'mode': arm, 'attempted': False, 'changed': False,
                                            'status': 'budget_skip'}}]
    rows[2]['steps'] = [{'inference_review': {'mode': arm, 'attempted': True, 'changed': False,
                                            'status': 'invalid', 'prompt_tokens': 110}}]
    samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = ac.main(['--base', str(base), f'--{arm}', str(treatment), '--score', 'strict'])
    r = result['comparisons'][0]
    assert r['groups']['arm_triggered']['n'] == r['groups']['touched']['n'] == 2
    assert r['costs']['arm']['review_attempts'] == 2
    assert r['costs']['arm']['review_failures'] == 1
    assert r['costs']['arm']['review_budget_skips'] == 1
    assert r['costs']['arm']['review_prompt_tokens'] == 210
