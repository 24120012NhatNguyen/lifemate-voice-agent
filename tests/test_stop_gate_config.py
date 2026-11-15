"""LM-91: load every inherited config; isolate arms and freeze public-form inputs."""
from copy import deepcopy

import pytest

from lifemate.config import REPO, load

ROOT = REPO / 'configs/experiments'
GUARDS = dict(validate=True, normalize=True, calendar_duration=True, id_guard='warn',
              id_sources=['turn', 'history', 'utterance'], id_harvest='all', dup_guard='warn',
              **dict.fromkeys(['clock_fix', 'money_k', 'span_min', 'ask_missing', 'pick_one', 'phone_fix',
                               'rel_day_fix', 'now_offset_fix', 'clock_face', 'ask_resolve', 'empty_retry',
                               'calendar_create_end'], True))


@pytest.mark.parametrize('suffix,subset', [('', 'val_random'), ('_val-app', 'val_app')])
def test_all_arms_are_public_shaped_and_only_change_declared_flags(suffix, subset):
    base = load(ROOT / f'260_base_pub{suffix}.yaml')
    assert base['split'] == 'train' and base['subset'] == subset
    # Worker count is an operational override (Pro uses 8 locally); every arm must inherit it.
    assert type(base['workers']) is int and base['workers'] > 0
    assert base['limit'] is None and base['repeats'] == 1
    assert base['asr'] == dict(name='hf_whisper', history=True, hint='none',
                               model='models/phowhisper-large-ft', num_beams=4, batch_size=2, n_best=4,
                               merge_with=dict(model='models/phowhisper-large-ft2', num_beams=1,
                                               batch_size=16, hint='profile_history_apps'))
    assert base['guards'] == GUARDS
    assert base['agent']['llm']['model'] == 'models/qwen35-4b-soup-175-175s1'
    assert base['agent']['decision'] == dict(record=True, mode='threshold', scope='pre_write',
                                            ask_min=0.7, later_ask_min=0.7, stop_min=0.0, later_stop_min=0.0,
                                            max_overrides=2, ask_apps='unseen', known='train_rest',
                                            stop_scope='pre_write', unseen_only=False)
    for arm in ['261_retry', '262_stop', '263_ask85']:
        cfg = load(ROOT / f'{arm}_pub{suffix}.yaml')
        expected = deepcopy(base)
        expected['name'] = cfg['name']
        if arm == '261_retry':
            expected['agent']['act_after_retry'] = True
        elif arm == '262_stop':
            expected['agent']['decision'].update(stop_scope='any', later_stop_min=0.8, max_overrides=1)
        else:
            expected['agent']['decision'].update(ask_min=0.85, later_ask_min=0.85)
        assert cfg == expected


def test_the_two_subsets_differ_only_in_name_and_subset():
    for arm in ['260_base', '261_retry', '262_stop', '263_ask85']:
        random = load(ROOT / f'{arm}_pub.yaml')
        app = load(ROOT / f'{arm}_pub_val-app.yaml')
        random.update(name=app['name'], subset='val_app')
        assert random == app
