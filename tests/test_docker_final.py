"""CPU packaging checks: reject stale/partial/extra weights and validate opt-in settings."""
import ast
import importlib.util
import json
import struct
import sys

import pytest

from lifemate.config import REPO, load
from lifemate.guards import DEFAULTS

sys.path.insert(0, str(REPO / 'scripts'))
import docker_final as final  # noqa: E402


def tensor_file(path, name='model.weight', count=2):
    header = json.dumps({name: {'dtype': 'F32', 'shape': [count], 'data_offsets': [0, count * 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\x00' * count * 4)


def test_baseline_contract_and_flags_off():
    cfg = load(REPO / 'configs/docker-final.yaml')
    assert final.normalized(cfg) == final.normalized(load(final.BASE))
    assert all(DEFAULTS[k] is False and not cfg['guards'].get(k) for k in final.LIGHT)
    assert not cfg['agent'].get('inference_review', {}).get('enabled')
    assert not cfg['agent'].get('calc')
    assert cfg['agent']['act_after_retry'] is True
    assert cfg['agent']['decision']['stop_scope'] == 'any'
    assert cfg['agent']['decision']['later_stop_min'] == .8
    assert cfg['agent']['decision']['max_overrides'] == 1


def test_flags_and_no_training_lookup():
    base = load(final.BASE)
    cfg = final.flags_config(base, ['guards.day_filter=true agent.calc=true',
                                   'agent.inference_review.enabled=true agent.inference_review.mode=verify'])
    assert cfg['guards']['day_filter'] and cfg['agent']['calc']
    assert cfg['agent']['inference_review']['known'] == base['agent']['decision']['known']
    assert 'inference_review' not in base['agent']


@pytest.mark.parametrize('value', ['guards.day_filte=true', 'guards.day_filter=1',
                                 'agent.decision.later_stop_min=nan', 'agent.decision.max_overrides=true',
                                 'agent.inference_review.mode=oops', 'asr.model=other', 'workers=99',
                                 'agent.decision.ask_min=1.01'])
def test_invalid_flags_rejected(value):
    with pytest.raises(ValueError):
        final.flags_config(load(final.BASE), [value])


def test_header_validated_and_no_missing_weight_false_pass(tmp_path):
    with pytest.raises(ValueError, match='no safetensors'):
        final.inspect_weights(tmp_path)
    tensor_file(tmp_path / 'model.safetensors')
    assert final.inspect_weights(tmp_path)['tensor_count'] == 1
    p = tmp_path / 'model.safetensors'
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload'):
        final.inspect_weights(tmp_path)


def test_extra_adapter_or_nested_checkpoint_rejected(tmp_path):
    tensor_file(tmp_path / 'model.safetensors')
    adapter = tmp_path / 'adapter_config.json'
    adapter.write_text('{}')
    with pytest.raises(ValueError, match='adapter'):
        final.inspect_weights(tmp_path)
    adapter.unlink()
    (tmp_path / 'old').mkdir()
    tensor_file(tmp_path / 'old/model.safetensors')
    with pytest.raises(ValueError, match='unaccounted'):
        final.inspect_weights(tmp_path)


def test_index_must_cover_every_tensor(tmp_path):
    tensor_file(tmp_path / 'model.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'wrong': 'model.safetensors'}}))
    with pytest.raises(ValueError, match='index'):
        final.inspect_weights(tmp_path)


def test_stale_receipt_is_rejected(tmp_path, monkeypatch):
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text('changed')
    rec = tmp_path / 'receipt.json'
    rec.write_text('{"config_sha256":"stale"}')
    monkeypatch.setattr(final, 'CONFIG', cfg)
    monkeypatch.setattr(final, 'RECEIPT', rec)
    with pytest.raises(ValueError, match='changed'):
        final.current_sources({})


def test_masked_assistant_preserves_input_and_masks_entire_body():
    # Extract pure encoder functions; do not import trainer, torch, or start training.
    tree = ast.parse((REPO / 'scripts/train_sft_q35r.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('trained', 'encode', 'encode_scaffold')]
    namespace = {"SCAFFOLD": "<think>\n\n</think>\n\n"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<encoders>', 'exec'), namespace)
    class Tokenizer:
        def encode(self, text, **kw):
            return list(text.encode())
    tok = Tokenizer()
    messages = [{'role': 'user', 'content': 'read'}, {'role': 'assistant', 'content': 'bad'},
                {'role': 'user', 'content': 'note'}, {'role': 'assistant', 'content': 'good'}]
    for name in ('encode', 'encode_scaffold'):
        clean = namespace[name](tok, messages)
        explicit = namespace[name](tok, [{**m, 'train': True} for m in messages])
        masked = namespace[name](tok, [{**m, **({'train': False} if i == 1 else {})} for i, m in enumerate(messages)])
        assert clean == explicit
        assert clean[0] == masked[0]
        assert sum(v != -100 for v in clean[1]) - sum(v != -100 for v in masked[1]) == len(b'bad<|im_end|>')
        assert any(v != -100 for v in masked[1])


def test_submission_builders_can_import_without_external_worktrees():
    for file in ('build_v2.py', 'build_guard_traces.py', 'build_rename_v2.py'):
        spec = importlib.util.spec_from_file_location('_final_' + file[:-3], REPO / 'scripts' / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
