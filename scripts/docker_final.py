"""CPU-only final candidate selection and exact model build-context inventory (LM-99)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

import yaml

from lifemate.config import REPO, load
from docker_report import param_table

CONFIG = REPO / 'configs/docker-final.yaml'
BASE = REPO / 'configs/docker-final-268.yaml'
RECEIPT = REPO / 'artifacts/docker/final-cpu.json'
LIGHT = ('xref_guard', 'explicit_defaults', 'ask_schema', 'day_filter')
FLAGS = {**{f'guards.{k}': bool for k in LIGHT}, 'agent.calc': bool, 'agent.act_after_retry': bool,
         'agent.inference_review.enabled': bool, 'agent.inference_review.mode': ('source', 'verify'),
         'agent.decision.stop_scope': ('pre_write', 'any'), 'agent.decision.later_stop_min': float,
         'agent.decision.ask_min': float, 'agent.decision.later_ask_min': float,
         'agent.decision.max_overrides': int}
ASR = 'models/phowhisper-large-base'
ADAPTERS = ['models/phowhisper-large-ft-all-lora', 'models/phowhisper-large-ft2-all-lora']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flags_config(raw, values):
    cfg = copy.deepcopy(raw)
    for text in values:
        for item in shlex.split(text):
            key, sep, value = item.partition('=')
            if not sep or key not in FLAGS:
                raise ValueError(f'unsupported flag: {item}; use dotted keys: {", ".join(FLAGS)}')
            value = yaml.safe_load(value)
            typ = FLAGS[key]
            if isinstance(typ, tuple):
                valid = value in typ
            elif typ is float:
                valid = type(value) in (int, float) and 0 <= value <= 1
            elif typ is int:
                valid = type(value) is int and 0 <= value <= 2
            else:
                valid = type(value) is typ
            if not valid:
                raise ValueError(f'invalid flag value: {item}')
            node = cfg
            *head, tail = key.split('.')
            for k in head:
                node = node.setdefault(k, {})
            node[tail] = value
    # A final image contains no training dataset. Never resolve known apps from data at runtime.
    review = cfg['agent'].get('inference_review', {})
    if review.get('enabled'):
        review['known'] = list(cfg['agent']['decision']['known'])
    return cfg


def normalized(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop('name', None)
    cfg['agent']['llm'].pop('model', None)
    for a in (cfg['asr'], cfg['asr']['merge_with']):
        a.pop('model', None)
        a.pop('adapter', None)
    return cfg


def changes(a, b, prefix=''):
    out = {}
    for k in sorted(a.keys() | b.keys()):
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out.update(changes(a[k], b[k], path))
        elif a.get(k) != b.get(k):
            out[path] = {'268': a.get(k), 'candidate': b.get(k)}
    return out


def inspect_weights(path, adapter=False):
    """Validate the entire copied directory, indices, offsets and file sizes without loading tensors."""
    path = Path(path).resolve(strict=True)
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError(f'no safetensors: {path}')
    for p in path.rglob('*'):
        if p.is_file() and (p.suffix in ('.bin', '.pt', '.pth', '.gguf', '.safetensors')):
            if p not in files:
                raise ValueError(f'extra/unaccounted weights in build context: {p}')
    if not adapter and list(path.glob('adapter_*')):
        raise ValueError(f'unmerged/extra adapter in model: {path}')
    index = path / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    entries, headers = {}, {}
    widths = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    for p in files:
        with p.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if not 0 < size < 64 * 1024 * 1024:
                raise ValueError(f'invalid header length: {p}')
            blob = f.read(size)
        header = json.loads(blob)
        headers[p.name] = {'header_sha256': hashlib.sha256(blob).hexdigest(), 'bytes': p.stat().st_size}
        end = 0
        for name, value in sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                                  key=lambda kv: kv[1]['data_offsets']):
            if name in entries:
                raise ValueError(f'duplicate tensor {name}: {path}')
            start, stop = value['data_offsets']
            count = 1
            for dim in value['shape']:
                if type(dim) is not int or dim < 0:
                    raise ValueError(f'bad shape: {name}')
                count *= dim
            if start != end or stop - start != count * widths[value['dtype']]:
                raise ValueError(f'invalid tensor offsets: {p}/{name}')
            entries[name], end = p.name, stop
        if not header or p.stat().st_size != 8 + size + end:
            raise ValueError(f'truncated/extra payload: {p}')
    if mapping is not None and mapping != entries:
        raise ValueError(f'index does not cover exactly all tensors/shards: {path}')
    if mapping is None and len(files) > 1:
        raise ValueError(f'multiple shards require index: {path}')
    if not adapter and any('lora_' in k or 'visual.' in k for k in entries):
        raise ValueError(f'LoRA/vision tensors in merged model: {path}')
    if adapter:
        cfg = json.loads((path / 'adapter_config.json').read_text())
        if cfg.get('peft_type') != 'LORA' or not all('lora_' in k for k in entries):
            raise ValueError(f'not a plain ASR LoRA: {path}')
    return {'source': str(path), 'files': headers, 'tensor_count': len(entries)}


def check(cfg, sources, app_rank):
    if cfg['asr']['model'] != ASR or cfg['asr']['merge_with']['model'] != ASR:
        raise ValueError('both ASR passes must use the original PhoWhisper base')
    if [cfg['asr']['adapter'], cfg['asr']['merge_with']['adapter']] != ADAPTERS:
        raise ValueError('final ASR requires exactly ft-all + ft2-all, no ASR soup')
    if cfg['asr']['merge_with']['hint'] != 'profile_history_apps':
        raise ValueError('secondary ASR requires profile_history_apps')
    if cfg['agent']['llm']['temperature'] != 0:
        raise ValueError('temperature must be zero')
    if cfg['agent'].get('inference_review', {}).get('enabled'):
        if not isinstance(cfg['agent']['inference_review'].get('known'), list):
            raise ValueError('inference review must have an explicit known-app list')
    expected = [ASR, cfg['agent']['llm']['model'], *ADAPTERS]
    if len(set(expected)) != 4 or set(sources) != set(expected):
        raise ValueError('build contexts must contain exactly base, merged LLM and two adapters')
    inventory = {k: inspect_weights(sources[k], k in ADAPTERS) for k in expected}
    llm = Path(sources[cfg['agent']['llm']['model']])
    if json.loads((llm / 'config.json').read_text()).get('model_type') != 'qwen3_5_text':
        raise ValueError('candidate must be Qwen3.5 text-only')
    for p in (llm / 'tokenizer.json', llm / 'tokenizer_config.json',
              Path(sources[ASR]) / 'config.json', Path(sources[ASR]) / 'preprocessor_config.json'):
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'missing required file: {p}')
    table = param_table([str(Path(sources[k]).resolve()) for k in expected])
    if len(table['models']) != 4 or not table['under_limit'] or any(r['params'] <= 0 for r in table['models'].values()):
        raise ValueError(f'invalid parameter budget: {table}')
    rank = json.loads(Path(app_rank).read_text())
    if not isinstance(rank, dict) or not rank or not all(isinstance(k, str) and type(v) in (int, float) for k, v in rank.items()):
        raise ValueError('app rank must be a nonempty numeric mapping')
    for p in ('docker/predict.sh', 'scripts/docker_build.sh', 'scripts/docker_swap.sh',
              'scripts/docker_cpu_host_check.sh', 'scripts/docker_cpu_check.sh', 'scripts/serve_vllm.sh'):
        subprocess.run(['bash', '-n', str(REPO / p)], check=True)
    return {'status': 'CPU_PASS', 'parameter_table': table, 'inventory': inventory,
            'app_rank': {'source': str(Path(app_rank).resolve()), 'sha256': digest(app_rank), 'entries': len(rank)},
            'config_delta_from_268_excluding_paths': changes(normalized(load(BASE)), normalized(cfg)),
            'docker_runtime_overrides': {'workers': 1, 'agent.time_budget': 25},
            'limitations': 'Headers and syntax only; no payload hash, image build, GPU, inference or latency test.'}


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        f.write(content)
    os.replace(f.name, path)


def current_sources(cfg):
    if RECEIPT.exists():
        rec = json.loads(RECEIPT.read_text())
        if rec.get('config_sha256') != digest(CONFIG):
            raise ValueError('config changed since CPU receipt; rerun docker_swap.sh')
        return {k: v['source'] for k, v in rec['inventory'].items()}, rec['app_rank']['source']
    keys = [ASR, cfg['agent']['llm']['model'], *ADAPTERS]
    return {k: str(REPO / k) for k in keys}, str(REPO / 'models/app_rank_train.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    swap = sub.add_parser('swap')
    swap.add_argument('model_dir', type=Path)
    swap.add_argument('flags', nargs='*', help='quoted space-separated dotted.key=value flags')
    sub.add_parser('check')
    sub.add_parser('build-env')
    a = ap.parse_args()
    if a.command == 'swap':
        model = a.model_dir.resolve(strict=True)
        if not model.is_dir() or any(c.isspace() for c in model.name):
            raise ValueError('model must be a directory with no whitespace in its basename')
        cfg = flags_config(load(BASE), a.flags)
        cfg['name'] = 'docker-final'
        cfg['agent']['llm']['model'] = f'models/{model.name}'
        cfg['asr'].update(model=ASR, adapter=ADAPTERS[0])
        cfg['asr']['merge_with'].update(model=ASR, adapter=ADAPTERS[1])
        sources = {k: str(REPO / k) for k in [ASR, *ADAPTERS]}
        sources[cfg['agent']['llm']['model']] = str(model)
        app_rank = str(REPO / 'models/app_rank_train.json')
        report = check(cfg, sources, app_rank)  # fail before replacing the existing candidate
        content = '# Generated by docker_swap.sh; base 268; optional flags reset on every swap.\n'
        content += yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        atomic_write(CONFIG, content)
        report['config_sha256'] = digest(CONFIG)
        atomic_write(RECEIPT, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    else:
        cfg = load(CONFIG)
        sources, app_rank = current_sources(cfg)
        report = check(cfg, sources, app_rank)
    if a.command == 'build-env':
        values = {'VARIANT': 'full2', 'CONFIG': 'configs/docker-final.yaml', 'ASR': sources[ASR],
                  'LLM': sources[cfg['agent']['llm']['model']],
                  'ADAPTERS': ' '.join(sources[k] for k in ADAPTERS), 'APP_RANK': app_rank}
        for k, v in values.items():
            print(f'export {k}={shlex.quote(v)}')
    else:
        print(json.dumps({'status': report['status'], 'total': report['parameter_table']['total'],
                          'margin': report['parameter_table']['limit'] - report['parameter_table']['total'],
                          'models': list(report['inventory']),
                          'delta_268': report['config_delta_from_268_excluding_paths']}, ensure_ascii=False, indent=2))
        if a.command == 'swap':
            print(f'Receipt: {RECEIPT}\nNext (manager): scripts/docker_build.sh')


if __name__ == '__main__':
    main()
