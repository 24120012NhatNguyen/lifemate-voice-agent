"""Compare actual 07e01a4 agent/guards against this checkout with scripted gold replies on CPU/train only.

Evidence covers prompts, schema requests, full traces, observations, calls and state hashes; no model inference.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from lifemate import data, guards, registry
from lifemate.agent import agents
from lifemate.config import REPO, load
from lifemate.synth.scenarios import transcript

BASELINE = '07e01a452446e5efcb86ac9cf6b3983e35aaffdb'


def baseline_module(name, path):
    code = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=REPO, text=True)
    module = ModuleType(name)
    module.__file__ = str(REPO / path)
    sys.modules[name] = module
    original = registry.register
    try:
        registry.register = lambda *_: lambda cls: cls
        exec(compile(code, module.__file__, 'exec'), module.__dict__)
    finally:
        registry.register = original
    return module


class Scripted:
    def __init__(self, sample):
        self.actions = [json.dumps({'action': 'call', 'thought': '', **c}, ensure_ascii=False)
                        for c in sample['label']['tool_calls']]
        self.actions.append('{"action":"stop","thought":""}')
        self.requests = []

    def chat(self, messages, **kw):
        self.requests.append(copy.deepcopy({'messages': messages, **kw}))
        return SimpleNamespace(content=self.actions.pop(0) if self.actions else '{"action":"stop"}'), {
            'sec': 1.0, 'prompt_tokens': 10, 'completion_tokens': 5}


def run(am, gm, sample, cfg, explicit_off=False):
    ac = copy.deepcopy(cfg['agent'])
    ac.pop('name')
    ac['decision'] = {}  # all optional inference gates OFF for this parity contract
    ac['act_after_retry'] = False
    if explicit_off:
        ac['inference_review'] = {'enabled': False, 'mode': 'verify'}
    agent = am.ReactJSON(**ac)
    agent.llm = Scripted(sample)
    env = data.tool_env('train', sample['id'])
    gc = dict(cfg['guards'])
    if explicit_off:
        gc.update({k: False for k in ('xref_guard', 'explicit_defaults', 'ask_schema', 'day_filter')})
    guard = gm.GuardedEnv(env, 'train', sample, sample['audio_text'], **gc)
    trace = agent.run('train', sample, transcript(sample), guard)
    return {'trace': trace, 'requests': agent.llm.requests, 'guard_log': guard.log,
            'calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}


def main():
    old_a = baseline_module('lifemate.agent._lm99_base', 'src/lifemate/agent/agents.py')
    old_g = baseline_module('lifemate._lm99_baseguards', 'src/lifemate/guards.py')
    original_init = agents.LLM.__init__
    agents.LLM.__init__ = lambda self, **kw: None
    cfg = load(REPO / 'configs/docker-final-268.yaml')
    totals = [hashlib.sha256() for _ in range(3)]
    samples = data.samples('train')
    try:
        for s in samples:
            outputs = [run(old_a, old_g, s, cfg), run(agents, guards, s, cfg), run(agents, guards, s, cfg, True)]
            assert outputs[0] == outputs[1] == outputs[2], s['id']
            for h, result in zip(totals, outputs):
                h.update(json.dumps(result, sort_keys=True, ensure_ascii=False).encode())
                h.update(b'\n')
    finally:
        agents.LLM.__init__ = original_init
    result = {'baseline': BASELINE, 'samples': len(samples), 'mismatches': 0,
              'sha256': dict(zip(('baseline', 'default_off', 'explicit_off'), (h.hexdigest() for h in totals))),
              'coverage': 'gold scripted replies, prompts/requests/trace/calls/observations/state; no inference or GPU'}
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
