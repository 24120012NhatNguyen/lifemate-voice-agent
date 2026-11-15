#!/usr/bin/env python3
"""LM-91/97: paired val comparison, averaging repeats within each sample (CPU only).

Example: python scripts/arm_compare.py --base BASE_R1 BASE_R2 --stop STOP_R1 STOP_R2 --out comparison.json
Supply separate run directories, in repeat order. runner repeats=2 only saves the first trace; rejected here.
`touched` is the union of samples with any override/activation in base or arm, in any repeat.
`arm_triggered` isolates retry, stop, the added [0.7, 0.85) ask range, or 265 review attempts. `budget_exposed` shows
base samples with >1 override (262 also reduces the shared ask/stop budget). These are descriptive,
post-treatment groups, not causal estimates. Untouched deltas help expose inference noise.
FAQ is rescored from train labels and gold state hashes; --score strict uses stored strict scores.
No inference, audio, models or public split is read. Input configs, IDs and completeness must match.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.config import REPO, load  # noqa: E402

ARMS = {'base': '260_base', 'retry': '261_retry', 'stop': '262_stop', 'ask85': '263_ask85',
        'source': '265_source', 'verify': '265_verify', 'light': '267_light'}
LIGHT_DECISIONS = ('(xref_guard)', '(ask_schema)')


def normalized(cfg):
    return {k: v for k, v in cfg.items() if k != 'name'}


def load_run(path, arm, metric='faq'):
    path = Path(path)
    cfg = load(path / 'config.yaml')
    if cfg.get('split') != 'train' or cfg.get('subset') not in ('val_app', 'val_random'):
        raise ValueError(f'{path}: only train/val_app or train/val_random is allowed')
    if cfg.get('repeats', 1) != 1:
        raise ValueError(f'{path}: repeats>1 does not save all traces; use separate runs')
    suffix = '_val-app' if cfg['subset'] == 'val_app' else ''
    expected = load(REPO / f'configs/experiments/{ARMS[arm]}_pub{suffix}.yaml')
    if normalized(cfg) != normalized(expected):
        raise ValueError(f'{path}: resolved config differs from the declared {arm} arm')
    metrics = json.loads((path / 'metrics.json').read_text())
    rows = {}
    for line in (path / 'samples.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row['id']
        if sid in rows:
            raise ValueError(f'{path}: duplicate ID {sid}')
        if 'steps' not in row or not isinstance(row['steps'], list):
            raise ValueError(f'{path}: missing steps for {sid}')
        score = row['score']
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'{path}: invalid score for {sid}')
        rows[sid] = row
    expected_n = {'val_app': 301, 'val_random': 225}[cfg['subset']]
    if len(rows) != expected_n or metrics.get('n') != len(rows):
        raise ValueError(f'{path}: incomplete run ({len(rows)}, expected {expected_n})')
    if metrics.get('run_errors') != 0 or any(r.get('error_run') for r in rows.values()):
        raise ValueError(f'{path}: inference errors; inspect before comparing')
    from lifemate import splits

    gold = {s['id']: s for s in splits.select('train', cfg['subset'])}
    if rows.keys() != gold.keys():
        raise ValueError(f'{path}: IDs differ from the declared validation subset')
    for sid, row in rows.items():
        if row['gold_calls'] != gold[sid]['label']['tool_calls']:
            raise ValueError(f'{path}: gold calls differ for {sid}')
    if metric == 'faq':
        from lifemate.runner import gold_hash
        from lifemate.scoring import score_faq

        for sid, row in rows.items():
            row = dict(row)
            row['score'] = score_faq('train', row['tool_calls'], row['state_hash'], row['gold_calls'],
                                     gold_hash('train', gold[sid]))['score']
            rows[sid] = row
    return {'path': str(path.resolve()), 'subset': cfg['subset'], 'rows': rows}


def touched(row):
    return any(s.get('decision_override') or s.get('act_after_retry')
               or (s.get('inference_review') or {}).get('attempted') for s in row['steps'])


def light_triggered(row):
    """267: rule names whose trace marks appear in the guard log (decision suffix, `defaults`, `ask_schema`)."""
    out = set()
    for rec in row.get('guard_log') or []:
        d = rec.get('decision') or ''
        out |= {m.strip('()') for m in LIGHT_DECISIONS if d.endswith(m)}
        if rec.get('defaults'):
            out.add('explicit_defaults')
        if rec.get('ask_schema'):
            out.add('ask_schema')
        if rec.get('day_filter'):
            out.add('day_filter')
    return out


def arm_triggered(row, arm):
    if arm == 'light':
        return bool(light_triggered(row))
    for s in row['steps']:
        review = s.get('inference_review') or {}
        if arm in ('source', 'verify') and review.get('mode') == arm and review.get('attempted'):
            return True
        event = s.get('decision_override') or {}
        if arm == 'retry' and s.get('act_after_retry'):
            return True
        if arm == 'stop' and event.get('from') == 'stop':
            return True
        if arm == 'ask85' and event.get('from') == 'ask' and 0.7 <= event.get('p', -1) < 0.85:
            return True
    return False


def summary(pairs, ids):
    ids = sorted(ids)
    if not ids:
        return {'n': 0, 'base': None, 'arm': None, 'delta_pp': None, 'se_pp': None,
                'wins': 0, 'losses': 0, 'ties': 0}
    base = [st.mean(b[i]['score'] for b, _ in pairs) for i in ids]
    arm = [st.mean(a[i]['score'] for _, a in pairs) for i in ids]
    delta = [100 * (a-b) for b, a in zip(base, arm)]
    return {'n': len(ids), 'base': 100 * st.mean(base), 'arm': 100 * st.mean(arm),
            'delta_pp': st.mean(delta), 'se_pp': st.stdev(delta) / math.sqrt(len(ids)) if len(ids) > 1 else None,
            'wins': sum(d > 1e-9 for d in delta), 'losses': sum(d < -1e-9 for d in delta),
            'ties': sum(abs(d) <= 1e-9 for d in delta)}


def costs(runs):
    rows = [r for run in runs for r in run['rows'].values()]
    def mean(key):
        values = [r[key] for r in rows if key in r]
        return st.mean(values) if values else None
    events = [s['inference_review'] for r in rows for s in r['steps'] if s.get('inference_review')]
    return {'samples_across_repeats': len(rows),
            'sec_agent_mean': mean('sec_agent'), 'sec_asr_mean': mean('sec_asr'),
            'prompt_tokens_mean': mean('prompt_tokens'),
            'tool_calls_mean': st.mean(len(r['tool_calls']) for r in rows),
            'review_attempts': sum(bool(e.get('attempted')) for e in events),
            'review_changes': sum(bool(e.get('changed')) for e in events),
            'review_failures': sum(e.get('status') in ('invalid', 'timeout', 'api_error') for e in events),
            'review_budget_skips': sum(e.get('status') == 'budget_skip' for e in events),
            'review_prompt_tokens': sum(e.get('prompt_tokens', 0) for e in events),
            'review_completion_tokens': sum(e.get('completion_tokens', 0) for e in events)}


def compare(base, arm_runs, arm):
    if len(base) != len(arm_runs) or len(base) not in (1, 2):
        raise ValueError('supply the same number of repeats (1 or 2) for base and each arm')
    ids = set(base[0]['rows'])
    all_runs = base + arm_runs
    if any(r['subset'] != base[0]['subset'] or set(r['rows']) != ids for r in all_runs):
        raise ValueError('subset/ID mismatch: refusing silent intersection or cross-subset pairing')
    pairs = [(b['rows'], a['rows']) for b, a in zip(base, arm_runs)]
    hit = {i for b, a in pairs for i in ids if touched(b[i]) or touched(a[i])}
    specific = {i for _, a in pairs for i in ids if arm_triggered(a[i], arm)}
    budget = {i for b, _ in pairs for i in ids
              if sum(bool(s.get('decision_override')) for s in b[i]['steps']) > 1} if arm == 'stop' else set()
    groups = {'all': ids, 'touched': hit, 'untouched': ids-hit, 'arm_triggered': specific,
              'budget_exposed': budget}
    if arm == 'light':  # per rule: samples where that rule left a mark in any arm repeat
        for rule in ('xref_guard', 'explicit_defaults', 'ask_schema', 'day_filter'):
            groups[f'rule:{rule}'] = {i for _, a in pairs for i in ids if rule in light_triggered(a[i])}
        groups['light_untriggered'] = ids - specific
    samples = []
    for i in sorted(ids):
        bs = [b[i]['score'] for b, _ in pairs]
        ars = [a[i]['score'] for _, a in pairs]
        samples.append({'id': i, 'base_scores': bs, 'arm_scores': ars,
                        'delta_pp': 100 * (st.mean(ars)-st.mean(bs)),
                        'touched': i in hit, 'arm_triggered': i in specific, 'budget_exposed': i in budget})
    return {'arm': arm, 'subset': base[0]['subset'], 'repeats': len(pairs),
            'paths': [{'base': b['path'], 'arm': a['path']} for b, a in zip(base, arm_runs)],
            'groups': {g: summary(pairs, group) for g, group in groups.items()},
            'per_repeat': [{g: summary([pair], group) for g, group in groups.items()} for pair in pairs],
            'base_repeat_noise': summary([(pairs[0][0], pairs[1][0])], ids) if len(pairs) == 2 else None,
            'arm_repeat_noise': summary([(pairs[0][1], pairs[1][1])], ids) if len(pairs) == 2 else None,
            'samples': samples, 'costs': {'base': costs(base), 'arm': costs(arm_runs)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', nargs='+', required=True, type=Path)
    for arm in ('retry', 'stop', 'ask85', 'source', 'verify', 'light'):
        ap.add_argument(f'--{arm}', nargs='+', type=Path)
    ap.add_argument('--score', choices=['faq', 'strict'], default='faq')
    ap.add_argument('--out', type=Path, help='JSON evidence including per-sample paired scores')
    args = ap.parse_args(argv)
    arms = {arm: getattr(args, arm) for arm in ('retry', 'stop', 'ask85', 'source', 'verify', 'light') if getattr(args, arm)}
    if not arms:
        ap.error('at least one non-base arm is required')
    try:
        paths = args.base + [p for runs in arms.values() for p in runs]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError('each input must be a distinct run (do not repeat a path as repeat 2)')
        if any(len(runs) != len(args.base) for runs in arms.values()) or len(args.base) not in (1, 2):
            raise ValueError('supply matching counts of 1 or 2 repeats')
        base = [load_run(p, 'base', args.score) for p in args.base]
        results = [compare(base, [load_run(p, arm, args.score) for p in paths], arm)
                   for arm, paths in arms.items()]
    except (OSError, ValueError, KeyError) as exc:
        ap.error(str(exc))
    report = {'metric': args.score, 'units': '0–100 points',
              'method': 'Average repeats per ID, then paired mean and SE across IDs; not 2N independent samples.',
              'caveats': ['Touched is union of any activation/override in either arm across repeats.',
                          'arm_triggered is branch-specific; groups are descriptive, not causal estimates.',
                          '262 changes shared max_overrides 2→1; inspect budget_exposed as well as stop triggers.',
                          'SE does not capture all generation variance; compare repeat_noise and untouched.',
                          '265 reviews may change correct actions; compare wins/losses and review costs.',
                          'Legacy llm_calls counts steps, not extra generations; review attempts are reported separately.'],
              'comparisons': results}
    if len(base) == 1:
        report['caveats'].append('Only one repeat is available; generation variance is unresolved.')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Metric: {args.score}; {base[0]['subset']}; repeats: {len(base)}; scores / deltas in points")
    for r in results:
        for group, s in r['groups'].items():
            fmt = lambda v: 'n/a' if v is None else f'{v:.3f}'
            print(f"{r['arm']:6} {group:15} n={s['n']:3} base={fmt(s['base'])} arm={fmt(s['arm'])} "
                  f"delta={fmt(s['delta_pp'])} SE={fmt(s['se_pp'])} W/L/T={s['wins']}/{s['losses']}/{s['ties']}")
    for caveat in report['caveats']:
        print('NOTE:', caveat)
    return report


if __name__ == '__main__':
    main()
