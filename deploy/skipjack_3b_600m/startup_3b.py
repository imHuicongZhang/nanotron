#!/usr/bin/env python
"""Bounded startup validation inside the real 3B run (replaces the separate smoke cycle; owner authorization 2026-10-20).

    python deploy/skipjack_3b_600m/startup_3b.py derive      --workdir <fresh dir>    # <workdir>/cfg/*.yaml
    python deploy/skipjack_3b_600m/startup_3b.py summarize   --workdir <dir>          # -> <workdir>/startup_report.json
    python deploy/skipjack_3b_600m/startup_3b.py seed-trunk  --config <trunk1.yaml>   # init -> <trunk>/0 + latest.txt
    python deploy/skipjack_3b_600m/startup_3b.py resume-config --config <trunk1.yaml> --out <yaml>
    python deploy/skipjack_3b_600m/startup_3b.py check-log   --config <seg.yaml> --log <seg log> [--json]

Phases (startup_3b.sbatch, one 4 x H100 node, all on quality-first's real data and real trunk folder):
  init        only if the shared init does not exist yet (init_3b.sh, under a lock shared with segment_3b.sbatch)
  s_trunk     the rendered production trunk1 config with train_steps 20: writes trunk/20 (real progress)
  s_resume    the same config with train_steps 30 and optimizer + LR state loaded: resumes trunk/20, writes trunk/30
  s_straight  30 uninterrupted steps from the init in a scratch folder (nothing saved): reference for the resume
The production trunk1 job then resumes trunk/30 (segment_3b.sbatch loads optimizer + LR state whenever latest > 0),
so the validated steps are kept. Only train_steps, the load flags and (for the reference) the folders differ from the
rendered config; the LR schedule has explicit warmup/decay steps, so it does not depend on train_steps.

Checks: finite loss and grad_norm; every logged lr equals nanotron's scheduler for the PRODUCTION config; consumed
tokens = step x 2,097,152 (data position); s_resume starts at 21; resume vs uninterrupted loss within
max(0.02, display rounding, 3 x the steps-1..20 noise) with identical lr; trunk/20 and trunk/30 written, latest.txt 30;
peak GPU memory <= 74.5 GiB.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry_3b as R  # noqa: E402

CFG_DIR = R.REPO / 'configs' / '3B-rewrite-600m' / f'seed{R.SEED}'
QF_TRUNK1 = CFG_DIR / f'{R.seg_name("quality-first", "trunk1")}.yaml'
INIT_PARENT = R.INIT_ROOT / f'_init_3B_seed{R.SEED}'
PHASES = ['s_trunk', 's_resume', 's_straight']
MEM_BUDGET_GIB = 74.5

STEP_RE = re.compile(r'iteration: (\d+) / \d+ .*?consumed_tokens: ([\d.]+)([KMBT]?) .*?time_per_iteration_ms: ([\d.]+)(K?) '
                     r'.*?grad_norm: ([\w.+-]+) \| lm_loss: ([\w.+-]+) \| lr: ([\w.+-]+)')
UNIT = {'': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}


def _dump(c: dict, path: Path):
    path.write_text(yaml.safe_dump(c, sort_keys=False))


def derive(work: Path):
    if (work / 'cfg').exists():
        sys.exit(f'{work}/cfg exists; use a fresh workdir')
    (work / 'cfg').mkdir(parents=True)
    prod = yaml.safe_load(QF_TRUNK1.read_text())
    a, s = R.STARTUP_STEPS

    c = yaml.safe_load(QF_TRUNK1.read_text())
    c['tokens']['train_steps'] = a
    _dump(c, work / 'cfg' / 's_trunk.yaml')

    c = yaml.safe_load(QF_TRUNK1.read_text())
    c['tokens']['train_steps'] = s
    c['checkpoints'].update(load_optimizer=True, load_lr_scheduler=True)
    _dump(c, work / 'cfg' / 's_resume.yaml')

    c = yaml.safe_load(QF_TRUNK1.read_text())
    c['general']['run'] = '3b_startup_straight_reference'
    c['tokens']['train_steps'] = s
    c['checkpoints'].update(checkpoints_path=str(work / 'ckpt' / 'straight'), resume_checkpoint_path=str(work / 'ckpt' / 'straight'),
                            checkpoint_interval=100000, save_final_state=False)
    _dump(c, work / 'cfg' / 's_straight.yaml')
    assert prod['checkpoints']['load_optimizer'] is False
    print(f'wrote {PHASES} to {work}/cfg from {QF_TRUNK1}')


def derive_init(out: Path, ckpt: Path):
    """1 step with save_initial_state from the quality-first trunk1 config; its step 0 becomes the shared init."""
    c = yaml.safe_load(QF_TRUNK1.read_text())
    c['general']['run'] = '3b_init_seed42'
    c['tokens']['train_steps'] = 1
    c['checkpoints'].update(checkpoints_path=str(ckpt), resume_checkpoint_path=None, load_optimizer=False,
                            load_lr_scheduler=False, checkpoint_interval=100000, save_initial_state=True, save_final_state=False)
    _dump(c, out)


def seed_trunk(cfg: Path) -> str:
    """Seed <trunk>/0 from the shared init (digest-checked) if the trunk has no latest.txt; never over progress."""
    import capped_submit_3b as C   # init_digest / seed_trunk_3b (the 1.5B tool's seeding rule, digest-checked)
    c = yaml.safe_load(cfg.read_text())
    s = type('S', (), {'ckpt': Path(c['checkpoints']['checkpoints_path'])})()
    if not C.INIT_MANIFEST.is_file():
        sys.exit(f'no {C.INIT_MANIFEST}: create the init first (init_3b.sh)')
    return 'seeded' if C.seed_trunk_3b({}, R.SEED, s) else 'already seeded'


def resume_config(cfg: Path, out: Path):
    """trunk1 restarted past step 0 must load optimizer + LR state + metadata (data position), like trunk2..6.
    The rendered trunk1 is weights-only (correct only from the step-0 init): resuming it from trunk/<n> would restart
    Adam, the warmup and the data stream. Only the two load flags change."""
    c = yaml.safe_load(cfg.read_text())
    assert c['general']['run'].endswith('_trunk1'), cfg
    c['checkpoints'].update(load_optimizer=True, load_lr_scheduler=True)
    _dump(c, out)


@functools.lru_cache(maxsize=None)
def _lambda(cfg: str):
    from lr_schedule_report import _builder
    import torch
    build, LRArgs = _builder()
    c = yaml.safe_load(Path(cfg).read_text())
    a = LRArgs(**c['optimizer']['learning_rate_scheduler'])
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate)
    opt.get_base_optimizer = lambda: opt
    return a.learning_rate, build(opt, a, c['tokens']['train_steps']).lr_lambdas[0]


def expected_lr(cfg: Path, step: int) -> float:
    """The lr nanotron logs at iteration `step` (lambda(step), the next update's lr) for that config."""
    peak, lam = _lambda(str(cfg))
    return peak * lam(step)


def parse_log(text: str) -> list[dict]:
    steps = []
    for m in STEP_RE.finditer(text):
        ms = float(m.group(4)) * (1000 if m.group(5) else 1)
        steps.append({'step': int(m.group(1)), 'tokens': float(m.group(2)) * UNIT[m.group(3)], 's': ms / 1000,
                      'grad_norm': float(m.group(6)), 'loss': float(m.group(7)), 'lr': float(m.group(8))})
    return steps


def step_problems(steps: list[dict], sched_cfg: Path, label: str) -> list[str]:
    out = []
    for st in steps:
        if not (math.isfinite(st['loss']) and math.isfinite(st['grad_norm'])):
            out.append(f'{label}: step {st["step"]} non-finite loss {st["loss"]} / grad_norm {st["grad_norm"]}')
            break
        want = expected_lr(sched_cfg, st['step'])
        if abs(st['lr'] - want) > max(5e-3 * want, 1e-12):   # logged with 3 significant digits
            out.append(f'{label}: step {st["step"]} logged lr {st["lr"]:.4g}, scheduler says {want:.4g}')
            break
        if abs(st['tokens'] - st['step'] * R.TOKENS_PER_STEP) > 5e-3 * st['step'] * R.TOKENS_PER_STEP:
            out.append(f'{label}: step {st["step"]} consumed_tokens {st["tokens"]:.4g} != step x {R.TOKENS_PER_STEP}')
            break
    return out


def peak_gib(nvsmi: Path):
    if not nvsmi.is_file():
        return None
    vals = [int(r.split(',')[1]) for r in nvsmi.read_text().splitlines() if r.count(',') >= 2 and r.split(',')[1].strip().isdigit()]
    return round(max(vals) / 1024, 2) if vals else None


def summarize(work: Path) -> int:
    rep = {'workdir': str(work), 'production_config': str(QF_TRUNK1), 'phases': {}, 'problems': []}
    a, s = R.STARTUP_STEPS
    trunk = R.ckpt_dir('quality-first', 'trunk1')
    for ph in PHASES:
        log = work / 'logs' / f'{ph}.log'
        if not log.is_file():
            rep['problems'].append(f'{ph}: no log')
            continue
        steps = parse_log(log.read_text(errors='replace'))
        rep['problems'] += step_problems(steps, QF_TRUNK1, ph)
        timed = [x['s'] for x in steps[2:]]
        pk = peak_gib(work / 'logs' / f'{ph}.nvsmi.csv')
        if pk is not None and pk > MEM_BUDGET_GIB:
            rep['problems'].append(f'{ph}: peak {pk} GiB above the {MEM_BUDGET_GIB} GiB budget')
        rep['phases'][ph] = {'steps_logged': len(steps), 'first_step': steps[0]['step'] if steps else None,
                             'last_step': steps[-1]['step'] if steps else None,
                             's_per_it_median': statistics.median(timed) if timed else None, 'peak_gib': pk, 'steps': steps}
    P = rep['phases']
    want_n = {'s_trunk': (1, a), 's_resume': (a + 1, s), 's_straight': (1, s)}
    for ph, (f, l) in want_n.items():
        if ph in P and (P[ph]['first_step'], P[ph]['last_step'], P[ph]['steps_logged']) != (f, l, l - f + 1):
            rep['problems'].append(f'{ph}: logged steps {P[ph]["first_step"]}..{P[ph]["last_step"]} '
                                   f'({P[ph]["steps_logged"]}), expected {f}..{l}')
    if all(ph in P for ph in PHASES):
        by = {ph: {x['step']: x for x in P[ph]['steps']} for ph in PHASES}
        noise = [abs(by['s_trunk'][n]['loss'] - by['s_straight'][n]['loss']) for n in by['s_trunk'] if n in by['s_straight']]
        diff = [abs(by['s_resume'][n]['loss'] - by['s_straight'][n]['loss']) for n in by['s_resume'] if n in by['s_straight']]
        lr_eq = all(by['s_resume'][n]['lr'] == by['s_straight'][n]['lr'] for n in by['s_resume'] if n in by['s_straight'])
        tok_eq = all(by['s_resume'][n]['tokens'] == by['s_straight'][n]['tokens'] for n in by['s_resume'] if n in by['s_straight'])
        losses = [x['loss'] for x in by['s_straight'].values()]
        unit = max(10 ** (math.floor(math.log10(v)) - 2) for v in losses) if losses else 0
        tol = max(0.02, unit, 3 * max(noise, default=0))
        rep['resume_vs_uninterrupted'] = {'max_abs_loss_diff_steps_1_20': max(noise, default=None),
                                          'max_abs_loss_diff_steps_21_30': max(diff, default=None), 'tolerance': tol,
                                          'lr_identical': lr_eq, 'consumed_tokens_identical': tok_eq}
        if len(diff) != s - a or not lr_eq or not tok_eq or max(diff) > tol:
            rep['problems'].append(f'resume differs from the uninterrupted run: {rep["resume_vs_uninterrupted"]}')
        timed = [x['s'] for x in P['s_trunk']['steps'][2:]] + [x['s'] for x in P['s_resume']['steps'][2:]]
        rep['s_per_it_median_production_layout'] = statistics.median(timed) if timed else None
    for n in (a, s):
        if not (trunk / str(n) / 'model_config.json').is_file():
            rep['problems'].append(f'missing checkpoint {trunk}/{n}')
    lt = trunk / 'latest.txt'
    if not lt.is_file() or lt.read_text().strip() != str(s):
        rep['problems'].append(f'{lt} is {lt.read_text().strip() if lt.is_file() else "missing"}, expected {s}')
    rep['peak_gib'] = max((p['peak_gib'] or 0) for p in P.values()) if P else None
    rep['ok'] = not rep['problems']
    (work / 'startup_report.json').write_text(json.dumps(rep, indent=1))
    brief = {k: v for k, v in rep.items() if k != 'phases'}
    brief['phases'] = {ph: {k: v for k, v in p.items() if k != 'steps'} for ph, p in P.items()}
    print(json.dumps(brief, indent=1))
    print('STARTUP OK' if rep['ok'] else f'STARTUP PROBLEMS: {len(rep["problems"])}')
    return 0 if rep['ok'] else 1


def check_log(cfg: Path, log: Path, as_json: bool) -> int:
    """Health of a running/finished production segment log: finite, lr == scheduler, tokens == step x T, s/it."""
    steps = parse_log(log.read_text(errors='replace')) if log.is_file() else []
    probs = step_problems(steps, cfg, log.name)
    tail = [x['s'] for x in steps[-50:]][2:] if len(steps) > 2 else []
    r = {'log': str(log), 'steps_logged': len(steps), 'last': steps[-1] if steps else None,
         's_per_it_median_last50': statistics.median(tail) if tail else None, 'problems': probs}
    print(json.dumps(r) if as_json else json.dumps(r, indent=1))
    return 1 if probs else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['derive', 'derive-init', 'summarize', 'seed-trunk', 'resume-config', 'check-log'])
    ap.add_argument('--workdir', type=Path)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--log', type=Path)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    if args.command == 'derive':
        derive(args.workdir)
    elif args.command == 'derive-init':
        derive_init(args.out, args.ckpt)
    elif args.command == 'summarize':
        return summarize(args.workdir)
    elif args.command == 'seed-trunk':
        print(seed_trunk(args.config))
    elif args.command == 'resume-config':
        resume_config(args.config, args.out)
    elif args.command == 'check-log':
        return check_log(args.config, args.log, args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
