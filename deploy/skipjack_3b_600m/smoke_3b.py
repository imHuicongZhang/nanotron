#!/usr/bin/env python
"""Derive the smoke-test configs for the 3B grid, and summarize the smoke logs (no GPU needed for either).

    python deploy/skipjack_3b_600m/smoke_3b.py derive    --workdir <fresh dir>   # writes <workdir>/cfg/*.yaml
    python deploy/skipjack_3b_600m/smoke_3b.py summarize --workdir <dir>         # -> <workdir>/smoke_report.json

Phases (all on real tokenized data, one 4 x H100 node, run in order by smoke_3b.sbatch):
  init      1 step with save_initial_state from the production quality-first trunk1 config; its step-0 directory
            becomes the grid's shared init (<init_root>/_init_3B_seed42/0 + init_manifest.json) if none exists yet
  a_trunk   layout A (production), 20 steps from a copy of the init, checkpoint at 20
  a_resume  layout A, resumes the same folder to 30 with optimizer + LR state            -> save/resume
  a_branch  layout A, starts from step dir 20 into a separate folder, decays 20 -> 30 to 0 -> branch mechanics
  a_straight layout A, 30 uninterrupted steps from the init (no checkpoint)  -> reference for a_trunk + a_resume
  a_div / a_da   layout A, 3 steps each on the other two corpora from the init           -> data iteration
  b_trunk   layout B (fallback), 10 steps                                                 -> memory / throughput
  c_trunk   layout C (optional, mbs 32 = the 1.5B grid's micro-batch), 10 steps             -> does it fit?
The smoke configs differ from production only in run names, folders, train_steps, checkpoint cadence and a 5-step
warmup (so the decay branch is exercised within 30 steps). A smoke test does not establish the best learning rate.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_configs_3b as G  # noqa: E402
import registry_3b as R  # noqa: E402

PHASES = ['init', 'a_trunk', 'a_resume', 'a_branch', 'a_straight', 'a_div', 'a_da', 'b_trunk']
OPTIONAL = ['c_trunk']   # measured last; an OOM here is a finding, not a smoke failure


def derive(work: Path):
    if (work / 'cfg').exists():
        sys.exit(f'{work}/cfg exists; use a fresh workdir')
    (work / 'cfg').mkdir(parents=True)
    init = R.INIT_ROOT / f'_init_3B_seed{R.SEED}' / '0'

    def cfg(name, setting, layout, steps, ckpt, resume, *, load_opt, decay=None, interval=100000, save_init=False,
            save_final=True):
        c = G.build(setting, 'trunk1', R.LAYOUTS[layout])
        c['general']['run'] = f'3b_smoke_{name}'
        c['tokens']['train_steps'] = steps
        lrs = c['optimizer']['learning_rate_scheduler']
        lrs['lr_warmup_steps'] = 5
        lrs['lr_decay_starting_step'], lrs['lr_decay_steps'] = decay or (10 ** 6, 1)
        c['checkpoints'].update(checkpoints_path=str(ckpt), resume_checkpoint_path=str(resume) if resume else None,
                                load_optimizer=load_opt, load_lr_scheduler=load_opt, checkpoint_interval=interval,
                                save_initial_state=save_init, save_final_state=save_final)
        (work / 'cfg' / f'{name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))

    def trunk(name):    # smoke_3b.sbatch seeds <trunk>/0 from the init + latest.txt, as the submitter does
        return work / 'ckpt' / name

    cfg('init', 'quality-first', 'A', 1, work / 'ckpt' / 'init', None, load_opt=False, save_init=True)
    cfg('a_trunk', 'quality-first', 'A', 20, trunk('a'), trunk('a'), load_opt=False, interval=20)
    cfg('a_resume', 'quality-first', 'A', 30, trunk('a'), trunk('a'), load_opt=True)
    cfg('a_branch', 'quality-first', 'A', 30, work / 'ckpt' / 'a_branch', trunk('a') / '20', load_opt=True, decay=(20, 10))
    cfg('a_straight', 'quality-first', 'A', 30, trunk('straight'), trunk('straight'), load_opt=False, save_final=False)
    cfg('a_div', 'diversity-oriented', 'A', 3, trunk('div'), trunk('div'), load_opt=False)
    cfg('a_da', 'disagreement-aware', 'A', 3, trunk('da'), trunk('da'), load_opt=False)
    cfg('b_trunk', 'quality-first', 'B', 10, trunk('b'), trunk('b'), load_opt=False)
    cfg('c_trunk', 'quality-first', 'C', 10, trunk('c'), trunk('c'), load_opt=False)
    print(f'wrote {len(PHASES) + len(OPTIONAL)} smoke configs to {work}/cfg (init source: {init})')


STEP_RE = re.compile(r'iteration: (\d+) / \d+ .*?consumed_tokens: ([\d.]+)([KMB]?) .*?time_per_iteration_ms: ([\d.]+)(K?) '
                     r'.*?grad_norm: ([\w.+-]+) \| lm_loss: ([\w.+-]+) \| lr: ([\w.+-]+)')
UNIT = {'': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9}


def expected_lr(work: Path, ph: str, step: int) -> float:
    """The lr nanotron logs at iteration `step` (the next update's lr, lambda(step)) for the phase's config."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lr_schedule_report import _builder
    import torch
    build, LRArgs = _builder()
    c = yaml.safe_load((work / 'cfg' / f'{ph}.yaml').read_text())
    a = LRArgs(**c['optimizer']['learning_rate_scheduler'])
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate)
    opt.get_base_optimizer = lambda: opt
    return a.learning_rate * build(opt, a, c['tokens']['train_steps']).lr_lambdas[0](step)


def summarize(work: Path):
    rep = {'workdir': str(work), 'phases': {}, 'problems': []}
    for ph in PHASES + OPTIONAL:
        log = work / 'logs' / f'{ph}.log'
        if not log.is_file():
            if ph not in OPTIONAL:
                rep['problems'].append(f'{ph}: no log')
            continue
        steps = []
        for m in STEP_RE.finditer(log.read_text(errors='replace')):
            ms = float(m.group(4)) * (1000 if m.group(5) else 1)
            steps.append({'step': int(m.group(1)), 'tokens': float(m.group(2)) * UNIT[m.group(3)], 's': ms / 1000,
                          'grad_norm': float(m.group(6)), 'loss': float(m.group(7)), 'lr': float(m.group(8))})
        nv = work / 'logs' / f'{ph}.nvsmi.csv'
        peak = max((int(r.split(',')[1]) for r in nv.read_text().splitlines() if r.strip()), default=None) if nv.is_file() else None
        timed = [s['s'] for s in steps[2:]]
        rec = {'steps_logged': len(steps), 'first': steps[0] if steps else None, 'last': steps[-1] if steps else None,
               's_per_it_median': statistics.median(timed) if timed else None,
               'peak_mib': peak, 'peak_gib': round(peak / 1024, 1) if peak else None}
        if ph not in OPTIONAL and any(not math.isfinite(s['loss']) or not math.isfinite(s['grad_norm']) for s in steps):
            rep['problems'].append(f'{ph}: non-finite loss or grad_norm')
        if peak and peak / 1024 > 74.5 and ph not in OPTIONAL:
            rep['problems'].append(f'{ph}: peak {peak / 1024:.1f} GiB above the 74.5 GiB budget')
        for st in steps:   # logged with 3 significant digits
            want = expected_lr(work, ph, st['step'])
            if abs(st['lr'] - want) > max(5e-3 * want, 1e-12):
                rep['problems'].append(f'{ph}: step {st["step"]} logged lr {st["lr"]:.3g}, scheduler says {want:.4g}')
                break
            if abs(st['tokens'] - st['step'] * R.TOKENS_PER_STEP) > 5e-3 * st['step'] * R.TOKENS_PER_STEP:
                rep['problems'].append(f'{ph}: step {st["step"]} consumed_tokens {st["tokens"]:.3g} != step x {R.TOKENS_PER_STEP}')
                break
        rec['steps'] = steps
        rep['phases'][ph] = rec
    a = rep['phases']
    if a.get('a_resume', {}).get('first') and a['a_resume']['first']['step'] != 21:
        rep['problems'].append(f'a_resume started at step {a["a_resume"]["first"]["step"]}, expected 21')
    if a.get('a_branch', {}).get('last'):
        if a['a_branch']['first']['step'] != 21 or a['a_branch']['last']['lr'] > 1e-12:
            rep['problems'].append('a_branch did not start at 21 or did not decay to lr 0 at step 30')
    by = {ph: {st['step']: st for st in a.get(ph, {}).get('steps', [])} for ph in ('a_trunk', 'a_resume', 'a_straight')}
    if by['a_straight']:   # save/resume vs uninterrupted, same init and data; bf16 kernels are not bitwise deterministic
        noise = [abs(by['a_trunk'][n]['loss'] - by['a_straight'][n]['loss']) for n in by['a_trunk'] if n in by['a_straight']]
        diff = [abs(by['a_resume'][n]['loss'] - by['a_straight'][n]['loss']) for n in by['a_resume'] if n in by['a_straight']]
        lr_eq = all(by['a_resume'][n]['lr'] == by['a_straight'][n]['lr'] for n in by['a_resume'] if n in by['a_straight'])
        rep['resume_vs_uninterrupted'] = {'max_abs_loss_diff_steps_1_20': max(noise, default=None),
                                          'max_abs_loss_diff_steps_21_30': max(diff, default=None), 'lr_identical_21_30': lr_eq}
        unit = max(10 ** (math.floor(math.log10(by['a_straight'][n]['loss'])) - 2) for n in by['a_straight'])  # 3 sig. digits
        if len(diff) != 10 or not lr_eq or max(diff) > max(0.02, unit, 3 * max(noise, default=0)):
            rep['problems'].append(f'resume differs from the uninterrupted run: {rep["resume_vs_uninterrupted"]}')
    for need in ('a/20', 'a/30', 'a_branch/30'):
        if not (work / 'ckpt' / need / 'model_config.json').is_file():
            rep['problems'].append(f'missing checkpoint ckpt/{need}')
    if (work / 'ckpt' / 'a_branch' / '20').exists() or (work / 'ckpt' / 'a' / 'latest.txt').read_text().strip() != '30':
        rep['problems'].append('branch wrote into the trunk folder or the trunk latest.txt moved')
    (work / 'smoke_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print('SMOKE OK' if not rep['problems'] else f'SMOKE PROBLEMS: {len(rep["problems"])}')
    return 0 if not rep['problems'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['derive', 'summarize'])
    ap.add_argument('--workdir', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == 'derive':
        derive(args.workdir)
        return 0
    return summarize(args.workdir)


if __name__ == '__main__':
    sys.exit(main())
