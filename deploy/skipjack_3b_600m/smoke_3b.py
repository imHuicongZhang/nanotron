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
  a_div / a_da   layout A, 3 steps each on the other two corpora from the init           -> data iteration
  b_trunk   layout B (fallback), 10 steps                                                 -> memory / throughput
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

PHASES = ['init', 'a_trunk', 'a_resume', 'a_branch', 'a_div', 'a_da', 'b_trunk']


def derive(work: Path):
    if (work / 'cfg').exists():
        sys.exit(f'{work}/cfg exists; use a fresh workdir')
    (work / 'cfg').mkdir(parents=True)
    init = R.INIT_ROOT / f'_init_3B_seed{R.SEED}' / '0'

    def cfg(name, setting, layout, steps, ckpt, resume, *, load_opt, decay=None, interval=100000, save_init=False):
        c = G.build(setting, 'trunk1', R.LAYOUTS[layout])
        c['general']['run'] = f'3b_smoke_{name}'
        c['tokens']['train_steps'] = steps
        lrs = c['optimizer']['learning_rate_scheduler']
        lrs['lr_warmup_steps'] = 5
        lrs['lr_decay_starting_step'], lrs['lr_decay_steps'] = decay or (10 ** 6, 1)
        c['checkpoints'].update(checkpoints_path=str(ckpt), resume_checkpoint_path=str(resume) if resume else None,
                                load_optimizer=load_opt, load_lr_scheduler=load_opt, checkpoint_interval=interval,
                                save_initial_state=save_init, save_final_state=True)
        (work / 'cfg' / f'{name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))

    def trunk(name):    # smoke_3b.sbatch seeds <trunk>/0 from the init + latest.txt, as the submitter does
        return work / 'ckpt' / name

    cfg('init', 'quality-first', 'A', 1, work / 'ckpt' / 'init', None, load_opt=False, save_init=True)
    cfg('a_trunk', 'quality-first', 'A', 20, trunk('a'), trunk('a'), load_opt=False, interval=20)
    cfg('a_resume', 'quality-first', 'A', 30, trunk('a'), trunk('a'), load_opt=True)
    cfg('a_branch', 'quality-first', 'A', 30, work / 'ckpt' / 'a_branch', trunk('a') / '20', load_opt=True, decay=(20, 10))
    cfg('a_div', 'diversity-oriented', 'A', 3, trunk('div'), trunk('div'), load_opt=False)
    cfg('a_da', 'disagreement-aware', 'A', 3, trunk('da'), trunk('da'), load_opt=False)
    cfg('b_trunk', 'quality-first', 'B', 10, trunk('b'), trunk('b'), load_opt=False)
    print(f'wrote {len(PHASES)} smoke configs to {work}/cfg (init source: {init})')


STEP_RE = re.compile(r'iteration: (\d+) / \d+ .*?time_per_iteration_ms: ([\d.]+)(K?) .*?grad_norm: ([\w.+-]+) \| '
                     r'lm_loss: ([\w.+-]+) \| lr: ([\w.+-]+)')


def summarize(work: Path):
    rep = {'workdir': str(work), 'phases': {}, 'problems': []}
    for ph in PHASES:
        log = work / 'logs' / f'{ph}.log'
        if not log.is_file():
            rep['problems'].append(f'{ph}: no log')
            continue
        steps = []
        for m in STEP_RE.finditer(log.read_text(errors='replace')):
            ms = float(m.group(2)) * (1000 if m.group(3) else 1)
            steps.append({'step': int(m.group(1)), 's': ms / 1000, 'grad_norm': float(m.group(4)),
                          'loss': float(m.group(5)), 'lr': float(m.group(6))})
        nv = work / 'logs' / f'{ph}.nvsmi.csv'
        peak = max((int(r.split(',')[1]) for r in nv.read_text().splitlines() if r.strip()), default=None) if nv.is_file() else None
        timed = [s['s'] for s in steps[2:]]
        rec = {'steps_logged': len(steps), 'first': steps[0] if steps else None, 'last': steps[-1] if steps else None,
               's_per_it_median': statistics.median(timed) if timed else None,
               'peak_mib': peak, 'peak_gib': round(peak / 1024, 1) if peak else None}
        if any(not math.isfinite(s['loss']) or not math.isfinite(s['grad_norm']) for s in steps):
            rep['problems'].append(f'{ph}: non-finite loss or grad_norm')
        if peak and peak / 1024 > 74.5:
            rep['problems'].append(f'{ph}: peak {peak / 1024:.1f} GiB above the 74.5 GiB budget')
        rep['phases'][ph] = rec
    a = rep['phases']
    if a.get('a_resume', {}).get('first') and a['a_resume']['first']['step'] != 21:
        rep['problems'].append(f'a_resume started at step {a["a_resume"]["first"]["step"]}, expected 21')
    if a.get('a_branch', {}).get('last'):
        if a['a_branch']['first']['step'] != 21 or a['a_branch']['last']['lr'] > 1e-12:
            rep['problems'].append('a_branch did not start at 21 or did not decay to lr 0 at step 30')
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
