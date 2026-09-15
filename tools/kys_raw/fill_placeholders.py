#!/usr/bin/env python
"""Produce runnable raw-baseline configs for an external cluster from its deploy/clusters.yaml entry.

    python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42

Reads the cluster entry (see the marc-cluster template in deploy/clusters.yaml), refuses if any
required field is still null, and re-renders configs/1.5B-baseline-seed<S>/templates/ through
tools/render_config.py with a profile built from that entry. Every renderer guard applies:
accum = 1024 / (mbs * dp) must be an integer, the mbs must fit hbm_gib under the memory model,
and the seed must be assigned to this cluster. Output: configs/1.5B-baseline-seed<S>/filled/
(24 configs + .env files), then tools/assert_invariants.py runs on each (placeholders, batch,
environment; corpus and resume checks need the data and checkpoints on disk and are run by the
launcher before every segment).
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
REQUIRED = ('data_root', 'tokenizer_path', 'ckpt_root', 'init_root', 'repo_dir', 'env_activate',
            'gpus_per_node', 'dp', 'tp', 'pp', 'micro_batch_size', 'recompute_layer', 'zero_stage')
REQUIRED_WANDB = ('mode', 'project', 'dir')
REQUIRED_SLURM = ('partition', 'gres', 'cpus_per_task', 'time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', default='marc-cluster')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--expected-mbs', type=int,
                    help='pass the mbs you set if it is not 32 (recorded as a deviation by assert_invariants.py)')
    ap.add_argument('--profile', type=Path, default=REPO / 'deploy' / 'clusters.yaml')
    ap.add_argument('--out-dir', type=Path, help='default: configs/1.5B-baseline-seed<S>/filled')
    args = ap.parse_args()

    prof = yaml.safe_load(args.profile.read_text())
    c = prof['clusters'].get(args.cluster)
    if c is None:
        sys.exit(f'no cluster {args.cluster!r} in deploy/clusters.yaml')
    missing = [k for k in REQUIRED if c.get(k) is None]
    missing += [f'wandb.{k}' for k in REQUIRED_WANDB if (c.get('wandb') or {}).get(k) is None]
    missing += [f'slurm.{k}' for k in REQUIRED_SLURM if (c.get('slurm') or {}).get(k) is None]
    if (c.get('wandb') or {}).get('mode') == 'online' and not c['wandb'].get('entity'):
        missing.append('wandb.entity (online mode)')
    if missing:
        sys.exit(f'{args.cluster} is not filled in yet; null fields: {missing}')
    if not isinstance(c['recompute_layer'], bool):
        sys.exit(f'recompute_layer must be true or false, got {c["recompute_layer"]!r}')

    tmp = copy.deepcopy(prof)
    for k in ('data_root', 'tokenizer_path', 'ckpt_root'):
        tmp[k] = c[k]
    tmp['wandb'] = dict(c['wandb'])
    if tmp['wandb']['mode'] == 'offline':
        tmp['wandb'].pop('entity', None)
    tmp['baseline_seed_assignment'] = {args.seed: args.cluster}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
        yaml.safe_dump(tmp, fh, sort_keys=False)
        profile = fh.name

    folder = REPO / 'configs' / f'1.5B-baseline-seed{args.seed}'
    templates = sorted((folder / 'templates').glob(f'raw_*_seed{args.seed}_*.yaml'))
    if len(templates) != 24:
        sys.exit(f'{folder}/templates holds {len(templates)} templates, expected 24 (4 settings x 6 segments)')
    out_dir = args.out_dir or folder / 'filled'
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for t in templates:
        out = out_dir / t.name
        subprocess.run([sys.executable, REPO / 'tools/render_config.py', '--template', t, '--cluster', args.cluster,
                        '--seed', str(args.seed), '--profile', profile, '--out', out], check=True,
                       stdout=subprocess.DEVNULL)
        cmd = [sys.executable, REPO / 'tools/assert_invariants.py', '--config', out, '--skip-corpus', '--skip-env']
        if args.expected_mbs:
            cmd += ['--expected-mbs', str(args.expected_mbs)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(('OK   ' if r.returncode == 0 else 'FAIL ') + out.name)
        if r.returncode:
            failures += 1
            print(r.stdout[-800:] + r.stderr[-800:])
    Path(profile).unlink()
    if failures:
        sys.exit(f'{failures} filled config(s) failed assert_invariants.py')
    print(f'wrote 24 filled configs to {out_dir}. Next: run tools/assert_invariants.py --check-resume per segment '
          f'(the launcher does), and tools/kys_raw/plan_submit.py --cluster {args.cluster} --seed {args.seed}.')


if __name__ == '__main__':
    main()
