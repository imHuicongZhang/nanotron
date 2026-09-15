#!/usr/bin/env python
"""Render the 24 raw-baseline templates per seed with site-specific values left as {{PLACEHOLDERS}}.

The output is what ships for an external cluster: 24 configs per seed that are complete in every
experiment and batch field (4 settings x 6 segments; the grid's dp 4 / mbs 32 / accum 8, LR schedule, steps, seeds) but
carry these markers wherever a value belongs to the cluster that runs them:

    {{DATA_ROOT}}  {{TOKENIZER_PATH}}  {{CKPT_ROOT}}  {{WANDB_ENTITY}}  {{WANDB_DIR}}
    {{CLUSTER}}    {{RECOMPUTE_LAYER}}

They are rendered by tools/render_config.py itself, from a temporary profile, so the
composition logic (dataset folder, checkpoint layout, branch resume step) is the same code
path the filled configs go through. tools/assert_invariants.py refuses any config or .env that
still contains a marker; tools/kys_raw/fill_placeholders.py produces the runnable versions.

Usage:
    python tools/kys_raw/render_placeholders.py --seeds 42,43,44
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
PLACEHOLDER_CLUSTER = '{{CLUSTER}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,43,44')
    args = ap.parse_args()

    prof = yaml.safe_load((REPO / 'deploy' / 'clusters.yaml').read_text())
    grid = prof['clusters']['kys_grid_1p5b']
    tmp = copy.deepcopy(prof)
    tmp['data_root'] = '{{DATA_ROOT}}'
    tmp['tokenizer_path'] = '{{TOKENIZER_PATH}}'
    tmp['ckpt_root'] = '{{CKPT_ROOT}}'
    tmp['wandb'] = {'project': prof['wandb']['project'], 'mode': 'online',
                    'entity': '{{WANDB_ENTITY}}', 'dir': '{{WANDB_DIR}}'}
    tmp['clusters'] = {PLACEHOLDER_CLUSTER: {**{k: grid[k] for k in ('dp', 'tp', 'pp', 'micro_batch_size', 'zero_stage')},
                                             'gpus_per_node': grid['dp'], 'hbm_gib': None, 'recompute_layer': False}}
    seeds = [int(s) for s in args.seeds.split(',')]
    tmp['baseline_seed_assignment'] = {s: PLACEHOLDER_CLUSTER for s in seeds}

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
        yaml.safe_dump(tmp, fh, sort_keys=False)
        profile = fh.name

    n = 0
    for seed in seeds:
        folder = REPO / 'configs' / f'1.5B-baseline-seed{seed}'
        templates = sorted((folder / 'templates').glob(f'raw_*_seed{seed}_*.yaml'))
        if len(templates) != 24:
            sys.exit(f'{folder}/templates holds {len(templates)} templates, expected 24 (4 settings x 6 segments)')
        for t in templates:
            out = folder / t.name
            subprocess.run([sys.executable, REPO / 'tools/render_config.py', '--template', t, '--cluster',
                            PLACEHOLDER_CLUSTER, '--seed', str(seed), '--profile', profile, '--out', out],
                           check=True, stdout=subprocess.DEVNULL)
            # recompute_layer depends on the GPUs that run it; leave it to the operator
            text = out.read_text().replace('recompute_layer: false', "recompute_layer: '{{RECOMPUTE_LAYER}}'")
            text = text.replace('recompute_layer=False', 'recompute_layer={{RECOMPUTE_LAYER}}')
            out.write_text(text)
            n += 1
    Path(profile).unlink()
    print(f'rendered {n} placeholder config(s) for seeds {seeds}')


if __name__ == '__main__':
    main()
