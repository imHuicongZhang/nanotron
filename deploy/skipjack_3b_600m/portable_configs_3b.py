#!/usr/bin/env python
"""Render the four-setting 3B grid (36 configs + .env) for a machine other than Skipjack, changing ONLY paths.

    python deploy/skipjack_3b_600m/portable_configs_3b.py --data <download of blab-jhu/KYS-3B-pretraining-data> \
        --ckpt <checkpoint root> [--wandb <dir>] --out <dir>

<data> is the dataset repo layout: tokenized/<setting>/ and tokenizer/. The configs are rendered by
generate_configs_3b.build/env_text with the KYS3B_* path overrides of registry_3b.py set, in the layout recorded in the
committed Skipjack configs, and then compared with those committed configs: every difference must be one of the path
keys below (and in the .env only WANDB_DIR), otherwise nothing is written. Scientific settings (architecture, tokenizer
contents, seed, batch, LR schedule, branch points, finals, resume rules) are therefore identical by construction.

Before training, point the downloaded tokenized metadata at the same tokenizer directory (nanotron reads line 1 of every
.ds.metadata and requires one identical string across folders):
    python tools/fix_ds_metadata.py --output-folder <data>/tokenized/<setting> --tokenizer-dir <data>/tokenizer
and then `python deploy/skipjack_3b_600m/assert_invariants_3b.py --grid --corpus` with the same KYS3B_* variables
(printed by this tool) checks the configs and the exact token totals.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PATH_KEYS = {'tokenizer.tokenizer_name_or_path', 'data_stages.0.data.dataset.dataset_folder.0',
             'checkpoints.checkpoints_path', 'checkpoints.resume_checkpoint_path'}
ENV_PATH_KEYS = {'WANDB_DIR'}


def _flat(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f'{pre}{k}.')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f'{pre}{i}.')
    else:
        yield pre[:-1], x


def _env(text: str) -> dict:
    return dict(ln[len('export '):].split('=', 1) for ln in text.splitlines() if ln.startswith('export '))


def overrides(data: Path, ckpt: Path, wandb: Path | None) -> dict:
    return {'KYS3B_TOKENIZED': str(data / 'tokenized'), 'KYS3B_TOKENIZER': str(data / 'tokenizer'),
            'KYS3B_CKPT_BASE': str(ckpt), 'KYS3B_WANDB_DIR': str(wandb or ckpt / 'wandb')}


def _child(out: Path) -> None:
    """Runs with the overrides in the environment: generate_configs_3b's own rendering, all four settings, one layout."""
    sys.path.insert(0, str(HERE))
    import generate_configs_3b as G
    import registry_3b as R
    lay = G.live_layout()
    for s in R.GRID_SETTINGS:
        G.main(['--settings', s, '--out', str(out), '--layout', lay])
    print(json.dumps({'layout': lay, 'settings': R.GRID_SETTINGS, 'kinds': R.KINDS,
                      'committed': {s: str(R.cfg_dir(s)) for s in R.GRID_SETTINGS}}))


def render(ov: dict, out: Path) -> dict:
    """Render in a child process (registry_3b reads the overrides at import)."""
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, __file__, '--child', str(out)], env={**os.environ, **ov},
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def compare(meta: dict, out: Path) -> list[str]:
    errs = []
    for s in meta['settings']:
        d = Path(meta['committed'][s])
        for k in meta['kinds']:
            n = f'3b_{s.replace("-", "_")}_seed42_{k}'
            a = dict(_flat(yaml.safe_load((out / f'{n}.yaml').read_text())))
            b = dict(_flat(yaml.safe_load((d / f'{n}.yaml').read_text())))
            bad = {x for x in set(a) | set(b) if a.get(x) != b.get(x)} - PATH_KEYS
            ea, eb = _env((out / f'{n}.env').read_text()), _env((d / f'{n}.env').read_text())
            bad |= {f'env:{x}' for x in set(ea) | set(eb) if ea.get(x) != eb.get(x)} - {f'env:{x}' for x in ENV_PATH_KEYS}
            if bad:
                errs.append(f'{n}: non-path keys differ from {d}: {sorted(bad)}')
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, required=True, help='local copy of the dataset repo (tokenized/, tokenizer/)')
    ap.add_argument('--ckpt', type=Path, required=True, help='checkpoint root (init/ and runs/ are created under it)')
    ap.add_argument('--wandb', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    ov = overrides(args.data.resolve(), args.ckpt.resolve(), args.wandb and args.wandb.resolve())
    meta = render(ov, args.out)
    errs = compare(meta, args.out)
    if errs:
        for p in args.out.glob('3b_*'):
            p.unlink()
        sys.exit('\n'.join(['REFUSED: rendered configs differ beyond paths:', *errs]))
    print(f'wrote {2 * len(meta["settings"]) * len(meta["kinds"])} files to {args.out} (layout {meta["layout"]}); '
          f'only {sorted(PATH_KEYS | ENV_PATH_KEYS)} differ from the committed configs')
    print('export ' + ' '.join(f'{k}={v}' for k, v in ov.items()))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        _child(Path(sys.argv[2]))
    else:
        main()
