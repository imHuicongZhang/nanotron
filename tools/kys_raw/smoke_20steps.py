#!/usr/bin/env python
"""20-step smoke test of a new training environment on the real raw-baseline data.

    python tools/kys_raw/smoke_20steps.py \
        --config configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_trunk1.yaml \
        --init <init_root>/_init_1.5B_seed42/0 --workdir <scratch dir>

Derives a config from a FILLED trunk1 config (same data, model, batch layout, LR schedule, seed)
that runs 20 optimizer steps from a scratch copy of the seed-42 init and writes no checkpoints,
then prints the torchrun command. It exercises everything a real segment needs: the environment
(torch, flash-attn, grouped_gemm, Triton compile), the tokenizer and .ds.metadata, the dataset
index helper, the init checkpoint, recomputation and the global batch.

Compare the logged lm_loss at steps 1, 10 and 20 with INSTALL.md §3.4. Matching values (to about
0.01) confirm the environment reproduces ours; the s/it it prints is your wall-time basis.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True, help='a filled raw_*_trunk1 config')
    ap.add_argument('--init', type=Path, required=True, help='<init_root>/_init_1.5B_seed<S>/0')
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if not cfg['general']['run'].endswith('_trunk1'):
        raise SystemExit('use a trunk1 config: it starts from the init checkpoint at step 0')
    if not (args.init / 'model_config.json').is_file():
        raise SystemExit(f'{args.init} is not a checkpoint directory (no model_config.json)')
    name = cfg['general']['run'].replace('_trunk1', f'_smoke{args.steps}')
    ckpt = args.workdir / name
    if not (ckpt / 'latest.txt').is_file():
        ckpt.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.init, ckpt / '0', dirs_exist_ok=True)
        (ckpt / 'latest.txt').write_text('0')
    cfg['general']['run'] = name
    cfg['tokens']['train_steps'] = args.steps
    cfg['checkpoints'].update(checkpoints_path=str(ckpt), resume_checkpoint_path=str(ckpt),
                              checkpoint_interval=100000, save_final_state=False)
    out = args.workdir / f'{name}.yaml'
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    p = cfg['parallelism']
    nproc = p['dp'] * p['tp'] * p['pp']
    print(f'wrote {out}\nrun (on a GPU node, env activated, WANDB_MODE=disabled or your wandb settings):\n'
          f'  python -m torch.distributed.run --nproc_per_node {nproc} --nnodes 1 --rdzv_backend c10d '
          f'--max_restarts 0 run_train.py --config-file {shlex.quote(str(out))}')


if __name__ == '__main__':
    main()
