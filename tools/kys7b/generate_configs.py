#!/usr/bin/env python
"""Generate the per-epoch-WSD ("know your sources") experiment templates — 7B grid.

7B fork of tools/generate_configs.py. Same grid shape, same guards, Llama-2 7B geometry and
a 50B-unique-token corpus. See tools/kys7b/README.md for the full constant-by-constant diff.

72 logical runs = 18 trunks (6 settings x 3 seeds) + 54 cooldown branches (3 per trunk),
emitted as 108 template files because each trunk is written as 3 SEGMENTS.

Why segments. nanotron only supports a single modulo `checkpoint_interval` plus
`save_final_state`; there is no "save at these exact steps" option. The three branch points
21458 / 42916 / 64373 share no common interval, so the only way to land a checkpoint exactly
on each is to end a run there. Verified with the arithmetic this docstring argues from:
gcd(21458, 42916, 64373) = 1 — 42916 = 2*21458 exactly, but 64373 = 3*21458 - 1, so the
three ends share no divisor above 1 and NO single interval can hit all three. In particular
TRUNK_CKPT_INTERVAL = 5000 lands on none of them (21458 = 4*5000 + 1458, 42916 = 8*5000 +
2916, 64373 = 12*5000 + 4373), so a restart checkpoint can never be mistaken for a branch
point. All three segments share ONE `checkpoints_path` and use latest.txt auto-resume, so a
segment boundary and a mid-segment crash-restart are handled by the same mechanism.

LR continuity. Every trunk segment carries the EP3 decay parameters
(lr_decay_starting_step=64373, lr_decay_steps=7153) but stops at or before 64373. In
nanotron's `lr_lambda` the decay branch at offset 0 returns exactly `initial_lr`, so the
trunk LR over steps 1..64373 is identical to a constant-LR trunk, and identical to what an
independent 3-epoch WSD run would have at those steps. That is what makes cooldown branching
exact rather than approximate.

Branch resume. A branch points `resume_checkpoint_path` at the trunk's step directory
DIRECTLY (`.../trunk/21458`), not at the trunk folder. `serialize/main.py:parse_ckpt_path`
resolves a folder via latest.txt but a directory containing `model_config.json` as the
checkpoint itself — pointing at the folder would silently resume every branch from the
trunk's LATEST step (64373) instead of its own branch point.

Deployment keys (`parallelism`, `micro_batch_size`, `batch_accumulation_per_replica`,
`zero_stage`, `sequence_length`) are deliberately absent — `tools/kys7b/render_config.py`
stamps them from `deploy/clusters_7b.yaml` and derives accum so the 1024-sequence global
batch holds by construction. Both of those numbers are UNCHANGED from the 1.5B grid: the
global batch is still 1024 sequences = 2,097,152 tokens/step, and accum is still derived as
1024 / (mbs * dp).

Usage:
    python tools/kys7b/generate_configs.py --out configs/know-your-sources-7b   # all 108
    python tools/kys7b/generate_configs.py --out /tmp/x --only quality_base:42:ep3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ----------------------------------------------------------------- locked experiment grid
# 50B unique tokens / 2,097,152 tokens per step = 23,841.86 steps per epoch; x3 -> 71,526.
TRAIN_STEPS = 71526                     # 3 epochs, token-matched, uniform across all six arms
WARMUP = 2000
LR = 1.8e-4
BRANCHES = {                            # name -> (decay_start, decay_steps, train_steps)
    'ep1': (21458, 2384, 23842),
    'ep2': (42916, 4768, 47684),
    'ep3': (64373, 7153, 71526),
}
TRUNK_SEGMENTS = [21458, 42916, 64373]  # each segment's absolute end step
# Restart insurance only; never lands on a branch point — see the module docstring for the
# arithmetic (21458/42916/64373 mod 5000 = 1458/2916/4373, none zero).
TRUNK_CKPT_INTERVAL = 5000
BRANCH_CKPT_INTERVAL = 100_000          # never fires; branches keep only save_final_state

SEEDS = [42, 43, 44]
# Setting names only — NO dataset paths. Templates must stay portable: the corpus location is
# a per-cluster deployment fact, so `tools/kys7b/render_config.py` composes each
# `dataset_folder` from the single `data_root` in deploy/clusters_7b.yaml plus a mapping it
# owns internally. Baking DSAI paths in here would make all 108 files wrong on any other
# machine.
SETTINGS = [
    'quality_base',
    'quality_first',
    'diversity_oriented',
    'wrap_inspired',
    'rewire_inspired',
    'disagreement_aware',
]

# No path constants live here any more — ckpt_root / tokenizer_path / data_root are all
# supplied per cluster in deploy/clusters_7b.yaml and composed by tools/kys7b/render_config.py.
#
# One wandb project for the WHOLE grid. All 108 configs carry this same string; runs are
# distinguished by general.run, tags and WANDB_RUN_GROUP. 108 one-run projects would make
# cross-run aggregation impossible. `general.project` is a required field on GeneralArgs and
# is used for nothing but wandb.init(project=...) — it feeds no checkpoint or log path.
PROJECT = 'zhc-7b-50b-wsd'

MODEL = {
    'ddp_bucket_cap_mb': 25,
    'dtype': 'bfloat16',
    'init_method': {'std': 0.02},
    'make_vocab_size_divisible_by': 1,
    'model_config': {
        'is_llama_config': True,
        'hidden_size': 4096,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,        # == num_attention_heads -> full MHA, no GQA
        'intermediate_size': 11008,
        'vocab_size': 32000,
        'tie_word_embeddings': False,     # untied: +131,072,000 params for the lm_head
        'hidden_act': 'silu',
        'max_position_embeddings': 2048,
        'rope_theta': 10000.0,
        'rope_interleaved': False,
        'rms_norm_eps': 1.0e-5,
        'attention_bias': False,
        'bos_token_id': 1,
        'eos_token_id': 2,
        'pad_token_id': None,
        'pretraining_tp': 1,
        'use_cache': False,
    },
}
OPT_FACTORY = {
    'adam_beta1': 0.9, 'adam_beta2': 0.95, 'adam_eps': 1.0e-8,
    'name': 'adamW', 'torch_adam_is_fused': True,
}


def build(setting, seed, kind):
    """kind in {trunk1, trunk2, trunk3, ep1, ep2, ep3}."""
    if kind.startswith('trunk'):
        seg = int(kind[-1])
        train_steps = TRUNK_SEGMENTS[seg - 1]
        # trunk carries the EP3 decay params but stops at/before the decay-start step, so the
        # decay branch is only ever entered at offset 0 -> lr == initial_lr (see module docstring)
        decay_start, decay_steps = BRANCHES['ep3'][0], BRANCHES['ep3'][1]
        interval = TRUNK_CKPT_INTERVAL
        resume_note = ('trunk folder <ckpt_root>/{setting}_seed{seed}_trunk: latest.txt '
                       'auto-resume. Pre-seed step 0 from _init_7B_seed<S>/0.')
    else:
        decay_start, decay_steps, train_steps = BRANCHES[kind]
        interval = BRANCH_CKPT_INTERVAL
        resume_note = (f'direct trunk step dir {decay_start} (== lr_decay_starting_step; it has '
                       'model_config.json, so parse_ckpt_path uses it as-is and ignores the '
                       'trunk latest.txt).')

    return {
        # `run` is the wandb run name verbatim (patch #8 removes upstream's timestamp prefix)
        # and matches this file's stem, so config name == wandb run name == checkpoint dir stem.
        'general': {'project': PROJECT, 'run': f'{setting}_seed{seed}_{kind}', 'seed': seed},
        'model': MODEL,
        # tokenizer_name_or_path, checkpoints_path and resume_checkpoint_path are all composed
        # by render_config.py from deploy/clusters_7b.yaml (tokenizer_path / ckpt_root). Baking
        # them in would point 108 files at DSAI paths. A missing resume_checkpoint_path is the
        # nastiest of the three: parse_ckpt_path logs "No previous checkpoint found" at INFO
        # and returns None, so a cooldown branch would train from RANDOM INIT and finish
        # normally with meaningless weights.
        'tokenizer': {},
        'tokens': {
            'train_steps': train_steps,
            'val_check_interval': -1,
            'limit_val_batches': 0,
            'limit_test_batches': 0,
        },
        'data_stages': [{
            'name': 'S0_top50B',
            'start_training_step': 1,
            'data': {
                # dataset_folder is intentionally ABSENT — render_config.py composes it from
                # deploy/clusters_7b.yaml's `data_root` plus its own setting->subdirectory map.
                # The setting is recovered from general.run, so it cannot drift from the file.
                'dataset': {},
                'num_loading_workers': 1,
                'seed': seed,
            },
        }],
        'checkpoints': {
            'load_optimizer': True,
            'load_lr_scheduler': True,
            'checkpoint_interval': interval,
            'save_initial_state': False,
            'save_final_state': True,
        },
        'optimizer': {
            'weight_decay': 0.1,
            'clip_grad': 1.0,
            'accumulate_grad_in_fp32': True,
            'learning_rate_scheduler': {
                'learning_rate': LR,
                'lr_warmup_steps': WARMUP,
                'lr_warmup_style': 'linear',
                'lr_decay_starting_step': decay_start,
                'lr_decay_style': 'linear',
                'lr_decay_steps': decay_steps,
                'min_decay_lr': 0.0,
            },
            'optimizer_factory': OPT_FACTORY,
        },
        'logging': {'iteration_step_info_interval': 1},
    }, resume_note


HEADER = """\
# {setting} / seed {seed} / {kind}   [7B grid]
#
# EXPERIMENT template — no `parallelism:`, no micro_batch_size / batch_accumulation_per_replica,
# no zero_stage, no sequence_length. Those are DEPLOYMENT and are stamped by
#   tools/kys7b/render_config.py --template <this file> --cluster <c> --seed {seed}
# which derives accum = 1024 / (mbs * dp) so the 2,097,152-token global batch is invariant.
# This file will NOT parse on its own; that is intentional.
#
# steps        : {steps}
# LR schedule  : warmup 0->{warmup} linear to {lr}; constant; decay from step {ds} over {dl} steps -> 0
# resume       : {resume_note}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', help='setting:seed:kind, e.g. quality_base:42:ep3')
    args = ap.parse_args()

    kinds = ['trunk1', 'trunk2', 'trunk3', 'ep1', 'ep2', 'ep3']
    todo = [(s, sd, k) for s in SETTINGS for sd in SEEDS for k in kinds]
    if args.only:
        s, sd, k = args.only.split(':')
        todo = [(s, int(sd), k)]

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for setting, seed, kind in todo:
        cfg, note = build(setting, seed, kind)
        lrs = cfg['optimizer']['learning_rate_scheduler']
        prev = 0 if kind in ('trunk1',) else None
        if kind.startswith('trunk'):
            seg = int(kind[-1])
            prev = 0 if seg == 1 else TRUNK_SEGMENTS[seg - 2]
        else:
            prev = lrs['lr_decay_starting_step']
        head = HEADER.format(
            setting=setting, seed=seed, kind=kind,
            steps=f"{prev+1} -> {cfg['tokens']['train_steps']}",
            warmup=lrs['lr_warmup_steps'], lr=lrs['learning_rate'],
            ds=lrs['lr_decay_starting_step'], dl=lrs['lr_decay_steps'],
            resume_note=note)
        p = args.out / f'{setting}_seed{seed}_{kind}.yaml'
        p.write_text(head + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        n += 1
    print(f'wrote {n} template(s) to {args.out}')


if __name__ == '__main__':
    main()
