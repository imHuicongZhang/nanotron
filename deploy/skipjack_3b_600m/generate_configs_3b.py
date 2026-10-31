#!/usr/bin/env python
"""SKIPJACK-ONLY: render the Skipjack configs (setting x 6 trunk segments + 3 decay branches) of the 3B grid.

    python deploy/skipjack_3b_600m/generate_configs_3b.py [--layout A|B] [--out configs/3B-rewrite-600m/seed42]
    python deploy/skipjack_3b_600m/generate_configs_3b.py --settings quality-base     # prepared setting -> seed42_prepared/

Every value comes from registry_3b.py (science from deploy/portable_3b/science_3b.py); the output is fully rendered
with Skipjack paths. Other sites: deploy/portable_3b/kys3b.py render --operator <operator.yaml>. Re-rendering a live setting (registry_3b.SETTINGS) is refused once a submission state
exists (the plan would no longer match the configs it scheduled), unless --force is given before anything was
submitted. A prepared setting (registry_3b.PREPARED_SETTINGS) is not in any submission plan; it is rendered into
registry_3b.cfg_dir(setting) with the layout recorded in the live grid's configs, so the grid keeps one layout.

Per segment:
  trunk1        weights-only load of the shared init (seeded into trunk/0 by the submitter); writes trunk/<step>
  trunk2..6     resume optimizer + LR state from trunk/latest.txt
  ep1/ep2/ep3   resume from the trunk STEP directory of their branch point (8583 / 17166 / 25749) with
                optimizer + LR state, decay linearly to 0, write to their own ep<N>/ namespace: branch weights
                never feed back into the trunk.
Trunk segments carry ep3's decay parameters (start 25749, 2861 steps) and stop at or before 25749, so their LR is
warmup-then-constant: nanotron's linear decay returns exactly the peak LR at offset 0 (tools/generate_configs.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry_3b as R  # noqa: E402


def build(setting: str, kind: str, layout: dict) -> dict:
    """The Skipjack config of one segment: science_3b.build_config with the Skipjack paths, W&B project and the 4-GPU
    layout (the same builder renders the portable configs, so only those inputs can differ)."""
    return R.SCI.build_config(setting, kind, tokenized_root=R.TOKENIZED, tokenizer_dir=R.TOKENIZER, runs_root=R.CKPT_ROOT,
                              project=R.WANDB_PROJECT, dp=R.DP, micro_batch_size=layout['micro_batch_size'],
                              zero_stage=layout['zero_stage'], recompute_layer=layout['recompute_layer'])


HEADER = """\
# RENDERED by deploy/skipjack_3b_600m/generate_configs_3b.py from registry_3b.py — do not edit; re-render.
# grid     : 3B rewrite-600m, setting {setting}, seed {seed}, segment {kind} (steps {first}->{end})
# layout   : {layout_name} = zero_stage {zero} / mbs {mbs} x accum {accum} x dp {dp} = 1024 seq = 2,097,152 tokens/step
# LR       : warmup 0->{warmup} linear to {lr}; constant; linear decay from step {ds} over {dl} steps -> 0
# resume   : {resume}
"""


def env_text(setting: str, kind: str, layout_name: str, layout: dict) -> str:
    accum = R.GLOBAL_BATCH_SEQ // (layout['micro_batch_size'] * R.DP)
    tags = (f'grid:3b-rewrite600m,setting:{setting},seed:{R.SEED},kind:{kind},'
            f'phase:{"trunk" if kind.startswith("trunk") else "branch"},layout:{layout_name},'
            f'mbs:{layout["micro_batch_size"]},zero:{layout["zero_stage"]},dp:{R.DP},accum:{accum},tok_per_step:{R.TOKENS_PER_STEP}')
    return ('# sourced by deploy/skipjack_3b_600m/segment_3b.sbatch; written by generate_configs_3b.py — do not edit\n'
            f'export WANDB_PROJECT={R.WANDB_PROJECT}\nexport WANDB_MODE=online\nexport WANDB_ENTITY={R.WANDB_ENTITY}\n'
            f'export WANDB_DIR={R.WANDB_DIR}\nexport WANDB_RUN_GROUP=3b_{R.slug(setting)}_seed{R.SEED}\n'
            f'export WANDB_JOB_TYPE={kind}\nexport WANDB_TAGS={tags}\n'
            f'export KYS3B_LAYOUT={layout_name}\nexport KYS3B_EXPECTED_MBS={layout["micro_batch_size"]}\n'
            f'export KYS3B_EXPECTED_TOK_PER_STEP={R.TOKENS_PER_STEP}\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--layout', choices=sorted(R.LAYOUTS), default=None,
                    help=f'default {R.DEFAULT_LAYOUT}; for prepared settings, the live grid\'s recorded layout')
    ap.add_argument('--settings', nargs='+', choices=R.GRID_SETTINGS, default=R.SETTINGS)
    ap.add_argument('--out', type=Path, default=None, help='default registry_3b.cfg_dir(setting)')
    ap.add_argument('--force', action='store_true', help='re-render although a submission state file exists')
    args = ap.parse_args(argv)
    live = [s for s in args.settings if s in R.SETTINGS]
    if live and len(set(map(R.cfg_dir, args.settings))) != 1 and args.out is None:
        sys.exit('render live and prepared settings separately (they go to different directories)')
    states = [p for p in (R.STATE_ROOT / f'seed{R.SEED}' / 'state.json', R.TRAIN8_STATE / 'state.json') if p.exists()]
    if live and states and not args.force and args.out is None:
        sys.exit(f'{states[0]} exists: the grid was planned/submitted from the current configs; refusing to re-render')
    if args.layout is None:
        args.layout = R.DEFAULT_LAYOUT if live else live_layout()
    elif not live and args.layout != live_layout():
        sys.exit(f'layout {args.layout} differs from the live grid\'s {live_layout()}: one layout for the whole grid')
    layout = R.LAYOUTS[args.layout]
    args.out = args.out or R.cfg_dir(args.settings[0])
    args.out.mkdir(parents=True, exist_ok=True)
    for setting in args.settings:
        for kind in R.KINDS:
            cfg = build(setting, kind, layout)
            lrs = cfg['optimizer']['learning_rate_scheduler']
            head = HEADER.format(setting=setting, seed=R.SEED, kind=kind, first=R.FIRST_STEP[kind], end=R.END_STEP[kind],
                                 layout_name=args.layout, zero=layout['zero_stage'], mbs=layout['micro_batch_size'],
                                 accum=cfg['tokens']['batch_accumulation_per_replica'], dp=R.DP, warmup=R.WARMUP_STEPS,
                                 lr=R.PEAK_LR, ds=lrs['lr_decay_starting_step'], dl=lrs['lr_decay_steps'],
                                 resume=cfg['checkpoints']['resume_checkpoint_path'])
            name = R.seg_name(setting, kind)
            (args.out / f'{name}.yaml').write_text(head + yaml.safe_dump(cfg, sort_keys=False))
            (args.out / f'{name}.env').write_text(env_text(setting, kind, args.layout, layout))
    print(f'wrote {len(args.settings) * len(R.KINDS)} configs (+ .env) to {args.out} with layout {args.layout} {layout}')


def live_layout() -> str:
    """The single KYS3B_LAYOUT recorded in the live grid's rendered .env files."""
    lays = {ln.split('=', 1)[1].strip() for p in R.cfg_dir(R.SETTINGS[0]).glob('*.env')
            for ln in p.read_text().splitlines() if ln.startswith('export KYS3B_LAYOUT=')}
    if len(lays) != 1:
        sys.exit(f'{R.cfg_dir(R.SETTINGS[0])}: live grid layouts {sorted(lays)}; expected exactly one')
    return lays.pop()


if __name__ == '__main__':
    main()
