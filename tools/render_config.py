#!/usr/bin/env python
"""Stamp a per-cluster deployment profile onto an experiment template.

The experiment templates (configs/kys/*.yaml) describe WHAT is being trained: model, data,
LR schedule, branch point, step counts, seeds. They contain no `parallelism:` block and no
`micro_batch_size` / `batch_accumulation_per_replica`. Those three numbers are *deployment*,
not experiment, and live only in deploy/clusters.yaml.

The invariant this exists to protect: every run in the grid, on every cluster, takes exactly
`global_batch_seq` (1024) sequences per optimizer step. So the renderer never copies an
accumulation value — it DERIVES it:

    batch_accumulation_per_replica = global_batch_seq / (micro_batch_size * dp)

and refuses to emit anything if that is not an exact integer. There is therefore no config
in the tree that can silently disagree about the global batch.

It also enforces the split rule: the grid may be divided across clusters only by SEED, never
by setting, so that any between-setting comparison stays inside one hardware generation.
`--seed` is checked against `seed_assignment` in the profile and rendering aborts on a
mismatch.

Usage:
    python tools/render_config.py --template configs/kys/quality-base_seed42_trunk.yaml \
        --cluster dsai --seed 42 --out /tmp/rendered.yaml
    python tools/render_config.py --template ... --cluster dsai --seed 42 --print
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = HERE / 'deploy' / 'clusters.yaml'

# Keys the renderer owns. A template that already sets one of these is a bug — it means the
# config tree has started to fork by cluster, which is the thing we are preventing.
OWNED = {
    'parallelism': ('dp', 'tp', 'pp'),
    'tokens': ('micro_batch_size', 'batch_accumulation_per_replica'),
    'optimizer': ('zero_stage',),
    'tokenizer': ('tokenizer_name_or_path',),
    'checkpoints': ('checkpoints_path', 'resume_checkpoint_path'),
}

# ---------------------------------------------------------------------------------------
# setting -> corpus subdirectory under `data_root`. HARDCODED HERE ON PURPOSE.
#
# Three different naming schemes are in play for the same six arms, and they do not line up:
#
#   paper setting        HF/repo folder      setting name (ours)     corpus dir (ON DISK)
#   -------------------  ------------------  ----------------------  ----------------------------
#   QUALITY-BASE         quality_base        quality_base            10B-base-shuf42
#   QUALITY-FIRST        quality_first       quality_first           quality-first
#   DIVERSITY-ORIENTED   diversity_oriented  diversity_oriented      diversity-first
#   WRAP-INSPIRED        wrap_inspired       wrap                    wrap
#   REWIRE-INSPIRED      rewire_inspired     rewire                  rewrite
#   DISAGREEMENT-AWARE   disagreement_aware  disagreement_aware_0p5  signal-disagreement-lambda05
#
# The right-hand column is the ONLY one that touches the filesystem, and none of those names
# changed when the setting labels were renamed (2026-08-09) — they are the actual directory
# names of the tokenized corpora. Note especially: diversity_oriented's corpus is
# `diversity-first`, rewire's is `rewrite`, disagreement_aware_0p5's is
# `signal-disagreement-lambda05`, and quality_base's is `10B-base-shuf42` (NOT `quality_base`,
# and NOT the old unshuffled `10B-base`).
#
# Wiring these by hand is a trap: a wrong-but-existing path does not crash. Training runs to
# completion on the wrong corpus and the numbers are silently meaningless. So nobody wires
# them by hand — `data_root` is the single value a deployer sets, and this table does the rest.
# tools/assert_invariants.py then verifies each resolved path by token count before any GPU
# time is spent.
# ---------------------------------------------------------------------------------------
SETTING_CORPUS = {
    'quality_base':           '10B-base-shuf42',
    'quality_first':          'quality-first',
    'diversity_oriented':     'diversity-first',
    'wrap':                   'wrap',
    'rewire':                 'rewrite',
    'disagreement_aware_0p5': 'signal-disagreement-lambda05',
}
CORPUS_LEAF = 'tokenized'   # <data_root>/<corpus dir>/tokenized/*.ds


def die(msg):
    print(f'render_config: {msg}', file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--print', action='store_true', dest='do_print')
    args = ap.parse_args()

    prof = yaml.safe_load(args.profile.read_text())
    if args.cluster not in prof['clusters']:
        die(f'unknown cluster {args.cluster!r}; have {sorted(prof["clusters"])}')
    c = prof['clusters'][args.cluster]

    # --- split rule: by seed, never by setting -------------------------------------------
    assign = prof.get('seed_assignment', {})
    if args.seed not in assign:
        die(f'seed {args.seed} has no entry in seed_assignment')
    if assign[args.seed] != args.cluster:
        die(f'seed {args.seed} is assigned to cluster {assign[args.seed]!r}, refusing to '
            f'render it for {args.cluster!r}. The grid splits by SEED only — all six '
            f'settings of a seed must run on one cluster.')

    # --- the profile must actually be filled in ------------------------------------------
    missing = [k for k in ('gpus_per_node', 'dp', 'tp', 'pp', 'micro_batch_size') if c.get(k) is None]
    if missing:
        die(f'cluster {args.cluster!r} profile is incomplete: {missing} are null. '
            f'Fill deploy/clusters.yaml from the real node spec — do not assume 8 GPUs/node.')

    gbs = prof['global_batch_seq']
    dp, mbs = int(c['dp']), int(c['micro_batch_size'])
    if dp * mbs > gbs or gbs % (dp * mbs) != 0:
        die(f'global_batch_seq {gbs} is not divisible by dp*mbs = {dp}*{mbs} = {dp*mbs}; '
            f'pick a micro_batch_size that divides {gbs // dp} exactly.')
    accum = gbs // (dp * mbs)

    # --- micro_batch_size must be identical across every cluster in the grid ----------------
    # dp is mathematically free (the micro-batch partition is consecutive blocks of size mbs
    # regardless of dp), but mbs is not: masked_mean's denominator is the count of UNMASKED
    # tokens in each micro-batch, and label_mask drops the token before every document
    # boundary, so regrouping into a different mbs re-weights the per-token contributions by
    # ~1e-3 relative. See deploy/clusters.yaml for the measurements.
    used = {}
    for sd, cl in sorted(assign.items()):
        m = prof['clusters'].get(cl, {}).get('micro_batch_size')
        if m is not None:
            used.setdefault(int(m), []).append(f'seed {sd} -> {cl}')
    if len(used) > 1:
        lines = '; '.join(f'mbs={m}: {", ".join(v)}' for m, v in sorted(used.items()))
        die(f'the clusters in seed_assignment disagree about micro_batch_size ({lines}). '
            f'mbs changes the objective, not just rounding — every run in the grid must use '
            f'the same value. Pick one that fits the smallest card in use and set it on all '
            f'of them.')

    # --- will this mbs actually fit the cards? ----------------------------------------------
    mm, hbm = prof.get('memory_model'), c.get('hbm_gib')
    if mm and hbm:
        peak = mm['static_gib'] + mm['per_micro_batch_seq_gib'] * mbs
        budget = float(hbm) * mm['reserve_frac']
        if peak > budget:
            die(f'micro_batch_size {mbs} needs ~{peak:.1f} GiB on {args.cluster} but only '
                f'{budget:.1f} GiB is usable ({hbm} GiB x {mm["reserve_frac"]}). '
                f'Largest power-of-two that fits: '
                f'{max([x for x in (1,2,4,8,16,32,64) if mm["static_gib"]+mm["per_micro_batch_seq_gib"]*x <= budget], default=0)}.')
        print(f'render_config: mbs={mbs} -> ~{peak:.1f} GiB peak of {hbm} GiB '
              f'({peak/float(hbm)*100:.0f}%) on {args.cluster}', file=sys.stderr)

    if c['gpus_per_node'] and dp * int(c['tp']) * int(c['pp']) % int(c['gpus_per_node']):
        print(f'render_config: NOTE dp*tp*pp = {dp*int(c["tp"])*int(c["pp"])} does not fill '
              f'whole {c["gpus_per_node"]}-GPU nodes', file=sys.stderr)

    cfg = yaml.safe_load(args.template.read_text())

    for section, keys in OWNED.items():
        present = [k for k in keys if isinstance(cfg.get(section), dict) and k in cfg[section]]
        if present:
            die(f'template already sets {section}.{present} — deployment keys must not appear '
                f'in experiment templates (that is a per-cluster fork of the config tree).')

    cfg.setdefault('tokens', {})
    cfg.setdefault('optimizer', {})
    cfg['parallelism'] = {
        'dp': dp, 'tp': int(c['tp']), 'pp': int(c['pp']),
        'expert_parallel_size': 1, 'pp_engine': '1f1b',
        'tp_mode': 'REDUCE_SCATTER', 'tp_linear_async_communication': True,
    }
    cfg['tokens']['micro_batch_size'] = mbs
    cfg['tokens']['batch_accumulation_per_replica'] = accum
    cfg['tokens']['sequence_length'] = prof['sequence_length']
    cfg['optimizer']['zero_stage'] = int(c.get('zero_stage', 0))

    # --- the invariant, asserted on the rendered output ----------------------------------
    seq_per_step = mbs * accum * dp
    tok_per_step = seq_per_step * prof['sequence_length']
    if seq_per_step != gbs:
        die(f'INTERNAL: rendered global batch {seq_per_step} != {gbs}')
    if cfg.get('general', {}).get('seed') not in (None, args.seed):
        die(f'template general.seed={cfg["general"]["seed"]} != --seed {args.seed}')

    # --- wandb destination: mandatory, never defaulted ---------------------------------------
    # nanotron calls wandb.init(project=..., name=..., config=...) with NO entity= and NO tags=,
    # so entity/tags can only come from the environment. If WANDB_ENTITY is unset, wandb falls
    # back to whichever account holds the cached credential on that machine — which is exactly
    # how a run silently lands in a personal default project. Hence: refuse without it.
    wb = prof.get('wandb') or {}
    mode = wb.get('mode', 'offline')
    if mode not in ('online', 'offline'):
        die(f'wandb.mode must be online|offline, got {mode!r}')
    if not wb.get('project'):
        die(f'wandb.project not set in {args.profile}. Mandatory — wandb otherwise files runs '
            f'under a default project.')
    if wb['project'] != cfg.get('general', {}).get('project'):
        die(f'wandb.project ({wb["project"]!r}) != template general.project '
            f'({cfg.get("general", {}).get("project")!r}); they must match or the run and its '
            f'config will land in different projects.')
    if mode == 'offline':
        # An offline run records entity='' and takes its destination from `wandb sync --entity`
        # at sync time. Setting WANDB_ENTITY at init would bake a destination into every run
        # directory, which defeats the point of syncing as the owner.
        if wb.get('entity'):
            die(f'wandb.entity is set ({wb["entity"]!r}) while wandb.mode is offline. Offline '
                f'runs must stay unattributed — the entity is chosen by whoever runs '
                f'`wandb sync --entity ...`, and that is what makes them theirs. Remove it.')
        if not wb.get('dir'):
            die(f'wandb.dir not set in {args.profile}. Offline runs are written to WANDB_DIR; '
                f'if it is unset wandb falls back to the working directory, which on a compute '
                f'node is the standard way to lose every log in the grid. Point it at shared '
                f'storage that outlives the job and is readable from the sync host.')
    else:
        if not wb.get('entity'):
            die(f'wandb.entity not set in {args.profile} and wandb.mode is online. Without an '
                f'explicit entity, wandb logs to whatever cached credential exists on the '
                f'machine.')

    run_name = cfg['general']['run']
    setting, seedpart, kind = run_name.rsplit('_', 2)[0], f'seed{args.seed}', run_name.rsplit('_', 1)[1]

    # --- dataset_folder: composed, never copied ---------------------------------------------
    data_root = prof.get('data_root')
    if not data_root:
        die(f'data_root not set in {args.profile}. It is the ONE path a deployer fills in: the '
            f'directory holding the six tokenized corpora. Templates carry no dataset paths, so '
            f'nothing can be rendered until this points somewhere real.')
    if setting not in SETTING_CORPUS:
        die(f'no corpus mapping for setting {setting!r} (from general.run={run_name!r}). '
            f'Known: {sorted(SETTING_CORPUS)}')
    dataset_folder = str(Path(data_root) / SETTING_CORPUS[setting] / CORPUS_LEAF)
    stage = cfg['data_stages'][0]['data']
    if stage.get('dataset', {}).get('dataset_folder'):
        die(f'template already sets data_stages[0].data.dataset.dataset_folder — dataset paths '
            f'must not be baked into templates; they are composed from data_root.')
    stage.setdefault('dataset', {})['dataset_folder'] = [dataset_folder]

    # --- tokenizer and checkpoint paths: same treatment, same reason -------------------------
    tok_path = prof.get('tokenizer_path')
    ckpt_root = prof.get('ckpt_root')
    for k, v in (('tokenizer_path', tok_path), ('ckpt_root', ckpt_root)):
        if not v:
            die(f'{k} not set in {args.profile}. Templates carry no absolute paths at all — '
                f'tokenizer, checkpoints and data are all composed from deploy/clusters.yaml.')
    if cfg.get('tokenizer', {}).get('tokenizer_name_or_path'):
        die('template already sets tokenizer.tokenizer_name_or_path — composed from tokenizer_path.')
    if cfg.get('checkpoints', {}).get('checkpoints_path'):
        die('template already sets checkpoints.checkpoints_path — composed from ckpt_root.')
    cfg.setdefault('tokenizer', {})['tokenizer_name_or_path'] = str(tok_path)

    trunk_dir = Path(ckpt_root) / f'{setting}_{seedpart}_trunk'
    if kind.startswith('trunk'):
        # All three segments share ONE directory and resume via latest.txt, so a segment
        # boundary and a mid-segment crash-restart use the same mechanism. Pre-seed step 0.
        ckpt_path, resume = trunk_dir, trunk_dir
    else:
        # Resume from the trunk's STEP DIRECTORY, never the folder: parse_ckpt_path resolves a
        # folder through latest.txt, which would silently point every branch at the trunk's
        # latest step (12875) instead of its own branch point. The step is the branch's own
        # lr_decay_starting_step by construction, so the two cannot drift apart.
        branch_step = cfg['optimizer']['learning_rate_scheduler']['lr_decay_starting_step']
        ckpt_path = Path(ckpt_root) / f'{setting}_{seedpart}_{kind}'
        resume = trunk_dir / str(branch_step)
    cfg.setdefault('checkpoints', {})['checkpoints_path'] = str(ckpt_path)
    cfg['checkpoints']['resume_checkpoint_path'] = str(resume)
    # Provenance that surfaces in the wandb UI and is filterable. The authoritative record is
    # still config.nanotron_config.tokens.*, which nanotron uploads wholesale; these tags exist
    # so `mbs` can be audited at a glance across 108 runs without opening each config.
    tags = [f'setting:{setting}', f'seed:{args.seed}', f'kind:{kind}',
            f'phase:{"trunk" if kind.startswith("trunk") else "endpoint"}',
            f'cluster:{args.cluster}', f'mbs:{mbs}', f'dp:{dp}', f'accum:{accum}',
            f'tok_per_step:{tok_per_step}']
    env_lines = [
        '# sourced by the launcher; written by tools/render_config.py — do not edit',
        f'export WANDB_PROJECT={wb["project"]}',
        f'export WANDB_MODE={mode}',
    ]
    if mode == 'offline':
        env_lines += [
            f'export WANDB_DIR={wb["dir"]}',
            '# NOTE: WANDB_ENTITY is deliberately NOT exported. Offline runs stay unattributed;',
            '# the owner is decided by `wandb sync --entity <E> --project <P>` on the sync host.',
            '# Do NOT run `wandb sync` here — leave the offline-run directories in place.',
        ]
    else:
        env_lines += [f'export WANDB_ENTITY={wb["entity"]}']
    env_lines += [
        f'export WANDB_RUN_GROUP={setting}_{seedpart}',   # one group per (setting,seed) chain
        f'export WANDB_JOB_TYPE={kind}',
        f'export WANDB_TAGS={",".join(tags)}',
        f'export KYS_EXPECTED_MBS={mbs}',
        f'export KYS_EXPECTED_TOK_PER_STEP={tok_per_step}',
    ]

    banner = (f'# RENDERED by tools/render_config.py — do not edit.\n'
              f'# template : {args.template}\n'
              f'# cluster  : {args.cluster}  (dp={dp} tp={c["tp"]} pp={c["pp"]} '
              f'mbs={mbs} accum={accum}, {c["gpus_per_node"]} GPU/node)\n'
              f'# seed     : {args.seed}  -> assigned cluster {assign[args.seed]}\n'
              f'# global batch: {mbs} x {accum} x {dp} = {seq_per_step} seq '
              f'= {tok_per_step:,} tokens/step  [INVARIANT]\n'
              f'# corpus   : setting {setting} -> {SETTING_CORPUS[setting]}/{CORPUS_LEAF}\n'
              f'#            composed from data_root={data_root}\n')
    text = banner + yaml.safe_dump(cfg, sort_keys=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        envp = args.out.with_suffix('.env')
        envp.write_text('\n'.join(env_lines) + '\n')
        print(f'wrote {args.out}  (mbs={mbs} accum={accum} dp={dp} -> {tok_per_step:,} tok/step)')
        dest = (f'entity set at sync time; WANDB_DIR={wb["dir"]}' if mode == 'offline'
                else f'WANDB_ENTITY={wb["entity"]}')
        print(f'wrote {envp}  (project={wb["project"]} mode={mode}, {dest})')
    if args.do_print or not args.out:
        print(text)


if __name__ == '__main__':
    main()
