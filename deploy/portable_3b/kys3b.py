#!/usr/bin/env python
"""Portable entry point of the KYS 3B grid: render, check, initialize and run it from ONE operator config.

    python deploy/portable_3b/kys3b.py validate          --operator <operator.yaml>
    python deploy/portable_3b/kys3b.py render            --operator <operator.yaml> [--force]
    python deploy/portable_3b/kys3b.py relocate-metadata --operator <operator.yaml>
    python deploy/portable_3b/kys3b.py check             --operator <operator.yaml> [--corpus [--hash]]
    python deploy/portable_3b/kys3b.py plan              --operator <operator.yaml>
    python deploy/portable_3b/kys3b.py init-config       --config <rendered trunk1.yaml> --out <init.yaml> --ckpt <dir>
    python deploy/portable_3b/kys3b.py finalize-init     --operator <operator.yaml>
    python deploy/portable_3b/kys3b.py seed              --operator <operator.yaml> --config <yaml>
    python deploy/portable_3b/kys3b.py prepare           --operator <operator.yaml> --run <segment run name>
    python deploy/portable_3b/kys3b.py finish            --operator <operator.yaml> --run <segment run name>
    python deploy/portable_3b/kys3b.py startup-derive    --operator <operator.yaml> --workdir <fresh dir>
    python deploy/portable_3b/kys3b.py startup-summarize --operator <operator.yaml> --workdir <dir> [--mem-budget-gib X]
    python deploy/portable_3b/kys3b.py check-log         --config <yaml> --log <file>
    python deploy/portable_3b/kys3b.py wandb-check

Everything site-specific comes from the operator config (operator.template.yaml: paths, Python, W&B entity/project,
deployment, launch, scheduler). There are no defaults for any of it: an unfilled <<placeholder>>, a missing or unknown
key, a value of the wrong type, a Skipjack path / W&B entity / Slurm account, or a deployment whose micro-batch x
accumulation x data-parallel size is not exactly 1024 is refused with the key and what to put there. The science
comes from science_3b.py only. This module never imports the Skipjack tools (deploy/skipjack_3b_600m).

render writes, under CONFIG_ROOT:
  grid/<run>.yaml + .env       the 36 segment configs of all four settings (+ W&B environment of each)
  init/init.yaml + init.env    the 1-step init run, derived from grid/<quality-first trunk1>.yaml (startup.derive_init)
  site.env                     code, Python, paths, launch settings for the shell launchers
  plan.json                    init + 4 settings x 9 segments with dependencies, in a valid execution order
  operator.resolved.yaml       the validated operator config the render used
  render_manifest.json         sha256 of every rendered file, code commit, deployment
  slurm/                       per-segment sbatch files and submit_grid.sh (only with scheduler.type: slurm)
prepare / finish are run around every segment (run_segment.sh does it): prepare refuses a drifted config, a changed
deployment (CHECKPOINT_ROOT/kys3b_deployment.json), a missing W&B login, a missing resume source or one written at
another data-parallel size; it seeds trunk1 from the shared init, and gives trunk1 past step 0 full-state loading.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import netrc
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import initlib  # noqa: E402
import science_3b as S  # noqa: E402
import startup  # noqa: E402

CODE_ROOT_HERE = HERE.parents[1]
BASELINE_1P5B = CODE_ROOT_HERE / 'configs' / '1.5B-baseline-seed42' / 'filled' / 'raw_diversity_oriented_seed42_trunk1.yaml'
INIT_SOURCE_SETTING = 'quality-first'          # the init config is derived from this setting's rendered trunk1
EXIT_COMPLETE = 3                              # prepare: the segment's final checkpoint already exists

# The publisher's (Skipjack) identity. Refused in operator inputs; never used as a default.
PUBLISHER = {
    'path_prefixes': ('/weka/projects/bvandur1', '/weka/scratch/jhu/bvandur1', '/scratch/jhu/bvandur1',
                      '/home/jhu/zhuicon1', '/home/zhuicon1'),
    'wandb_entities': ('imwytro-johns-hopkins-university', 'imwytro'),
    'slurm_accounts': ('bvandur1_blabscify',),
    'slurm_qos': ('jhu2',),
}

PLACEHOLDER = re.compile(r'<<.*?>>')
RUNTIME_REF = re.compile(r'\$\{[A-Za-z_][A-Za-z0-9_]*\}')
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
SLURM_TIME = re.compile(r'^(\d+-)?\d{1,2}:\d{2}:\d{2}$')

# section -> key -> (kind, help). kind: abspath | file | name | posint | zero | bool | str | runtime | choice:<a|b> |
# optstr | strlist | section
SCHEMA = {
    'paths': {
        'CODE_ROOT': ('abspath', 'this nanotron checkout (branch release/kys-3b-600m)'),
        'DATA_ROOT': ('abspath', 'your download of blab-jhu/KYS-3B-pretraining-data (holds tokenized/ and tokenizer/)'),
        'CONFIG_ROOT': ('abspath', 'where render writes the configs, launch files and plan'),
        'INIT_ROOT': ('abspath', 'where the shared seed-42 init is created (_init_3B_seed42/)'),
        'CHECKPOINT_ROOT': ('abspath', 'training checkpoints of all segments (seed42/<setting>/<trunk|epN>/...)'),
        'LOG_ROOT': ('abspath', 'launcher and scheduler logs'),
        'REPORT_ROOT': ('abspath', 'check / prepare / startup reports'),
        'WANDB_DIR': ('abspath', 'local W&B run files (shared storage)'),
    },
    'environment': {'PYTHON': ('abspath', 'python of the env built by install_env.sh (<env>/bin/python)')},
    'wandb': {'WANDB_ENTITY': ('name', 'YOUR W&B user or team (runs log online there)'),
              'WANDB_PROJECT': ('name', 'YOUR W&B project')},
    'deployment': {
        'data_parallel_size': ('posint', 'data-parallel world size = nnodes x nproc_per_node (tp = pp = 1)'),
        'micro_batch_size': ('posint', 'sequences per GPU per micro-batch; one value for all four settings'),
        'zero_stage': ('zero', '0 or 1 (ZeRO-1 shards the optimizer state over data-parallel ranks)'),
        'recompute_layer': ('bool', 'true = activation recomputation per layer (less memory, ~1/3 more compute)'),
    },
    'launch': {
        'nnodes': ('posint', 'nodes per segment'),
        'nproc_per_node': ('posint', 'GPUs (training processes) per node'),
        'rdzv_backend': ('str', 'torchrun rendezvous backend, normally c10d'),
        'rdzv_endpoint': ('runtime', 'host:port reachable from all nodes; may use ${VAR} resolved at launch, e.g. '
                                     '${MASTER_ADDR}:29500 (the Slurm files set MASTER_ADDR to the first node)'),
        'rdzv_id': ('runtime', 'unique id per launch, e.g. ${SLURM_JOB_ID}'),
    },
    'scheduler': {
        'type': ('choice:slurm|none', 'slurm = also render sbatch files; none = run the launchers yourself'),
        'slurm': ('section', 'required when type is slurm (see the template)'),
    },
}
SLURM_SCHEMA = {
    'account': ('str', 'your Slurm account'), 'partition': ('str', 'your GPU partition'),
    'qos': ('optstr', 'QOS, or null'), 'time': ('str', 'wall time per job, [D-]HH:MM:SS'),
    'gpus_per_node': ('posint', 'GPUs requested per node (must equal launch.nproc_per_node)'),
    'cpus_per_task': ('posint', 'CPUs per node task'), 'mem': ('optstr', 'memory per node (e.g. 500G), or null'),
    'constraint': ('optstr', 'node feature constraint, or null'), 'requeue': ('bool', 'let Slurm requeue preempted jobs'),
    'extra_sbatch': ('strlist', 'extra "#SBATCH ..." argument strings, e.g. ["--exclusive"], or []'),
}


class OperatorError(SystemExit):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('operator config refused:\n  ' + '\n  '.join(self.problems))


# ---- operator config --------------------------------------------------------------------------------------------------
def _check_value(where: str, kind: str, help_: str, v, problems: list):
    hint = f' ({help_})'
    if isinstance(v, str) and PLACEHOLDER.search(v):
        problems.append(f'{where}: still the placeholder {v!r}; fill in{hint}')
        return
    if kind == 'abspath':
        if not isinstance(v, str) or not v.startswith('/'):
            problems.append(f'{where}: must be an absolute path, got {v!r}{hint}')
        elif os.path.normpath(v) != v.rstrip('/') or '${' in v:
            problems.append(f'{where}: {v!r} is not a normalized literal path{hint}')
        elif any(v == p or v.startswith(p + '/') for p in PUBLISHER['path_prefixes']):
            problems.append(f'{where}: {v} is under the publisher\'s (Skipjack) storage; use your own{hint}')
    elif kind == 'name':
        if not isinstance(v, str) or not NAME.match(v):
            problems.append(f'{where}: must be a W&B name (letters, digits, . _ -), got {v!r}{hint}')
        elif where.endswith('WANDB_ENTITY') and v in PUBLISHER['wandb_entities']:
            problems.append(f'{where}: {v} is the publisher\'s W&B entity; use your own and share the project with '
                            f'Huicong through W&B instead')
    elif kind == 'posint':
        if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            problems.append(f'{where}: must be a positive integer, got {v!r}{hint}')
    elif kind == 'zero':
        if v not in (0, 1) or isinstance(v, bool):
            problems.append(f'{where}: must be 0 or 1, got {v!r}{hint}')
    elif kind == 'bool':
        if not isinstance(v, bool):
            problems.append(f'{where}: must be true or false, got {v!r}{hint}')
    elif kind == 'str':
        if not isinstance(v, str) or not v.strip() or '${' in v:
            problems.append(f'{where}: must be a non-empty literal string, got {v!r}{hint}')
    elif kind == 'optstr':
        if v is not None and (not isinstance(v, str) or not v.strip()):
            problems.append(f'{where}: must be a non-empty string or null, got {v!r}{hint}')
    elif kind == 'runtime':
        if not isinstance(v, str) or not v.strip() or any(c in v for c in ' \'"`;|&'):
            problems.append(f'{where}: must be a string without spaces or quotes (may use ${{VAR}}), got {v!r}{hint}')
        elif '$' in RUNTIME_REF.sub('', v):
            problems.append(f'{where}: only ${{NAME}} references are allowed, got {v!r}{hint}')
    elif kind == 'strlist':
        if not (isinstance(v, list) and all(isinstance(x, str) and x.startswith('--') and '\n' not in x for x in v)):
            problems.append(f'{where}: must be a list of "--option[=value]" strings, got {v!r}{hint}')
    elif kind.startswith('choice:'):
        if v not in kind.split(':', 1)[1].split('|'):
            problems.append(f'{where}: must be one of {kind.split(":", 1)[1].replace("|", ", ")}, got {v!r}{hint}')


def _check_section(where: str, schema: dict, data, problems: list):
    if not isinstance(data, dict):
        problems.append(f'{where}: missing or not a mapping')
        return
    for k in data:
        if k not in schema:
            problems.append(f'{where}.{k}: unknown key (allowed: {", ".join(schema)})')
    for k, (kind, help_) in schema.items():
        if kind == 'section':
            continue
        if k not in data:
            problems.append(f'{where}.{k}: missing ({help_})')
        else:
            _check_value(f'{where}.{k}', kind, help_, data[k], problems)


def validate_operator(op) -> dict:
    """Refuse an incomplete or inconsistent operator config; return it (unchanged) if everything holds."""
    problems: list[str] = []
    if not isinstance(op, dict):
        raise OperatorError(['the operator file is not a YAML mapping'])
    for k in op:
        if k not in SCHEMA:
            problems.append(f'{k}: unknown section (allowed: {", ".join(SCHEMA)})')
    for sec, schema in SCHEMA.items():
        _check_section(sec, schema, op.get(sec), problems)
    sch = op.get('scheduler') if isinstance(op.get('scheduler'), dict) else {}
    if sch.get('type') == 'slurm':
        _check_section('scheduler.slurm', SLURM_SCHEMA, sch.get('slurm'), problems)
    elif sch.get('type') == 'none' and sch.get('slurm') is not None:
        problems.append('scheduler.slurm: must be absent or null when scheduler.type is none')
    if problems:
        raise OperatorError(problems)
    d, ln = op['deployment'], op['launch']
    if ln['nnodes'] * ln['nproc_per_node'] != d['data_parallel_size'] * S.TP * S.PP:
        problems.append(f'launch: nnodes {ln["nnodes"]} x nproc_per_node {ln["nproc_per_node"]} = '
                        f'{ln["nnodes"] * ln["nproc_per_node"]} processes, but deployment.data_parallel_size '
                        f'{d["data_parallel_size"]} x tp {S.TP} x pp {S.PP} = {d["data_parallel_size"] * S.TP * S.PP}')
    try:
        S.accumulation(d['micro_batch_size'], d['data_parallel_size'])
    except ValueError as e:
        problems.append(f'deployment: {e}')
    if sch.get('type') == 'slurm':
        sl = sch['slurm']
        if sl['gpus_per_node'] != ln['nproc_per_node']:
            problems.append(f'scheduler.slurm.gpus_per_node {sl["gpus_per_node"]} != launch.nproc_per_node {ln["nproc_per_node"]}')
        if not SLURM_TIME.match(sl['time']):
            problems.append(f'scheduler.slurm.time {sl["time"]!r}: use [D-]HH:MM:SS')
        if sl['account'] in PUBLISHER['slurm_accounts'] or (sl.get('qos') in PUBLISHER['slurm_qos']):
            problems.append('scheduler.slurm: account/qos are the publisher\'s (Skipjack) allocation; use your own')
    p = op['paths']
    if not (Path(p['CODE_ROOT']) / 'run_train.py').is_file() or not (Path(p['CODE_ROOT']) / 'deploy/portable_3b/kys3b.py').is_file():
        problems.append(f'paths.CODE_ROOT {p["CODE_ROOT"]}: no run_train.py / deploy/portable_3b/kys3b.py there')
    if not os.access(op['environment']['PYTHON'], os.X_OK):
        problems.append(f'environment.PYTHON {op["environment"]["PYTHON"]}: not an executable (build it with install_env.sh)')
    if problems:
        raise OperatorError(problems)
    return op


def load_operator(path: Path) -> dict:
    try:
        op = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise OperatorError([f'{path}: cannot read ({e})'])
    return validate_operator(op)


# ---- rendering --------------------------------------------------------------------------------------------------------
class Site:
    """Paths and deployment derived from a validated operator config."""

    def __init__(self, op: dict):
        self.op = op
        p = {k: Path(v) for k, v in op['paths'].items()}
        self.code, self.data, self.config_root = p['CODE_ROOT'], p['DATA_ROOT'], p['CONFIG_ROOT']
        self.init_root, self.ckpt_root = p['INIT_ROOT'], p['CHECKPOINT_ROOT']
        self.log_root, self.report_root, self.wandb_dir = p['LOG_ROOT'], p['REPORT_ROOT'], p['WANDB_DIR']
        self.python = op['environment']['PYTHON']
        self.entity, self.project = op['wandb']['WANDB_ENTITY'], op['wandb']['WANDB_PROJECT']
        d = op['deployment']
        self.dp, self.mbs, self.zero, self.recompute = (d['data_parallel_size'], d['micro_batch_size'], d['zero_stage'],
                                                       d['recompute_layer'])
        self.accum = S.accumulation(self.mbs, self.dp)
        self.grid_dir = self.config_root / 'grid'
        self.init_dir = self.config_root / 'init'
        self.run_dir = self.config_root / 'run'
        self.init_scratch = self.init_root / f'{S.INIT_DIRNAME}.run'

    @property
    def tokenized(self) -> Path:
        return self.data / 'tokenized'

    @property
    def tokenizer(self) -> Path:
        return self.data / 'tokenizer'

    def deployment(self) -> dict:
        """What a checkpoint chain must keep constant: the topology and the per-micro-batch loss normalization."""
        return {'data_parallel_size': self.dp, 'micro_batch_size': self.mbs, 'batch_accumulation_per_replica': self.accum,
                'zero_stage': self.zero, 'tp': S.TP, 'pp': S.PP, 'global_batch_seq': S.GLOBAL_BATCH_SEQ,
                'tokens_per_step': S.TOKENS_PER_STEP}

    def config(self, setting: str, kind: str) -> dict:
        return S.build_config(setting, kind, tokenized_root=self.tokenized, tokenizer_dir=self.tokenizer,
                              runs_root=self.ckpt_root, project=self.project, dp=self.dp, micro_batch_size=self.mbs,
                              zero_stage=self.zero, recompute_layer=self.recompute)

    def cfg_path(self, setting: str, kind: str) -> Path:
        return self.grid_dir / f'{S.seg_name(setting, kind)}.yaml'

    def env_text(self, group: str, job_type: str, tags: str) -> str:
        return ('# KYS 3B W&B environment, rendered by deploy/portable_3b/kys3b.py from the operator config; do not edit\n'
                f'export WANDB_ENTITY={self.entity}\nexport WANDB_PROJECT={self.project}\nexport WANDB_MODE=online\n'
                f'export WANDB_DIR={self.wandb_dir}\nexport WANDB_RUN_GROUP={group}\nexport WANDB_JOB_TYPE={job_type}\n'
                f'export WANDB_TAGS={tags}\n')

    def seg_env(self, setting: str, kind: str) -> str:
        tags = (f'grid:3b-rewrite600m,setting:{setting},seed:{S.SEED},kind:{kind},'
                f'phase:{"trunk" if kind.startswith("trunk") else "branch"},dp:{self.dp},mbs:{self.mbs},'
                f'accum:{self.accum},zero:{self.zero},tok_per_step:{S.TOKENS_PER_STEP}')
        return self.env_text(f'3b_{S.slug(setting)}_seed{S.SEED}', kind, tags)

    def site_env(self) -> str:
        ln = self.op['launch']
        q = lambda v: '"' + str(v).replace('"', '') + '"'  # noqa: E731 - values are validated (no quotes/spaces)
        rows = {'KYS3B_CODE_ROOT': self.code, 'KYS3B_PYTHON': self.python, 'KYS3B_CONFIG_ROOT': self.config_root,
                'KYS3B_OPERATOR': self.config_root / 'operator.resolved.yaml', 'KYS3B_DATA_ROOT': self.data,
                'KYS3B_INIT_ROOT': self.init_root, 'KYS3B_CHECKPOINT_ROOT': self.ckpt_root, 'KYS3B_LOG_ROOT': self.log_root,
                'KYS3B_REPORT_ROOT': self.report_root, 'KYS3B_WANDB_DIR': self.wandb_dir, 'KYS3B_NNODES': ln['nnodes'],
                'KYS3B_NPROC_PER_NODE': ln['nproc_per_node'], 'KYS3B_DP': self.dp, 'KYS3B_RDZV_BACKEND': ln['rdzv_backend'],
                'KYS3B_RDZV_ENDPOINT': ln['rdzv_endpoint'], 'KYS3B_RDZV_ID': ln['rdzv_id'],
                'KYS3B_SCHEDULER': self.op['scheduler']['type']}
        return ('# KYS 3B site environment, rendered by deploy/portable_3b/kys3b.py; do not edit (re-render instead).\n'
                '# Sourced by run_segment.sh / launch_torchrun.sh / create_init.sh. ${VAR} references resolve at launch.\n'
                + ''.join(f'export {k}={q(v)}\n' for k, v in rows.items()))


def plan(site: Site) -> dict:
    segs = []
    for s in S.GRID_SETTINGS:
        for k in S.KINDS:
            c = site.config(s, k)
            segs.append({'run': S.seg_name(s, k), 'setting': s, 'kind': k, 'config': str(site.cfg_path(s, k)),
                         'env': str(site.cfg_path(s, k).with_suffix('.env')),
                         'depends_on': S.seg_name(s, S.DEPENDS[k]) if S.DEPENDS[k] else 'init',
                         'first_step': S.FIRST_STEP[k], 'end_step': S.END_STEP[k], 'final_model': k in S.BRANCH_KINDS,
                         'final_tokens': S.END_STEP[k] * S.TOKENS_PER_STEP if k in S.BRANCH_KINDS else None,
                         'checkpoints_path': c['checkpoints']['checkpoints_path'],
                         'resume_from': c['checkpoints']['resume_checkpoint_path']})
    order = ['init'] + [x['run'] for k in S.TRUNK_KINDS + S.BRANCH_KINDS for x in segs if x['kind'] == k]
    return {'grid': 'KYS 3B rewrite-600m, seed 42, four settings', 'settings': S.GRID_SETTINGS,
            'n_segments': len(segs), 'n_final_models': sum(x['final_model'] for x in segs),
            'init': {'config': str(site.init_dir / 'init.yaml'), 'env': str(site.init_dir / 'init.env'),
                     'init_dir': str(initlib.init_parent(site.init_root)), 'shared_by': S.GRID_SETTINGS},
            'deployment': site.deployment(), 'order': order, 'segments': segs}


def _sbatch(site: Site, job: str, cmd: str) -> str:
    sl = site.op['scheduler']['slurm']
    ln = site.op['launch']
    lines = [f'--job-name={job}', f'--account={sl["account"]}', f'--partition={sl["partition"]}', f'--time={sl["time"]}',
             f'--nodes={ln["nnodes"]}', '--ntasks-per-node=1', f'--gpus-per-node={sl["gpus_per_node"]}',
             f'--cpus-per-task={sl["cpus_per_task"]}', f'--output={site.log_root}/slurm/%x.%j.log', '--open-mode=append']
    lines += [f'--qos={sl["qos"]}'] if sl.get('qos') else []
    lines += [f'--mem={sl["mem"]}'] if sl.get('mem') else []
    lines += [f'--constraint={sl["constraint"]}'] if sl.get('constraint') else []
    lines += ['--requeue'] if sl['requeue'] else ['--no-requeue']
    lines += list(sl.get('extra_sbatch') or [])
    return ('#!/usr/bin/env bash\n' + ''.join(f'#SBATCH {x}\n' for x in lines) +
            '# Rendered by deploy/portable_3b/kys3b.py render from the operator config; do not edit (re-render).\n'
            'set -euo pipefail\n'
            'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"\n'
            f'{cmd}\n')


def render_files(site: Site) -> dict[str, str]:
    """relative path under CONFIG_ROOT -> text. Deterministic for a given operator config and code."""
    files: dict[str, str] = {}
    for s in S.GRID_SETTINGS:
        for k in S.KINDS:
            c = site.config(s, k)
            lrs = c['optimizer']['learning_rate_scheduler']
            head = (f'# RENDERED by deploy/portable_3b/kys3b.py from the operator config — do not edit; re-render.\n'
                    f'# grid     : 3B rewrite-600m, setting {s}, seed {S.SEED}, segment {k} (steps {S.FIRST_STEP[k]}->{S.END_STEP[k]})\n'
                    f'# batch    : mbs {site.mbs} x accum {site.accum} x dp {site.dp} = {S.GLOBAL_BATCH_SEQ} seq = '
                    f'{S.TOKENS_PER_STEP:,} tokens/step (zero_stage {site.zero})\n'
                    f'# LR       : warmup 0->{S.WARMUP_STEPS} linear to {S.PEAK_LR}; constant; linear decay from step '
                    f'{lrs["lr_decay_starting_step"]} over {lrs["lr_decay_steps"]} steps -> 0\n'
                    f'# resume   : {c["checkpoints"]["resume_checkpoint_path"]}\n')
            files[f'grid/{S.seg_name(s, k)}.yaml'] = head + yaml.safe_dump(c, sort_keys=False)
            files[f'grid/{S.seg_name(s, k)}.env'] = site.seg_env(s, k)
    files['site.env'] = site.site_env()
    files['init/init.env'] = site.env_text(f'3b_init_seed{S.SEED}', 'init',
                                           f'grid:3b-rewrite600m,seed:{S.SEED},kind:init,dp:{site.dp},mbs:{site.mbs}')
    files['plan.json'] = json.dumps(plan(site), indent=1) + '\n'
    files['operator.resolved.yaml'] = yaml.safe_dump(site.op, sort_keys=False)
    if site.op['scheduler']['type'] == 'slurm':
        rs = f'bash {site.code}/deploy/portable_3b/run_segment.sh {site.config_root}/site.env'
        files['slurm/init.sbatch'] = _sbatch(site, 'kys3b_init', f'exec bash {site.code}/deploy/portable_3b/create_init.sh '
                                                                 f'{site.config_root}/site.env')
        for s in S.GRID_SETTINGS:
            for k in S.KINDS:
                files[f'slurm/{S.seg_name(s, k)}.sbatch'] = _sbatch(site, f'kys3b_{S.slug(s)}_{k}', f'exec {rs} {S.seg_name(s, k)}')
        sub = ['#!/usr/bin/env bash', '# Rendered by kys3b.py: submits the init job and the 4 x 9 segment jobs with afterok',
               '# dependencies (plan.json). Review it before running; nothing is submitted by render.', 'set -euo pipefail',
               f'D={site.config_root}/slurm', 'INIT=$(sbatch --parsable "$D/init.sbatch")', 'echo "init $INIT"']
        for s in S.GRID_SETTINGS:
            v = S.slug(s).upper()
            for k in S.KINDS:
                dep = '$INIT' if S.DEPENDS[k] is None else f'${v}_{S.DEPENDS[k].upper()}'
                sub.append(f'{v}_{k.upper()}=$(sbatch --parsable --dependency=afterok:{dep} "$D/{S.seg_name(s, k)}.sbatch")')
                sub.append(f'echo "{S.seg_name(s, k)} ${v}_{k.upper()}"')
        files['slurm/submit_grid.sh'] = '\n'.join(sub) + '\n'
    return files


def code_commit(code: Path) -> str | None:
    r = subprocess.run(['git', '-C', str(code), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def render(site: Site, force: bool = False) -> dict:
    files = render_files(site)
    man_p = site.config_root / 'render_manifest.json'
    if man_p.is_file() and not force:
        old = json.loads(man_p.read_text())
        changed = sorted(k for k in set(old['files']) | set(files)
                         if old['files'].get(k) != hashlib.sha256(files.get(k, '').encode()).hexdigest())
        changed = [k for k in changed if k not in ('init/init.yaml', 'init/init_derive.json')]
        if changed:
            raise SystemExit(f'{site.config_root} was rendered from a different operator config/code ({len(changed)} '
                             f'files differ, e.g. {changed[:3]}); re-render with --force only if no training used it')
    lock = site.ckpt_root / 'kys3b_deployment.json'
    if lock.is_file() and json.loads(lock.read_text())['deployment'] != site.deployment():
        raise SystemExit(f'{lock} records the deployment training already used {json.loads(lock.read_text())["deployment"]}; '
                         f'this operator config has {site.deployment()}. Checkpoints of a chain must keep one deployment.')
    if site.op['scheduler']['type'] == 'slurm':
        (site.log_root / 'slurm').mkdir(parents=True, exist_ok=True)     # sbatch --output needs it at submission
    for rel, text in files.items():
        p = site.config_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if rel.endswith('.sh') or rel.endswith('.sbatch'):
            p.chmod(0o755)
    rec = startup.derive_init(site.cfg_path(INIT_SOURCE_SETTING, 'trunk1'), site.init_dir / 'init.yaml', site.init_scratch)
    (site.init_dir / 'init_derive.json').write_text(json.dumps(rec, indent=1) + '\n')
    files['init/init.yaml'] = (site.init_dir / 'init.yaml').read_text()
    files['init/init_derive.json'] = (site.init_dir / 'init_derive.json').read_text()
    man = {'rendered': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'code_root': str(site.code), 'code_commit': code_commit(site.code),
           'deployment': site.deployment(), 'settings': S.GRID_SETTINGS,
           'files': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())}}
    man_p.write_text(json.dumps(man, indent=1) + '\n')
    return man


# ---- checks -----------------------------------------------------------------------------------------------------------
def _flat(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f'{pre}{k}.')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f'{pre}{i}.')
    else:
        yield pre[:-1], x


def science_problems(site: Site, cfgs: dict) -> list[str]:
    """Checks that do not trust science_3b.build_config: the architecture against the 1.5B config, exact batch
    accounting, schedule, branch sources, separate output namespaces, one deployment for the grid."""
    errs = []
    base = yaml.safe_load(BASELINE_1P5B.read_text())['model']
    base['model_config'] = {**base['model_config'], 'num_hidden_layers': S.N_LAYERS}
    outs = {}
    for (s, k), c in cfgs.items():
        n = S.seg_name(s, k)
        if c['model'] != base:
            errs.append(f'{n}: model block differs from the 1.5B config beyond num_hidden_layers = {S.N_LAYERS}')
        t, par = c['tokens'], c['parallelism']
        if t['micro_batch_size'] * t['batch_accumulation_per_replica'] * par['dp'] != S.GLOBAL_BATCH_SEQ \
                or t['sequence_length'] != S.SEQ_LEN \
                or t['micro_batch_size'] * t['batch_accumulation_per_replica'] * par['dp'] * t['sequence_length'] != S.TOKENS_PER_STEP:
            errs.append(f'{n}: batch is not 1024 x 2048 = {S.TOKENS_PER_STEP} tokens per update')
        if (par['tp'], par['pp']) != (S.TP, S.PP):
            errs.append(f'{n}: tp/pp {par["tp"]}/{par["pp"]}, established {S.TP}/{S.PP}')
        if c['general']['seed'] != S.SEED or c['data_stages'][0]['data']['seed'] != S.SEED:
            errs.append(f'{n}: seed is not {S.SEED}')
        lrs = c['optimizer']['learning_rate_scheduler']
        if (lrs['learning_rate'], lrs['lr_warmup_steps'], lrs['min_decay_lr']) != (S.PEAK_LR, S.WARMUP_STEPS, 0.0) \
                or c['optimizer']['optimizer_factory'] != S.ADAMW or c['optimizer']['weight_decay'] != S.WEIGHT_DECAY:
            errs.append(f'{n}: LR / AdamW settings differ from the contract')
        ck = c['checkpoints']
        if t['train_steps'] != S.END_STEP[k]:
            errs.append(f'{n}: train_steps {t["train_steps"]} != {S.END_STEP[k]}')
        if k in S.BRANCH_KINDS:
            b = S.BRANCH_POINTS[int(k[-1]) - 1]
            trunk = cfgs[(s, 'trunk1')]['checkpoints']['checkpoints_path']
            if ck['resume_checkpoint_path'] != f'{trunk}/{b}' or not (ck['load_optimizer'] and ck['load_lr_scheduler']):
                errs.append(f'{n}: must resume {trunk}/{b} with optimizer + LR state')
            if (lrs['lr_decay_starting_step'], lrs['lr_decay_steps']) != (b, S.END_STEP[k] - b):
                errs.append(f'{n}: decay must start at {b} and last {S.END_STEP[k] - b} steps')
            if ck['checkpoints_path'] == trunk or ck['checkpoints_path'].startswith(trunk + '/'):
                errs.append(f'{n}: writes into the trunk folder')
        else:
            if (lrs['lr_decay_starting_step'], lrs['lr_decay_steps']) != (S.BRANCH_POINTS[-1], S.HORIZONS[-1] - S.BRANCH_POINTS[-1]):
                errs.append(f'{n}: trunk decay parameters changed')
            if ck['resume_checkpoint_path'] != ck['checkpoints_path'] or (ck['load_optimizer'] != (k != 'trunk1')):
                errs.append(f'{n}: trunk resume rule broken')
        outs.setdefault(ck['checkpoints_path'], []).append(n)
        for p in (v for _, v in _flat(c) if isinstance(v, str) and v.startswith('/')):
            if any(p.startswith(x) for x in PUBLISHER['path_prefixes']):
                errs.append(f'{n}: publisher path {p}')
    if len(outs) != len(S.GRID_SETTINGS) * (1 + len(S.BRANCH_KINDS)):
        errs.append(f'{len(outs)} distinct output folders, expected {len(S.GRID_SETTINGS) * 4} (one trunk + 3 branches per setting)')
    if len({(c['tokens']['micro_batch_size'], c['parallelism']['dp'], c['optimizer']['zero_stage']) for c in cfgs.values()}) != 1:
        errs.append('the grid mixes deployments; one micro-batch size / dp / zero stage for all four settings')
    return errs


def _sha256(p: Path) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def corpus_problems(site: Site, hash_binaries: bool) -> list[str]:
    errs = []
    for n, want in S.TOKENIZER_SHA256.items():
        p = site.tokenizer / n
        if not p.is_file() or _sha256(p) != want:
            errs.append(f'{p}: missing or not the published tokenizer (sha256 {want[:12]}...)')
    man_p = site.data / 'RELEASE_MANIFEST.json'
    man = json.loads(man_p.read_text())['files'] if man_p.is_file() else None
    if man is None:
        errs.append(f'{man_p}: missing (download it with the data)')
    want_line = f'{site.tokenizer}|2'
    for s in S.GRID_SETTINGS:
        folder = site.tokenized / s
        shards = sorted(folder.glob('*.ds'))
        total, lines = 0, set()
        for ds in shards:
            meta = Path(f'{ds}.metadata')
            if not meta.is_file() or not Path(f'{ds}.index').is_file():
                errs.append(f'{ds}: missing .metadata or .index')
                continue
            ml = meta.read_text().splitlines()
            lines.add(ml[0])
            n = int(ml[1])
            total += n
            if ds.stat().st_size != 2 * n:
                errs.append(f'{ds}: {ds.stat().st_size} bytes != 2 x {n} tokens')
            for f in (ds, Path(f'{ds}.index')):
                rel = f'tokenized/{s}/{f.name}'
                if man is not None:
                    if rel not in man or man[rel]['bytes'] != f.stat().st_size:
                        errs.append(f'{f}: size differs from RELEASE_MANIFEST.json')
                    elif hash_binaries and _sha256(f) != man[rel]['sha256']:
                        errs.append(f'{f}: sha256 differs from RELEASE_MANIFEST.json')
        if len(shards) != S.CORPUS_PARTS[s]:
            errs.append(f'{folder}: {len(shards)} shards, expected {S.CORPUS_PARTS[s]}')
        if total != S.TOKENIZED_TOKENS[s]:
            errs.append(f'{folder}: {total:,} tokens, expected {S.TOKENIZED_TOKENS[s]:,}')
        if shards and lines != {want_line}:
            errs.append(f'{folder}: .ds.metadata line 1 is {sorted(lines)[:2]}, expected {want_line!r} '
                        f'(run: kys3b.py relocate-metadata --operator ...)')
    return errs


def check(site: Site, corpus: bool, hash_binaries: bool) -> dict:
    errs = []
    files = render_files(site)
    for rel, text in files.items():
        p = site.config_root / rel
        if not p.is_file() or p.read_text() != text:
            errs.append(f'{p}: missing or differs from a fresh render of the operator config (re-render)')
    init = site.init_dir / 'init.yaml'
    if init.is_file():
        tmp = site.report_root / '.check_init.yaml'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        startup.derive_init(site.cfg_path(INIT_SOURCE_SETTING, 'trunk1'), tmp, site.init_scratch)
        if tmp.read_text() != init.read_text():
            errs.append(f'{init}: differs from derive_init of the rendered trunk1 config (re-render)')
        tmp.unlink()
    else:
        errs.append(f'{init}: missing (render)')
    cfgs = {(s, k): site.config(s, k) for s in S.GRID_SETTINGS for k in S.KINDS}
    errs += science_problems(site, cfgs)
    if corpus:
        errs += corpus_problems(site, hash_binaries)
    rep = {'ok': not errs, 'problems': errs, 'n_configs': len(cfgs), 'settings': S.GRID_SETTINGS,
           'deployment': site.deployment(), 'corpus_checked': corpus, 'binaries_hashed': hash_binaries,
           'checked': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    site.report_root.mkdir(parents=True, exist_ok=True)
    (site.report_root / 'check.json').write_text(json.dumps(rep, indent=1) + '\n')
    return rep


# ---- per-segment preparation ------------------------------------------------------------------------------------------
def _segment(site: Site, run: str) -> tuple[str, str]:
    for s in S.GRID_SETTINGS:
        for k in S.KINDS:
            if S.seg_name(s, k) == run:
                return s, k
    raise SystemExit(f'{run!r} is not a segment of the 3B grid (see {site.config_root}/plan.json)')


def wandb_login_problem() -> str | None:
    if os.environ.get('WANDB_API_KEY'):
        return None
    host = urlparse(os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai')).hostname
    try:
        if netrc.netrc(os.environ.get('NETRC') or os.path.expanduser('~/.netrc')).authenticators(host):
            return None
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return (f'no W&B credentials for {host} (WANDB_API_KEY unset, no ~/.netrc entry): run `wandb login` with YOUR key '
            f'on this host; runs log online to the operator entity')


def ckpt_dp(step_dir: Path) -> int | None:
    m = step_dir / 'checkpoint_metadata.json'
    return json.loads(m.read_text()).get('dp') if m.is_file() else None


def prepare(site: Site, run: str) -> tuple[int, str]:
    """(exit code, effective config path or message). EXIT_COMPLETE if the final checkpoint exists."""
    s, k = _segment(site, run)
    cfg_p = site.cfg_path(s, k)
    files = render_files(site)
    for rel in (f'grid/{run}.yaml', f'grid/{run}.env', 'site.env'):
        p = site.config_root / rel
        if not p.is_file() or p.read_text() != files[rel]:
            raise SystemExit(f'{p}: missing or differs from a fresh render of the operator config; re-render')
    c = startup.load(cfg_p)
    ckpt, end = Path(c['checkpoints']['checkpoints_path']), S.END_STEP[k]
    if (ckpt / str(end) / 'model_config.json').is_file():
        return EXIT_COMPLETE, f'{ckpt}/{end} already written: {run} is complete'
    if (w := wandb_login_problem()):
        raise SystemExit(w)
    lock = site.ckpt_root / 'kys3b_deployment.json'
    if lock.is_file():
        have = json.loads(lock.read_text())['deployment']
        if have != site.deployment():
            raise SystemExit(f'{lock}: this grid trained with {have}; the operator config now says {site.deployment()}. '
                             f'A checkpoint chain keeps one deployment (topology and per-micro-batch loss normalization).')
    else:
        site.ckpt_root.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({'deployment': site.deployment(), 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    'first_run': run}, indent=1) + '\n')
    for d in (site.log_root, site.report_root, site.wandb_dir, site.run_dir):
        d.mkdir(parents=True, exist_ok=True)
    meta = site.tokenized / s
    first = next(iter(sorted(meta.glob('*.ds.metadata'))), None)
    if first is None or first.read_text().splitlines()[0] != f'{site.tokenizer}|2':
        raise SystemExit(f'{meta}: tokenized metadata missing or not relocated to {site.tokenizer} '
                         f'(run kys3b.py relocate-metadata, then check --corpus)')
    eff, note = cfg_p, ''
    if k == 'trunk1':
        seeded, init_m = False, None
        if not (ckpt / 'latest.txt').is_file():          # only a fresh trunk needs (and is seeded from) the init
            init_m = initlib.verify_init(site.init_root)
            par = initlib.init_parent(site.init_root)
            seeded = initlib.seed_trunk(ckpt, par / '0', par / initlib.MANIFEST)
        latest = int((ckpt / 'latest.txt').read_text().strip())
        if latest != 0:
            eff = site.run_dir / f'{run}.resume.yaml'
            startup.resume_config(cfg_p, eff)
            note = f'trunk1 continues {ckpt}/{latest} with optimizer + LR scheduler state'
        else:
            init_m = init_m or initlib.read_manifest(site.init_root)
            note = f'trunk1 starts weights-only from the shared init {init_m["weights_sha256"][:12]}' + (' (seeded now)' if seeded else '')
        src = ckpt / str(latest)
    elif k.startswith('trunk'):
        lt = ckpt / 'latest.txt'
        if not lt.is_file() or int(lt.read_text().strip()) < S.FIRST_STEP[k]:
            raise SystemExit(f'{lt}: missing or before step {S.FIRST_STEP[k]}; run {S.seg_name(s, S.DEPENDS[k])} first')
        src = ckpt / lt.read_text().strip()
        note = f'resumes {src} with optimizer + LR scheduler state'
    else:
        src = Path(c['checkpoints']['resume_checkpoint_path'])
        if not (src / 'model_config.json').is_file():
            raise SystemExit(f'{src}: branch source step {S.FIRST_STEP[k]} missing (run {S.seg_name(s, S.DEPENDS[k])}); '
                             f'a branch never starts from anything else')
        note = f'branches from {src} with optimizer + LR scheduler state into {ckpt}'
    have_dp = ckpt_dp(src) if startup.load(eff)['checkpoints']['load_optimizer'] else None   # weights-only: any dp
    if have_dp is not None and have_dp != site.dp:
        raise SystemExit(f'{src} was written with data-parallel size {have_dp}; this deployment uses {site.dp}')
    rec = {'run': run, 'effective_config': str(eff), 'note': note, 'source': str(src), 'deployment': site.deployment(),
           'prepared': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'job': os.environ.get('SLURM_JOB_ID')}
    (site.report_root / 'prepare').mkdir(parents=True, exist_ok=True)
    (site.report_root / 'prepare' / f'{run}.{int(time.time())}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'[kys3b] {run}: {note}', file=sys.stderr)
    return 0, str(eff)


def finish(site: Site, run: str) -> tuple[int, str]:
    s, k = _segment(site, run)
    ckpt = Path(site.config(s, k)['checkpoints']['checkpoints_path'])
    end = S.END_STEP[k]
    ok = (ckpt / str(end) / 'model_config.json').is_file()
    if ok and k.startswith('trunk'):
        ok = (ckpt / 'latest.txt').is_file() and int((ckpt / 'latest.txt').read_text().strip()) == end
    return (0, f'{run} complete: {ckpt}/{end}') if ok else (1, f'{run}: {ckpt}/{end} missing after training; '
                                                               f'dependent segments must not start')


# ---- CLI --------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['validate', 'render', 'relocate-metadata', 'check', 'plan', 'init-config',
                                        'finalize-init', 'seed', 'prepare', 'finish', 'startup-derive',
                                        'startup-summarize', 'check-log', 'wandb-check'])
    ap.add_argument('--operator', type=Path)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--run')
    ap.add_argument('--workdir', type=Path)
    ap.add_argument('--log', type=Path)
    ap.add_argument('--mem-budget-gib', type=float)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--corpus', action='store_true')
    ap.add_argument('--hash', action='store_true')
    a = ap.parse_args(argv)
    needs = {'init-config': ['config', 'out', 'ckpt'], 'check-log': ['config', 'log'], 'wandb-check': [],
             'seed': ['operator', 'config'],
             'prepare': ['operator', 'run'], 'finish': ['operator', 'run'], 'startup-derive': ['operator', 'workdir'],
             'startup-summarize': ['operator', 'workdir']}.get(a.command, ['operator'])
    missing = [f'--{n}' for n in needs if getattr(a, n) is None]
    if missing:
        ap.error(f'{a.command} needs {" ".join(missing)}')
    if a.command == 'init-config':
        print(json.dumps(startup.derive_init(a.config, a.out, a.ckpt), indent=1))
        return 0
    if a.command == 'wandb-check':
        w = wandb_login_problem()
        print(w or '[kys3b] W&B credentials present (WANDB_API_KEY or netrc); the entity comes from the rendered .env')
        return 1 if w else 0
    if a.command == 'check-log':
        r = startup.check_log(a.config, a.log)
        print(json.dumps(r, indent=1))
        return 1 if r['problems'] else 0
    site = Site(load_operator(a.operator))
    if a.command == 'validate':
        print(f'[kys3b] operator config OK: {site.deployment()}')
    elif a.command == 'render':
        man = render(site, a.force)
        print(f'[kys3b] rendered {len(man["files"])} files to {site.config_root}: 4 settings x {len(S.KINDS)} segments '
              f'(+ init), mbs {site.mbs} x accum {site.accum} x dp {site.dp} = {S.GLOBAL_BATCH_SEQ}')
    elif a.command == 'relocate-metadata':
        for s in S.GRID_SETTINGS:
            subprocess.run([sys.executable, str(site.code / 'tools' / 'fix_ds_metadata.py'), '--output-folder',
                            str(site.tokenized / s), '--tokenizer-dir', str(site.tokenizer)], check=True)
    elif a.command == 'check':
        rep = check(site, a.corpus, a.hash)
        for e in rep['problems']:
            print(f'[kys3b] PROBLEM {e}')
        print(f'[kys3b] {"OK" if rep["ok"] else "FAILED"}: {rep["n_configs"]} configs of {len(S.GRID_SETTINGS)} settings'
              + (' + corpus' + (' (binaries hashed)' if a.hash else ' (sizes, totals, metadata)') if a.corpus else ''))
        return 0 if rep['ok'] else 1
    elif a.command == 'plan':
        print(json.dumps(plan(site), indent=1))
    elif a.command == 'finalize-init':
        rec = json.loads((site.init_dir / 'init_derive.json').read_text())
        m = initlib.finalize_init(site.init_scratch, site.init_root,
                                  {'init_config': str(site.init_dir / 'init.yaml'),
                                   'init_config_sha256': startup.sha256_file(site.init_dir / 'init.yaml'),
                                   'derived_from': rec, 'code_commit': code_commit(site.code),
                                   'deployment': site.deployment(), 'job': os.environ.get('SLURM_JOB_ID')})
        print(json.dumps(m, indent=1))
    elif a.command == 'seed':
        trunk = Path(startup.load(a.config)['checkpoints']['checkpoints_path'])
        par = initlib.init_parent(site.init_root)
        initlib.verify_init(site.init_root)
        print('seeded' if initlib.seed_trunk(trunk, par / '0', par / initlib.MANIFEST) else 'already seeded')
    elif a.command == 'prepare':
        rc, msg = prepare(site, a.run)
        print(msg)
        return rc
    elif a.command == 'finish':
        rc, msg = finish(site, a.run)
        print(msg)
        return rc
    elif a.command == 'startup-derive':
        startup.derive_startup(site.cfg_path(INIT_SOURCE_SETTING, 'trunk1'), a.workdir)
        print(f'[kys3b] wrote {startup.STARTUP_PHASES} to {a.workdir}/cfg')
    elif a.command == 'startup-summarize':
        rep = startup.summarize(site.cfg_path(INIT_SOURCE_SETTING, 'trunk1'), a.workdir, a.mem_budget_gib)
        print(json.dumps({k: v for k, v in rep.items() if k != 'phases'}, indent=1))
        return 0 if rep['ok'] else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
