"""Fixtures of the portable CPU tests: an operator config whose every path is under a fresh temporary root (none of
the publisher's), with its own W&B entity/project and a deployment unlike Skipjack's (dp 16 x mbs 4 x accum 16)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CODE = HERE.parents[1]
ENTITY, PROJECT = 'marc-lab-test', 'kys3b-portable-test'


def operator_dict(root: Path) -> dict:
    """Code and Python are reached through symlinks under `root`, so even they are not publisher paths."""
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (('code', CODE), ('python', Path(sys.executable))):
        if not (root / name).exists():
            (root / name).symlink_to(target)
    m = root / 'site'
    return {
        'paths': {'CODE_ROOT': str(root / 'code'), 'DATA_ROOT': str(m / 'data'), 'CONFIG_ROOT': str(m / 'cfg'),
                  'INIT_ROOT': str(m / 'init'), 'CHECKPOINT_ROOT': str(m / 'ckpt'), 'LOG_ROOT': str(m / 'logs'),
                  'REPORT_ROOT': str(m / 'reports'), 'WANDB_DIR': str(m / 'wandb')},
        'environment': {'PYTHON': str(root / 'python')},
        'wandb': {'WANDB_ENTITY': ENTITY, 'WANDB_PROJECT': PROJECT},
        'deployment': {'data_parallel_size': 16, 'micro_batch_size': 4, 'zero_stage': 1, 'recompute_layer': True},
        'launch': {'nnodes': 2, 'nproc_per_node': 8, 'rdzv_backend': 'c10d', 'rdzv_endpoint': '${MASTER_ADDR}:29500',
                   'rdzv_id': '${SLURM_JOB_ID}'},
        'scheduler': {'type': 'slurm', 'slurm': {'account': 'marc-acct', 'partition': 'gpu', 'qos': None,
                                                 'time': '2-00:00:00', 'gpus_per_node': 8, 'cpus_per_task': 64,
                                                 'mem': None, 'constraint': None, 'requeue': True, 'extra_sbatch': []}},
    }


def with_changes(op: dict, changes: dict) -> dict:
    """changes: {'section.key[.sub]': value}; value None under DELETE removes the key."""
    op = copy.deepcopy(op)
    for dotted, v in changes.items():
        *path, leaf = dotted.split('.')
        d = op
        for k in path:
            d = d[k]
        if v is DELETE:
            d.pop(leaf, None)
        else:
            d[leaf] = v
    return op


DELETE = object()


def write(op: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(op, sort_keys=False))
    return path


@pytest.fixture
def op_root(tmp_path):
    return tmp_path


@pytest.fixture
def operator(tmp_path):
    op = operator_dict(tmp_path)
    return op, write(op, tmp_path / 'operator.yaml')


def fake_tokenized(site, settings=None) -> None:
    """Relocated .ds.metadata stubs (what prepare's quick data check reads); no token data."""
    import science_3b as S
    for s in settings or S.GRID_SETTINGS:
        d = site.tokenized / s
        d.mkdir(parents=True, exist_ok=True)
        (d / '00000_unshuffled.ds.metadata').write_text(f'{site.tokenizer}|2\n0\n')


def fake_step(d: Path, dp: int, marker: str = 'w') -> Path:
    """A minimal nanotron step directory: model_config.json, weights, optimizer, lr_scheduler, checkpoint metadata."""
    import json
    for f, text in (('model_config.json', '{}'), ('model/model/w.safetensors', f'weights {marker}'),
                    ('optimizer/optimizer.pt', f'opt {marker}'), ('lr_scheduler/lr.pt', f'lr {marker}')):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(text)
    (d / 'checkpoint_metadata.json').write_text(json.dumps({'dp': dp, 'tp': 1}))
    return d
