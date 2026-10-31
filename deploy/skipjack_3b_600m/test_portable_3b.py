"""Portable rendering of the four-setting 3B grid (portable_configs_3b.py): CPU only, queues nothing.

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_portable_3b.py

Asserts:
  * with no KYS3B_* overrides the registry resolves to the Skipjack paths, so the committed configs are what it renders;
  * a portable render changes only the tokenizer, tokenized-folder, checkpoint and W&B paths and the W&B entity/project
    (36 configs + .env), logs online, and names nothing of the Skipjack owner's W&B account;
  * the W&B entity and project are required, and the Skipjack owner's entity is refused;
  * those rendered configs pass assert_invariants_3b --dir under the printed overrides (architecture vs the 1.5B config,
    batch, LR schedule, branch points, finals, resume rules), and the scientific constants are unchanged by overrides;
  * a render that would change a non-path key is refused and leaves no files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portable_configs_3b as PC  # noqa: E402
import registry_3b as R  # noqa: E402

ENTITY, PROJECT = 'example-entity', 'example-project'
WB = ['--wandb-entity', ENTITY, '--wandb-project', PROJECT]
SCIENCE = ('SEED', 'GRID_SETTINGS', 'HORIZONS', 'BRANCH_POINTS', 'TRUNK_ENDS', 'PEAK_LR', 'WARMUP_STEPS', 'TOKENS_PER_STEP',
           'MODEL', 'N_PARAMS', 'TOKENIZER_SHA256', 'CORPUS_TRAIN_TOKENS', 'TOKENIZED_TOKENS', 'LAYOUTS')


def _registry(env: dict) -> dict:
    code = (f'import sys, json; sys.path.insert(0, {str(HERE)!r}); import registry_3b as R\n'
            f'print(json.dumps({{k: getattr(R, k) for k in {SCIENCE!r}}}, default=str))\n'
            'print(json.dumps([str(R.TOKENIZED), str(R.TOKENIZER), str(R.CKPT_BASE), str(R.WANDB_DIR)]))')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    return {'science': json.loads(out[0]), 'paths': json.loads(out[1])}


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith('KYS3B_')}


def test_defaults_are_skipjack():
    r = _registry(_clean_env())
    assert r['paths'] == ['/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/tokenized',
                          '/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/tokenizer',
                          '/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m',
                          '/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/wandb']


def test_portable_render_changes_only_paths(tmp_path):
    data, ckpt, out = tmp_path / 'data', tmp_path / 'ckpt', tmp_path / 'cfg'
    PC.main(['--data', str(data), '--ckpt', str(ckpt), *WB, '--out', str(out)])
    ov = PC.overrides(data.resolve(), ckpt.resolve(), None, ENTITY, PROJECT)
    assert _registry({**_clean_env(), **ov})['science'] == _registry(_clean_env())['science']
    ymls = sorted(out.glob('*.yaml'))
    assert len(ymls) == len(sorted(out.glob('*.env'))) == len(R.GRID_SETTINGS) * len(R.KINDS)
    for p in ymls:
        c = yaml.safe_load(p.read_text())
        assert c['tokenizer']['tokenizer_name_or_path'] == ov['KYS3B_TOKENIZER']
        assert c['data_stages'][0]['data']['dataset']['dataset_folder'][0].startswith(ov['KYS3B_TOKENIZED'] + '/')
        assert c['checkpoints']['checkpoints_path'].startswith(ov['KYS3B_CKPT_BASE'] + '/runs/seed42/')
        assert not any(s.startswith('/weka/') for s in PC._flat(c) if isinstance(s, str)), p.name
        assert c['general']['project'] == PROJECT
        env = PC._env(p.with_suffix('.env').read_text())
        assert (env['WANDB_ENTITY'], env['WANDB_PROJECT'], env['WANDB_MODE']) == (ENTITY, PROJECT, 'online')
        assert R.SKIPJACK_WANDB_ENTITY not in p.read_text() + p.with_suffix('.env').read_text(), p.name
    r = subprocess.run([sys.executable, HERE / 'assert_invariants_3b.py', '--dir', out], env={**_clean_env(), **ov},
                       capture_output=True, text=True)
    assert r.returncode == 0 and f'OK: {len(ymls)} config(s)' in r.stdout, (r.stdout, r.stderr)


def test_refuses_non_path_difference(tmp_path, monkeypatch):
    real = PC.render

    def tampered(ov, out):
        meta = real(ov, out)
        p = out / '3b_quality_base_seed42_ep1.yaml'
        p.write_text(p.read_text().replace('learning_rate: 0.0004', 'learning_rate: 0.0003'))
        return meta
    monkeypatch.setattr(PC, 'render', tampered)
    with pytest.raises(SystemExit, match='non-path keys differ'):
        PC.main(['--data', str(tmp_path / 'd'), '--ckpt', str(tmp_path / 'c'), *WB, '--out', str(tmp_path / 'o')])
    assert not list((tmp_path / 'o').glob('3b_*'))


@pytest.mark.parametrize('wb, match', [([], None), (['--wandb-entity', ENTITY], None),
                                       (['--wandb-entity', 'imwytro-johns-hopkins-university', '--wandb-project', PROJECT],
                                        'Skipjack owner'),
                                       (['--wandb-entity', '', '--wandb-project', PROJECT], 'non-empty')])
def test_wandb_destination_required(tmp_path, wb, match):
    with pytest.raises(SystemExit) as e:
        PC.main(['--data', str(tmp_path / 'd'), '--ckpt', str(tmp_path / 'c'), *wb, '--out', str(tmp_path / 'o')])
    if match:
        assert match in str(e.value)
    assert not (tmp_path / 'o').exists() or not list((tmp_path / 'o').glob('3b_*'))
