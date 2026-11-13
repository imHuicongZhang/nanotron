"""Opt-in raw-baseline settings (raw_diversity_oriented, raw_disagreement_aware) of the portable 3B deployment (CPU only).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_portable_3b_raw.py

Asserts:
  * selection: only via operator grid.settings; never mixed with the four-setting grid; a raw selection requires
    grid.init_weights_sha256 (the EXISTING shared init); unknown grid keys are refused;
  * render: 2 x 9 = 18 configs and 6 annealed finals; every config's science (science_view without general.run) is
    IDENTICAL to its rewritten counterpart's (same architecture, seed 42, AdamW, LR schedule, global batch, branch
    points 8583/17166/25749, horizons 9537/19073/28610); data folder tokenized/<raw setting>; separate output folders;
  * init: no init job is rendered, submit_grid.sh runs verify-init first and trunk1 has no init dependency,
    finalize-init and create_init.sh refuse to create an init, prepare seeds trunk1 only from the pinned init;
  * the default (no grid section) render is unchanged by the raw extension (same files as the four-setting render).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest
import yaml

from conftest import CODE, fake_step, fake_tokenized, operator_dict, with_changes, write
import initlib
import kys3b as K
import science_3b as S

PIN = 'a' * 64


def _raw_op(tmp_path, settings=None, pin=PIN):
    op = operator_dict(tmp_path)
    op['grid'] = {'settings': list(settings or S.RAW_SETTINGS)}
    if pin is not None:
        op['grid']['init_weights_sha256'] = pin
    return op


def _site(tmp_path, op):
    s = K.Site(K.load_operator(write(op, tmp_path / 'operator.yaml')))
    s.path = tmp_path / 'operator.yaml'
    return s


@pytest.mark.parametrize('grid, needle', [
    ({'settings': S.RAW_SETTINGS}, 'grid.init_weights_sha256'),
    ({'settings': S.RAW_SETTINGS, 'init_weights_sha256': 'xyz'}, 'grid.init_weights_sha256'),
    ({'settings': S.GRID_SETTINGS + S.RAW_SETTINGS, 'init_weights_sha256': PIN}, 'never mixed'),
    ({'settings': ['quality-first', 'raw_diversity_oriented'], 'init_weights_sha256': PIN}, 'never mixed'),
    ({'settings': ['raw-diversity-oriented'], 'init_weights_sha256': PIN}, 'never mixed'),
    ({'settings': [], 'init_weights_sha256': PIN}, 'non-empty list'),
    ({'settings': S.RAW_SETTINGS, 'init_weights_sha256': PIN, 'seed': 7}, 'grid.seed: unknown key'),
    ('raw', 'must be a mapping'),
])
def test_grid_selection_refusals(tmp_path, grid, needle):
    op = operator_dict(tmp_path)
    op['grid'] = grid
    with pytest.raises(K.OperatorError) as e:
        K.load_operator(write(op, tmp_path / 'operator.yaml'))
    assert needle in str(e.value.problems), e.value.problems


def test_constants():
    assert S.RAW_SETTINGS == ['raw_diversity_oriented', 'raw_disagreement_aware']
    assert not set(S.RAW_SETTINGS) & set(S.GRID_SETTINGS) and len(S.GRID_SETTINGS) == 4
    assert S.CORPUS_ROWS['raw_diversity_oriented'] == 16_148_288 and S.CORPUS_ROWS['raw_disagreement_aware'] == 15_880_055
    assert S.TOKENIZED_TOKENS['raw_diversity_oriented'] == 20_000_001_178
    assert S.TOKENIZED_TOKENS['raw_disagreement_aware'] == 20_000_007_963
    for s in S.RAW_SETTINGS:
        assert S.CORPUS_PARTS[s] == -(-S.CORPUS_ROWS[s] // 500_000)
        assert S.CORPUS_ROWS[s] == 10_527_887 + S.RAW_STRATEGY_BLOCK[s]['docs']


def test_raw_render_matches_counterpart_science(tmp_path):
    raw = _site(tmp_path / 'raw', _raw_op(tmp_path / 'raw'))
    man = K.render(raw)
    grid = _site(tmp_path / 'grid', operator_dict(tmp_path / 'grid'))
    K.render(grid)
    assert man['settings'] == S.RAW_SETTINGS
    assert len(list(raw.grid_dir.glob('*.yaml'))) == 18 and len(list(raw.grid_dir.glob('*.env'))) == 18
    plan = json.loads((raw.config_root / 'plan.json').read_text())
    assert plan['n_segments'] == 18 and plan['n_final_models'] == 6 and plan['settings'] == S.RAW_SETTINGS
    assert plan['init']['weights_sha256'] == PIN and 'reuse' in plan['init']['mode']
    assert [x['run'] for x in plan['segments'] if x['final_model']] == [S.seg_name(s, k) for s in S.RAW_SETTINGS for k in S.BRANCH_KINDS]
    assert sorted(x['final_tokens'] for x in plan['segments'] if x['final_model']) == sorted(S.FINAL_TOKENS * 2)
    outs = set()
    for s in S.RAW_SETTINGS:
        cp = S.RAW_COUNTERPART[s]
        for k in S.KINDS:
            mine = yaml.safe_load(raw.cfg_path(s, k).read_text())
            ref = yaml.safe_load(grid.cfg_path(cp, k).read_text())
            a, b = S.science_view(mine), S.science_view(ref)
            assert a['general'].pop('run') == S.seg_name(s, k) and b['general'].pop('run') == S.seg_name(cp, k)
            assert a == b, (s, k)
            assert mine['data_stages'][0]['data']['dataset']['dataset_folder'] == [str(raw.tokenized / s)]
            assert mine['general']['seed'] == 42 and mine['tokens']['train_steps'] == S.END_STEP[k]
            outs.add(mine['checkpoints']['checkpoints_path'])
            if k in S.BRANCH_KINDS:
                assert mine['checkpoints']['resume_checkpoint_path'].endswith(f'/trunk/{s}/seed42/{S.FIRST_STEP[k]}')
    assert len(outs) == 8 and not outs & {yaml.safe_load(p.read_text())['checkpoints']['checkpoints_path']
                                          for p in grid.grid_dir.glob('*.yaml')}
    assert K.check(raw, corpus=False, hash_binaries=False)['ok']


def test_raw_render_never_creates_an_init(tmp_path):
    raw = _site(tmp_path, _raw_op(tmp_path))
    K.render(raw)
    slurm = raw.config_root / 'slurm'
    assert not (slurm / 'init.sbatch').exists()
    sub = (slurm / 'submit_grid.sh').read_text()
    assert 'verify-init' in sub and 'init.sbatch' not in sub and '$INIT' not in sub
    lines = [x for x in sub.splitlines() if 'sbatch --parsable' in x]
    assert len(lines) == 18
    for s in S.RAW_SETTINGS:
        t1 = next(x for x in lines if f'/{S.seg_name(s, "trunk1")}.sbatch' in x)
        assert '--dependency' not in t1
        e1 = next(x for x in lines if f'/{S.seg_name(s, "ep1")}.sbatch' in x)
        assert f'afterok:${S.slug(s).upper()}_TRUNK2' in e1
    env = (raw.config_root / 'site.env').read_text()
    assert 'KYS3B_INIT_MODE="reuse"' in env and f'KYS3B_INIT_WEIGHTS_SHA256="{PIN}"' in env
    with pytest.raises(SystemExit, match='refused for raw-baseline settings'):
        K.main(['finalize-init', '--operator', str(raw.path)])


def test_create_init_script_refuses_in_reuse_mode(tmp_path):
    raw = _site(tmp_path, _raw_op(tmp_path))
    K.render(raw)
    r = subprocess.run(['bash', str(CODE / 'deploy/portable_3b/create_init.sh'), str(raw.config_root / 'site.env')],
                       capture_output=True, text=True, env={**os.environ, 'MASTER_ADDR': 'localhost', 'SLURM_JOB_ID': '1'})
    assert r.returncode != 0 and 'never created' in r.stderr, r.stdout + r.stderr
    assert not initlib.init_parent(raw.init_root).exists()


def _make_init(site):
    fake_step(site.init_scratch / '0', site.dp, 'init')
    return initlib.finalize_init(site.init_scratch, site.init_root, {'test': True})


def test_prepare_seeds_raw_trunks_only_from_the_pinned_init(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY', 'test-not-a-key')
    grid = _site(tmp_path, operator_dict(tmp_path))      # the four-setting grid created the init on this site
    K.render(grid)
    m = _make_init(grid)
    run = S.seg_name('raw_diversity_oriented', 'trunk1')
    # wrong pin: refused, nothing seeded
    bad = with_changes(_raw_op(tmp_path), {'paths.CONFIG_ROOT': str(tmp_path / 'site' / 'cfg_raw_bad')})
    bsite = _site(tmp_path, bad)
    K.render(bsite)
    fake_tokenized(bsite, S.RAW_SETTINGS)
    with pytest.raises(SystemExit, match='grid.init_weights_sha256'):
        K.prepare(bsite, run)
    assert not (S.ckpt_dir(bsite.ckpt_root, 'raw_diversity_oriented', 'trunk1') / 'latest.txt').exists()
    with pytest.raises(SystemExit, match='grid.init_weights_sha256'):
        K.main(['verify-init', '--operator', str(bsite.path)])
    # the correct pin: trunk1 seeded with exactly the shared init's weights
    good = with_changes(_raw_op(tmp_path, pin=m['weights_sha256']), {'paths.CONFIG_ROOT': str(tmp_path / 'site' / 'cfg_raw')})
    rsite = _site(tmp_path, good)
    K.render(rsite)
    assert K.main(['verify-init', '--operator', str(rsite.path)]) == 0
    for s in S.RAW_SETTINGS:
        rc, eff = K.prepare(rsite, S.seg_name(s, 'trunk1'))
        assert rc == 0 and eff == str(rsite.cfg_path(s, 'trunk1'))
        trunk = S.ckpt_dir(rsite.ckpt_root, s, 'trunk1')
        assert initlib.init_digest(trunk / '0')['weights_sha256'] == m['weights_sha256']
    with pytest.raises(SystemExit, match='not a segment'):
        K.prepare(rsite, S.seg_name('quality-first', 'trunk1'))


def test_raw_trunk1_without_init_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY', 'test-not-a-key')
    rsite = _site(tmp_path, _raw_op(tmp_path))
    K.render(rsite)                                      # render/check work without any init on this machine
    assert K.check(rsite, corpus=False, hash_binaries=False)['ok']
    fake_tokenized(rsite, S.RAW_SETTINGS)
    with pytest.raises(SystemExit):
        K.prepare(rsite, S.seg_name('raw_disagreement_aware', 'trunk1'))
    assert not initlib.init_parent(rsite.init_root).exists()


def test_default_render_unchanged(tmp_path):
    """Without a grid section the render holds exactly the four-setting files (no raw file, no reuse-mode line)."""
    site = _site(tmp_path, operator_dict(tmp_path))
    man = K.render(site)
    assert man['settings'] == S.GRID_SETTINGS and len([f for f in man['files'] if f.startswith('grid/')]) == 72
    assert not [f for f in man['files'] if 'raw_' in f]
    assert 'KYS3B_INIT_MODE' not in (site.config_root / 'site.env').read_text()
    assert (site.config_root / 'slurm' / 'init.sbatch').is_file()
