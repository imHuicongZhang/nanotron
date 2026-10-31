"""Initialization, seeding and segment preparation from a PORTABLE config (CPU only; no GPU, nothing queued).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_portable_3b_init.py

Regression for the reviewed bug: startup_3b.py derive-init accepted --config but read the committed Skipjack
quality-first trunk1, so the init kept Skipjack tokenizer/data paths and the old W&B project. Here the actual CLIs
(kys3b.py init-config and the Skipjack startup_3b.py derive-init) run on a temporary portable config whose paths and
W&B values exist nowhere else, and the derived config must keep every value of THAT config except the documented init
keys, with no Skipjack path. Then, with stub checkpoints:
  * finalize-init creates <INIT_ROOT>/_init_3B_seed42 once (never overwrites; refuses a partial one);
  * trunk1 of all four settings is seeded from that one init (identical weights digest) and never over progress;
  * prepare: trunk1 at step 0 runs its weights-only config; trunk1 past step 0 (requeue / after the startup run)
    gets a config with optimizer + LR state; trunk2..6 need latest.txt >= their first step; a branch needs its fixed
    source step (8583/17166/25749); a resume source written at another dp, a changed deployment, a drifted config,
    unrelocated data metadata and a missing W&B login are refused; a finished segment reports complete;
  * startup derive / summarize read the supplied config (trunk folder = its checkpoints_path).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from conftest import CODE, ENTITY, PROJECT, fake_step, fake_tokenized, operator_dict, with_changes, write
import initlib
import kys3b as K
import science_3b as S
import startup

SKIPJACK_MARKERS = ('/weka/', '/home/jhu/', 'zhc-3b-600m', 'imwytro')


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY', 'test-not-a-key')
    path = write(operator_dict(tmp_path), tmp_path / 'operator.yaml')
    s = K.Site(K.load_operator(path))
    K.render(s)
    fake_tokenized(s)
    s.path = path
    return s


def _no_skipjack(c: dict):
    vals = [v for _, v in K._flat(c) if isinstance(v, str)]
    assert not [v for v in vals if any(m in v for m in SKIPJACK_MARKERS)], vals


@pytest.mark.parametrize('cli', ['portable', 'skipjack'])
def test_init_config_cli_honours_the_supplied_config(site, tmp_path, cli):
    src = site.cfg_path('quality-base', 'trunk1')            # any setting's rendered trunk1, deliberately not quality-first
    out, ckpt = tmp_path / f'init_{cli}.yaml', tmp_path / 'distinct_init_scratch'
    argv = ([sys.executable, str(CODE / 'deploy/portable_3b/kys3b.py'), 'init-config'] if cli == 'portable' else
            [sys.executable, str(CODE / 'deploy/skipjack_3b_600m/startup_3b.py'), 'derive-init'])
    r = subprocess.run(argv + ['--config', str(src), '--out', str(out), '--ckpt', str(ckpt)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    prod, init = yaml.safe_load(src.read_text()), yaml.safe_load(out.read_text())
    ov = startup.init_overrides(ckpt)
    changed = {k for k, v in K._flat(init) if dict(K._flat(prod)).get(k, object()) != v}
    assert changed <= set(ov), changed - set(ov)
    for k, v in ov.items():
        assert dict(K._flat(init)).get(k, v) == v or (v is None and k in ('checkpoints.resume_checkpoint_path',))
    assert init['tokenizer']['tokenizer_name_or_path'] == str(site.tokenizer)
    assert init['data_stages'][0]['data']['dataset']['dataset_folder'] == [str(site.tokenized / 'quality-base')]
    assert init['general']['project'] == PROJECT and init['general']['seed'] == 42
    assert init['checkpoints']['checkpoints_path'] == str(ckpt) and init['checkpoints']['save_initial_state'] is True
    assert init['tokens']['train_steps'] == 1
    assert init['parallelism'] == prod['parallelism'] and init['tokens']['micro_batch_size'] == site.mbs
    assert init['model'] == prod['model'] and init['optimizer'] == prod['optimizer']
    _no_skipjack(init)


def test_rendered_init_is_derived_from_the_rendered_config(site):
    init = yaml.safe_load((site.init_dir / 'init.yaml').read_text())
    env = (site.init_dir / 'init.env').read_text()
    assert init['data_stages'][0]['data']['dataset']['dataset_folder'] == [str(site.tokenized / 'quality-first')]
    assert init['checkpoints']['checkpoints_path'] == str(site.init_scratch)
    assert f'WANDB_ENTITY={ENTITY}' in env and f'WANDB_PROJECT={PROJECT}' in env and 'WANDB_MODE=online' in env
    _no_skipjack(init)
    rec = json.loads((site.init_dir / 'init_derive.json').read_text())
    assert rec['source_config'] == str(site.cfg_path('quality-first', 'trunk1').resolve())


def test_derive_init_refuses_a_non_trunk1_config(site, tmp_path):
    with pytest.raises(SystemExit, match='not a rendered trunk1 config'):
        startup.derive_init(site.cfg_path('quality-first', 'trunk2'), tmp_path / 'x.yaml', tmp_path / 'c')


def _make_init(site):
    fake_step(site.init_scratch / '0', site.dp, 'init')
    return initlib.finalize_init(site.init_scratch, site.init_root, {'test': True})


def test_finalize_once_and_seed_all_four_identically(site):
    m = _make_init(site)
    par = initlib.init_parent(site.init_root)
    assert (par / initlib.MANIFEST).is_file() and initlib.verify_init(site.init_root)['weights_sha256'] == m['weights_sha256']
    with pytest.raises(SystemExit, match='already holds a finished init'):
        initlib.finalize_init(site.init_scratch, site.init_root, {})
    digests = set()
    for s in S.GRID_SETTINGS:
        rc, eff = K.prepare(site, S.seg_name(s, 'trunk1'))
        assert rc == 0 and eff == str(site.cfg_path(s, 'trunk1'))          # weights-only rendered config at step 0
        trunk = S.ckpt_dir(site.ckpt_root, s, 'trunk1')
        assert (trunk / 'latest.txt').read_text().strip() == '0'
        digests.add(initlib.init_digest(trunk / '0')['weights_sha256'])
    assert digests == {m['weights_sha256']}


def test_partial_init_refused(site):
    par = initlib.init_parent(site.init_root)
    par.mkdir(parents=True)
    fake_step(site.init_scratch / '0', site.dp)
    with pytest.raises(SystemExit, match='exists without init_manifest.json'):
        initlib.finalize_init(site.init_scratch, site.init_root, {})


def test_trunk1_continuation_restores_state(site):
    _make_init(site)
    run = S.seg_name('disagreement-aware', 'trunk1')
    K.prepare(site, run)
    trunk = S.ckpt_dir(site.ckpt_root, 'disagreement-aware', 'trunk1')
    fake_step(trunk / '20', site.dp, 'progress')
    (trunk / 'latest.txt').write_text('20\n')
    rc, eff = K.prepare(site, run)                                        # a requeue after real progress
    c = yaml.safe_load(open(eff))
    assert rc == 0 and eff.endswith('.resume.yaml')
    assert c['checkpoints']['load_optimizer'] and c['checkpoints']['load_lr_scheduler']
    prod = yaml.safe_load(site.cfg_path('disagreement-aware', 'trunk1').read_text())
    assert {k for k, v in K._flat(c) if dict(K._flat(prod)).get(k) != v} == {'checkpoints.load_optimizer',
                                                                          'checkpoints.load_lr_scheduler'}
    assert initlib.init_digest(trunk / '0')['weights_sha256'] != initlib.init_digest(trunk / '20')['weights_sha256']
    with pytest.raises(SystemExit, match='refusing to seed over a progressed trunk'):
        (trunk / 'latest.txt').unlink()
        initlib.seed_trunk(trunk, initlib.init_parent(site.init_root) / '0',
                           initlib.init_parent(site.init_root) / initlib.MANIFEST)


def test_trunk_and_branch_sources(site):
    s = 'quality-base'
    trunk = S.ckpt_dir(site.ckpt_root, s, 'trunk1')
    with pytest.raises(SystemExit, match='missing or before step 4292'):
        K.prepare(site, S.seg_name(s, 'trunk2'))
    fake_step(trunk / '4292', site.dp)
    (trunk / 'latest.txt').write_text('4292\n')
    assert K.prepare(site, S.seg_name(s, 'trunk2'))[0] == 0
    fake_step(trunk / '8583', site.dp)                                   # trunk2 finished: it is complete now
    (trunk / 'latest.txt').write_text('8583\n')
    assert K.prepare(site, S.seg_name(s, 'trunk2'))[0] == K.EXIT_COMPLETE and K.finish(site, S.seg_name(s, 'trunk2'))[0] == 0
    with pytest.raises(SystemExit, match='branch source step 17166 missing'):
        K.prepare(site, S.seg_name(s, 'ep2'))
    rc, eff = K.prepare(site, S.seg_name(s, 'ep1'))
    assert rc == 0 and yaml.safe_load(open(eff))['checkpoints']['resume_checkpoint_path'] == f'{trunk}/8583'
    ep1 = S.ckpt_dir(site.ckpt_root, s, 'ep1')
    fake_step(ep1 / '9537', site.dp)
    assert K.prepare(site, S.seg_name(s, 'ep1'))[0] == K.EXIT_COMPLETE
    assert K.finish(site, S.seg_name(s, 'ep1'))[0] == 0 and K.finish(site, S.seg_name(s, 'ep2'))[0] == 1


def test_guards(site, monkeypatch):
    s = 'diversity-oriented'
    trunk = S.ckpt_dir(site.ckpt_root, s, 'trunk1')
    fake_step(trunk / '4292', dp=site.dp * 2)                             # written at another data-parallel size
    (trunk / 'latest.txt').write_text('4292\n')
    with pytest.raises(SystemExit, match='written with data-parallel size'):
        K.prepare(site, S.seg_name(s, 'trunk2'))
    lock = site.ckpt_root / 'kys3b_deployment.json'
    lock.write_text(json.dumps({'deployment': {**site.deployment(), 'micro_batch_size': 99}}))
    with pytest.raises(SystemExit, match='keeps one deployment'):
        K.prepare(site, S.seg_name(s, 'trunk2'))
    lock.unlink()
    p = site.cfg_path(s, 'trunk2')
    p.write_text(p.read_text() + '\n# edited\n')
    with pytest.raises(SystemExit, match='differs from a fresh render'):
        K.prepare(site, S.seg_name(s, 'trunk2'))
    K.render(site, force=True)
    (site.tokenized / s / '00000_unshuffled.ds.metadata').write_text('/somewhere/else/tokenizer|2\n0\n')
    with pytest.raises(SystemExit, match='not relocated'):
        K.prepare(site, S.seg_name(s, 'trunk2'))
    fake_tokenized(site, [s])
    monkeypatch.delenv('WANDB_API_KEY')
    monkeypatch.setenv('NETRC', str(site.config_root / 'no-netrc'))
    with pytest.raises(SystemExit, match='wandb login'):
        K.prepare(site, S.seg_name(s, 'trunk2'))


def test_startup_helpers_read_the_supplied_config(site, tmp_path):
    src = site.cfg_path('quality-first', 'trunk1')
    work = tmp_path / 'startup'
    rec = startup.derive_startup(src, work)
    assert rec['production_config'] == str(src.resolve())
    prod = yaml.safe_load(src.read_text())
    for ph in startup.STARTUP_PHASES:
        c = yaml.safe_load((work / 'cfg' / f'{ph}.yaml').read_text())
        assert c['tokenizer'] == prod['tokenizer'] and c['data_stages'] == prod['data_stages']
        assert c['general']['project'] == PROJECT and c['parallelism'] == prod['parallelism']
        _no_skipjack(c)
    assert yaml.safe_load((work / 'cfg' / 's_trunk.yaml').read_text())['checkpoints']['checkpoints_path'] == \
        prod['checkpoints']['checkpoints_path']
    rep = startup.summarize(src, work)
    assert any(str(S.ckpt_dir(site.ckpt_root, 'quality-first', 'trunk1')) in p for p in rep['problems'])
