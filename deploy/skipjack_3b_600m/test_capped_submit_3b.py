"""Tests for capped_submit_3b.py against the 1.5B tool's fake Slurm (afterok + kill_invalid_depend, no node limit).

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_capped_submit_3b.py

Every path the submitter writes (configs, checkpoints, init, state, logs, markers) is redirected into tmp_path, so the
tests never touch the real 3B namespaces, and the handoff gate is replaced by a switch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
import capped_submit_3b as C  # noqa: E402
import generate_configs_3b as G  # noqa: E402
import registry_3b as R  # noqa: E402
from test_capped_submit import FakeSlurm  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'CKPT_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(R, 'LOG_DIR', tmp_path / 'logs')
    monkeypatch.setattr(R, 'STATE_ROOT', tmp_path / 'state')
    cfg = tmp_path / 'cfg'
    G.main(['--out', str(cfg), '--layout', 'A'])
    init = tmp_path / 'init' / '_init_3B_seed42' / '0'
    for f in ('model_config.json', 'model/w0.safetensors', 'model/w1.safetensors', 'optimizer/o.pt', 'lr_scheduler/l.pt'):
        (init / f).parent.mkdir(parents=True, exist_ok=True)
        (init / f).write_text(f'weights {f}')
    manifest = init.parent / 'init_manifest.json'
    manifest.write_text(json.dumps(C.init_digest(init)))
    smoke = tmp_path / 'SMOKE_ACCEPTED'
    smoke.write_text(json.dumps({'layout': 'A', 's_per_it': 31.0}))
    monkeypatch.setattr(C, 'RETIRED', False)   # the retired tool's logic is still tested
    for k, v in {'CFG_DIR': cfg, 'STATE_DIR': tmp_path / 'state' / 'seed42', 'INIT': init, 'INIT_MANIFEST': manifest,
                 'SMOKE_ACCEPTED': smoke}.items():
        monkeypatch.setattr(C, k, v)
    gate = {'handoff': True, 'missing': []}
    gate['scopes'] = []
    monkeypatch.setattr(C.handoff_gate, 'check', lambda require_approval, scope='launch': gate['scopes'].append(scope) or dict(gate))
    real_run = subprocess.run

    def run(argv, *a, **k):   # the corpus check needs tokenized data; stub only that call
        if any(str(x).endswith('assert_invariants_3b.py') for x in argv):
            return subprocess.CompletedProcess(argv, 0, 'OK', '')
        return real_run(argv, *a, **k)
    monkeypatch.setattr(C.subprocess, 'run', run)
    return {'gate': gate, 'tmp': tmp_path}


def run_to_end(slurm, max_ticks=200):
    for _ in range(max_ticks):
        slurm.tick()
        if all(j['state'] in ('COMPLETED', 'FAILED', 'CANCELLED') for j in slurm.jobs.values()):
            return
    raise AssertionError('did not finish')


def test_schedule_constants():
    assert R.HORIZONS == [9537, 19073, 28610] and R.BRANCH_POINTS == [8583, 17166, 25749]
    assert [h * R.TOKENS_PER_STEP for h in R.HORIZONS] == [20_000_538_624, 39_998_980_096, 59_999_518_720]
    assert R.DEPENDS == {'trunk1': None, 'trunk2': 'trunk1', 'trunk3': 'trunk2', 'trunk4': 'trunk3', 'trunk5': 'trunk4',
                         'trunk6': 'trunk5', 'ep1': 'trunk2', 'ep2': 'trunk4', 'ep3': 'trunk6'}


def test_full_grid_respects_ceiling_and_seeds_identical_init(env):
    slurm = FakeSlurm()
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    assert len(slurm.jobs) == 27 and all(j['name'].startswith('kys_3b_') for j in slurm.jobs.values())
    run_to_end(slurm)
    assert slurm.peak <= 20
    assert all(j['state'] == 'COMPLETED' for j in slurm.jobs.values())
    digests = {C.init_digest(R.ckpt_dir(s, 'trunk1') / '0')['weights_sha256'] for s in R.SETTINGS}
    assert digests == {json.loads(C.INIT_MANIFEST.read_text())['weights_sha256']}


def test_branch_waits_for_its_trunk_segment(env):
    slurm = FakeSlurm()
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    by_name = {j['name']: (i, j) for i, j in slurm.jobs.items()}
    for s in R.SETTINGS:
        for b, t in (('ep1', 'trunk2'), ('ep2', 'trunk4'), ('ep3', 'trunk6')):
            assert by_name[f'kys_{R.seg_name(s, t)}'][0] in by_name[f'kys_{R.seg_name(s, b)}'][1]['deps']


def test_held_gate_submits_nothing(env):
    env['gate'].update(handoff=False, missing=['seed 44: no SEED_COMPLETE'])
    slurm = FakeSlurm()
    with pytest.raises(SystemExit, match='HELD'):
        C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    with pytest.raises(SystemExit, match='HELD'):
        C.main(['smoke'], slurm=slurm)
    with pytest.raises(SystemExit, match='HELD'):
        C.main(['guard', '--need-gpus', '1'], slurm=slurm)
    assert not slurm.jobs


def test_live_baseline_job_blocks_grid(env):
    slurm = FakeSlurm()
    slurm.jobs['1'] = {'name': 'kys_raw_random_seed44_ep3', 'deps': [], 'state': 'RUNNING', 'start': 'x', 'cfg': '', 'gpus': 4}
    with pytest.raises(SystemExit, match='other project GPU jobs are live'):
        C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    assert list(slurm.jobs) == ['1']


def test_guard_counts_all_project_gpus(env):
    slurm = FakeSlurm()
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    slurm.tick()   # 5 lane heads hold 20 GPUs
    with pytest.raises(SystemExit, match='cap'):
        C.main(['guard', '--need-gpus', '1'], slurm=slurm)


def test_duplicate_invocation_submits_nothing(env):
    slurm = FakeSlurm()
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    n = len(slurm.jobs)
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    assert len(slurm.jobs) == n


def test_smoke_acceptance_must_match_layout(env):
    C.SMOKE_ACCEPTED.write_text(json.dumps({'layout': 'B', 's_per_it': 31.0}))
    with pytest.raises(SystemExit, match='layout'):
        C.main(['submit', '--s-per-it', '31'], slurm=FakeSlurm())


def test_failed_trunk_halts_dependents_then_retry(env):
    name = f'kys_{R.seg_name("diversity-oriented", "trunk3")}'
    slurm = FakeSlurm(fail={name})
    C.main(['submit', '--s-per-it', '31'], slurm=slurm)
    run_to_end(slurm)
    assert not (R.ckpt_dir('diversity-oriented', 'ep3') / str(R.HORIZONS[2])).exists()
    C.main(['submit', '--s-per-it', '31', '--retry', R.seg_name('diversity-oriented', 'trunk3')], slurm=slurm)
    run_to_end(slurm)
    assert slurm.peak <= 20
    assert (R.ckpt_dir('diversity-oriented', 'ep3') / str(R.HORIZONS[2]) / 'model_config.json').is_file()


def test_smoke_needs_tokenization_and_lr_test(env, monkeypatch):
    monkeypatch.setattr(R, 'REPORTS', env['tmp'] / 'reports')
    slurm = FakeSlurm()
    with pytest.raises(SystemExit, match='tokenization not finished'):
        C.main(['smoke'], slurm=slurm)
    assert env['gate']['scopes'] == ['smoke'] and not slurm.jobs
    for s in R.SETTINGS:
        (R.REPORTS / 'tokenize' / s).mkdir(parents=True)
        (R.REPORTS / 'tokenize' / s / 'epoch_stats.json').write_text('{}')
    real = C.subprocess.run
    monkeypatch.setattr(C.subprocess, 'run', lambda argv, *a, **k: subprocess.CompletedProcess(argv, 1, 'F lr', '')
                        if 'test_lr_resume_3b.py' in str(argv) else real(argv, *a, **k))
    with pytest.raises(SystemExit, match='LR resume regression test failed'):
        C.main(['smoke'], slurm=slurm)
    monkeypatch.setattr(C.subprocess, 'run', lambda argv, *a, **k: subprocess.CompletedProcess(argv, 0, 'ok', '')
                        if 'test_lr_resume_3b.py' in str(argv) else real(argv, *a, **k))
    C.main(['smoke'], slurm=slurm)
    assert [j['name'] for j in slurm.jobs.values()] == ['kys_3b_smoke']


def test_smoke_authorization_never_opens_the_grid(tmp_path, monkeypatch):
    H = C.handoff_gate
    monkeypatch.setattr(H, 'APPROVAL', tmp_path / 'LAUNCH_APPROVED')
    monkeypatch.setattr(H, 'SMOKE_AUTHORIZED', tmp_path / 'SMOKE_AUTHORIZED')
    monkeypatch.setattr(H, 'BASELINE_STATE_ROOT', tmp_path / 'none')
    monkeypatch.setattr(H, 'live_jobs', lambda: [])
    (tmp_path / 'SMOKE_AUTHORIZED').write_text('{}')
    smoke, launch = H.check(True, 'smoke'), H.check(True, 'launch')
    assert not any('approval marker' in m for m in smoke['missing'])
    assert any('approval marker' in m for m in launch['missing'])
    assert not smoke['handoff'] and not launch['handoff']   # no SEED_COMPLETE here: held either way
