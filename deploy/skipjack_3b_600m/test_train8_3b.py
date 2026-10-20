"""Tests for train8_3b.py (dedicated 8-GPU budget) against the 1.5B tool's fake Slurm (afterok + kill_invalid_depend).

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_train8_3b.py

Every path the submitter writes (configs, checkpoints, state, logs, the startup marker) is redirected into tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
import generate_configs_3b as G  # noqa: E402
import registry_3b as R  # noqa: E402
import train8_3b as T  # noqa: E402
from test_capped_submit import FakeSlurm  # noqa: E402


class Fake8(FakeSlurm):
    """The startup job's last argv is its workdir, not a config: on success it writes the PASSED marker."""

    def tick(self):
        for j in self.jobs.values():
            if j['state'] == 'RUNNING' and j['name'] == R.TRAIN8_PREFIX + R.STARTUP and j['name'] not in self.fail:
                T.STARTUP_PASSED.mkdir(parents=True, exist_ok=True)
                (T.STARTUP_PASSED / 'model_config.json').write_text('{}')
                cfg = T.STARTUP_PASSED.parent / 'fake.yaml'
                cfg.write_text(yaml.safe_dump({'checkpoints': {'checkpoints_path': str(T.STARTUP_PASSED.parent / 'x')},
                                               'tokens': {'train_steps': 0}}))
                j['cfg'] = str(cfg)
        return super().tick()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'CKPT_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(R, 'LOG_DIR', tmp_path / 'logs')
    cfg = tmp_path / 'cfg'
    G.main(['--out', str(cfg), '--layout', 'A'])
    for k, v in {'CFG_DIR': cfg, 'STATE_DIR': tmp_path / 'state', 'STARTUP_WORK': tmp_path / 'startup' / 'run',
                 'STARTUP_PASSED': tmp_path / 'startup' / 'PASSED'}.items():
        monkeypatch.setattr(T, k, v)
    monkeypatch.setattr(T, 'preflight', lambda: None)
    return tmp_path


def run_to_end(slurm, max_ticks=300):
    for _ in range(max_ticks):
        slurm.tick()
        if all(j['state'] in ('COMPLETED', 'FAILED', 'CANCELLED') for j in slurm.jobs.values()):
            return
    raise AssertionError('did not finish')


def test_plan_shape(env):
    segs = T.load_segments()
    T.assign_breadth_first(segs, R.TRAIN8_LANES, 32.2)
    T.check_graph(segs, R.TRAIN8_LANES)
    assert len(segs) == 28 and R.TRAIN8_LANES == 2 and R.TRAIN8_GPU_CEILING == 8
    assert all(s.job_name.startswith('rw3b_gpu_') and not s.job_name.startswith('kys_') for s in segs.values())
    lane0 = sorted((s for s in segs.values() if s.lane == 0), key=lambda s: s.order)
    assert lane0[0].name == R.STARTUP and lane0[1].name == R.seg_name('quality-first', 'trunk1')
    for s in R.SETTINGS:   # every scientific edge of the registry is kept
        for k in R.KINDS:
            d = segs[R.seg_name(s, k)].science_dep
            assert d == (R.seg_name(s, R.DEPENDS[k]) if R.DEPENDS[k] else (R.STARTUP if s == 'quality-first' else None))


def test_full_grid_two_lanes_never_more_than_8_gpus(env):
    slurm = Fake8()
    slurm.jobs['1'] = {'name': 'kys_raw_random_seed42_trunk1', 'deps': [], 'state': 'PENDING', 'start': 'Unknown',
                       'cfg': '', 'gpus': 4}   # a live baseline job never blocks (or is counted by) the 3B grid
    T.main(['submit'], slurm=slurm)
    mine = {i: j for i, j in slurm.jobs.items() if j['name'].startswith('rw3b_gpu_')}
    assert len(mine) == 28
    del slurm.jobs['1']
    run_to_end(slurm)
    assert slurm.peak <= 8
    assert all(j['state'] == 'COMPLETED' for j in slurm.jobs.values())


def test_submit_is_idempotent(env):
    slurm = Fake8()
    T.main(['submit'], slurm=slurm)
    n = len(slurm.jobs)
    T.main(['submit'], slurm=slurm)
    assert len(slurm.jobs) == n == 28


def test_failed_startup_stops_quality_first_and_is_not_retried(env):
    slurm = Fake8(fail={R.TRAIN8_PREFIX + R.STARTUP})
    T.main(['submit'], slurm=slurm)
    run_to_end(slurm)
    by = {j['name']: j['state'] for j in slurm.jobs.values()}
    assert by[R.TRAIN8_PREFIX + R.STARTUP] == 'FAILED'
    assert by[R.TRAIN8_PREFIX + R.seg_name('quality-first', 'trunk1')] == 'CANCELLED'
    n = len(slurm.jobs)
    T.main(['submit'], slurm=slurm)          # no --retry: the failed startup is not resubmitted, nothing behind it either
    assert not any(j['name'] == R.TRAIN8_PREFIX + R.STARTUP and j['state'] == 'PENDING' for j in slurm.jobs.values())
    assert all(slurm.jobs[i]['name'] != R.TRAIN8_PREFIX + R.seg_name('quality-first', 'trunk1')
               for i in list(slurm.jobs)[n:])


def test_foreign_rw3b_gpu_job_refused(env):
    slurm = Fake8()
    slurm.jobs['1'] = {'name': 'rw3b_gpu_manual', 'deps': [], 'state': 'PENDING', 'start': 'Unknown', 'cfg': '', 'gpus': 4}
    with pytest.raises(SystemExit):
        T.main(['submit'], slurm=slurm)
