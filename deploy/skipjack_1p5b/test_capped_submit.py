"""Tests for capped_submit.py against a fake Slurm that enforces afterok and kill_invalid_depend.

    python -m pytest -q deploy/skipjack_1p5b/test_capped_submit.py

The fake starts every job whose afterok dependencies have COMPLETED, with no node limit, so any
concurrency bound observed comes from the dependency graph alone (the mechanism under test). It rejects
afterok on a job that is not live, as skipjack does once a finished job ages out (MinJobAge 30 s).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capped_submit as cs  # noqa: E402

SEED = 42
STEPS = {'trunk1': 4292, 'trunk2': 8583, 'trunk3': 12875, 'ep1': 4768, 'ep2': 9537, 'ep3': 14305}


class FakeSlurm:
    def __init__(self, fail: set[str] = ()):
        self.jobs: dict[str, dict] = {}
        self.next = 1000
        self.fail = set(fail)          # job names whose next run fails
        self.peak = 0
        self.crash_after_sbatch = None  # job name: create the job, then raise before returning its id

    # --- interface used by capped_submit -------------------------------------------------------------
    def sbatch(self, argv):
        opts = dict(a[2:].split('=', 1) for a in argv if a.startswith('--') and '=' in a)
        deps = opts.get('dependency', '')
        deps = deps.split(':')[1:] if deps else []
        for d in deps:
            if d not in self.jobs or self.jobs[d]['state'] not in cs.LIVE:
                raise RuntimeError(f'Job dependency problem: {d}')
        assert opts['gres'] == 'gpu:h100:4' and opts['nodes'] == '1'
        jid = str(self.next)
        self.next += 1
        self.jobs[jid] = {'name': opts['job-name'], 'deps': deps, 'state': 'PENDING', 'start': 'Unknown',
                          'cfg': argv[-1], 'gpus': 4}
        if self.crash_after_sbatch == opts['job-name']:
            self.crash_after_sbatch = None
            raise KeyboardInterrupt('simulated crash after sbatch')
        return jid

    def test_only(self, argv):
        return 0, 'ok'

    def queue(self):
        return [{'id': i, 'name': j['name'], 'state': j['state'], 'reason': 'None', 'gpus': j['gpus']}
                for i, j in self.jobs.items() if j['state'] in cs.LIVE]

    def accounting(self, ids):
        return {i: {'name': self.jobs[i]['name'], 'state': self.jobs[i]['state'], 'start': self.jobs[i]['start'],
                    'end': 'x'} for i in ids if i in self.jobs}

    def by_name(self, name, since):
        return [{'id': i, 'name': j['name'], 'state': j['state']} for i, j in self.jobs.items() if j['name'] == name]

    def cancel(self, ids):
        for i in ids:
            self.jobs[i]['state'] = 'CANCELLED'

    # --- simulation -----------------------------------------------------------------------------------
    def tick(self):
        """Finish every running job, then start every pending job whose deps completed (kill invalid ones)."""
        for j in self.jobs.values():
            if j['state'] == 'RUNNING':
                if j['name'] in self.fail:
                    self.fail.discard(j['name'])
                    j['state'] = 'FAILED'
                else:
                    c = yaml.safe_load(Path(j['cfg']).read_text())
                    d = Path(c['checkpoints']['checkpoints_path']) / str(c['tokens']['train_steps'])
                    d.mkdir(parents=True, exist_ok=True)
                    (d / 'model_config.json').write_text('{}')
                    j['state'] = 'COMPLETED'
        changed = True
        while changed:  # kill_invalid_depend cascades
            changed = False
            for j in self.jobs.values():
                if j['state'] == 'PENDING' and any(self.jobs[d]['state'] in ('FAILED', 'CANCELLED') for d in j['deps']):
                    j['state'] = 'CANCELLED'
                    changed = True
        for j in self.jobs.values():
            if j['state'] == 'PENDING' and all(self.jobs[d]['state'] == 'COMPLETED' for d in j['deps']):
                j['state'] = 'RUNNING'
                j['start'] = '2026-10-16T00:00:00'
        running = [j for j in self.jobs.values() if j['state'] == 'RUNNING']
        self.peak = max(self.peak, sum(j['gpus'] for j in running))
        assert all(j['gpus'] == 4 for j in running)
        return len(running)

    def requeue(self, name):
        for j in self.jobs.values():
            if j['name'] == name and j['state'] == 'RUNNING':
                j['state'] = 'PENDING'   # same id; deps already satisfied
                return
        raise AssertionError(f'{name} not running')


@pytest.fixture
def env(tmp_path):
    cfg_dir, ckpt, init = tmp_path / 'filled', tmp_path / 'runs', tmp_path / 'init' / f'_init_1.5B_seed{SEED}' / '0'
    cfg_dir.mkdir()
    for s in cs.SETTING_NAMES:
        for k in cs.KINDS:
            p = ckpt / f'seed{SEED}' / s / ('trunk' if k.startswith('trunk') else k) / s / f'seed{SEED}'
            (cfg_dir / f'{s}_seed{SEED}_{k}.yaml').write_text(yaml.safe_dump(
                {'checkpoints': {'checkpoints_path': str(p)}, 'tokens': {'train_steps': STEPS[k]}}))
    for f in ('model_config.json', 'optimizer/optimizer_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt',
              'lr_scheduler/lr_scheduler_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt'):
        (init / f).parent.mkdir(parents=True, exist_ok=True)
        (init / f).write_text('x')
    for i in range(177):
        (init / f'model/f{i}').parent.mkdir(parents=True, exist_ok=True)
        (init / f'model/f{i}').write_text('x')
    c = {'init_root': str(tmp_path / 'init'), 'repo_dir': None, 'dp': 4, 'tp': 1, 'pp': 1,
         'slurm': {'partition': 'h100', 'account': 'a', 'qos': 'q', 'nodes': 1, 'gres': 'gpu:h100:4',
                   'cpus_per_task': 32, 'time': '2-00:00:00'}}
    return {'cfg_dir': cfg_dir, 'c': c, 'state': tmp_path / 'state', 'log': tmp_path / 'logs', 'tmp': tmp_path}


def make(env, slurm, retry=(), plan=True):
    segs = cs.load_segments(env['cfg_dir'], SEED, cs.SETTING_NAMES)
    env['state'].mkdir(exist_ok=True)
    st = cs.State(env['state'] / 'state.json')
    if st.data['plan']:
        for n, s in segs.items():
            p = st.data['plan']['segments'][n]
            s.lane, s.order, s.lane_prev, s.est_start_h, s.est_end_h = (p['lane'], p['order'], p['lane_prev'],
                                                                        p['est_start_h'], p['est_end_h'])
    else:
        cs.assign_lanes(segs, 5, 22.8)
        st.data['plan'] = {'lanes': 5, 'segments': {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev,
                                                        'est_start_h': s.est_start_h, 'est_end_h': s.est_end_h}
                                                    for n, s in segs.items()}}
    cs.check_graph(segs, 5)
    status = cs.submit(segs, st, slurm, env['c'], SEED, env['log'], set(retry))
    st.save()
    return segs, st, status


def run_to_end(env, slurm, max_ticks=200):
    for _ in range(max_ticks):
        if not slurm.queue():
            return
        assert slurm.tick() <= 5
    raise AssertionError('did not drain')


# --- plan ------------------------------------------------------------------------------------------------

def test_plan_shape():
    segs = {}
    for s in cs.SETTING_NAMES:
        for k in cs.KINDS:
            n = f'{s}_seed{SEED}_{k}'
            segs[n] = cs.Segment(n, s, k, Path('x'), Path('y'), STEPS[k], STEPS[k] - cs.FIRST_STEP[k],
                                 science_dep=f'{s}_seed{SEED}_{cs.DEPENDS[k]}' if cs.DEPENDS[k] else None)
    makespan = cs.assign_lanes(segs, 5, 22.8)
    cs.check_graph(segs, 5)
    assert len(segs) == 42 and {s.lane for s in segs.values()} == set(range(5))
    assert cs.max_concurrency(segs) == 5
    chain_h = 15735 * 22.8 / 3600
    assert makespan < 2 * chain_h          # better than two whole chains in one lane
    assert makespan >= 7 * chain_h / 5      # cannot beat total work / 5
    for s in segs.values():                 # every science edge starts after its upstream ends
        if s.science_dep:
            assert s.est_start_h >= segs[s.science_dep].est_end_h - 1e-9


def test_lane_cap_rejected():
    with pytest.raises(SystemExit):
        cs.validate_profile({'dp': 4, 'tp': 1, 'pp': 1, 'slurm': {'partition': 'h100', 'gres': 'gpu:h100:4', 'nodes': 1,
                                                                  'cpus_per_task': 32, 'time': '2-00:00:00'}},
                            'x', None, 6)
    with pytest.raises(SystemExit):
        cs.validate_profile({'dp': 4, 'tp': 1, 'pp': 1, 'slurm': {'partition': 'h100', 'gres': 'gpu:h100:4', 'nodes': 1,
                                                                  'cpus_per_task': 32, 'time': '3-00:00:01'}},
                            'x', None, 5)
    with pytest.raises(SystemExit):
        cs.validate_profile({'dp': 8, 'tp': 1, 'pp': 1, 'slurm': {'partition': 'h100', 'gres': 'gpu:h100:8', 'nodes': 1,
                                                                  'cpus_per_task': 32, 'time': '2-00:00:00'}},
                            'x', None, 5)


# --- execution -------------------------------------------------------------------------------------------

def test_full_run_respects_ceiling(env):
    sl = FakeSlurm()
    make(env, sl)
    assert len(sl.jobs) == 42
    assert len([j for j in sl.jobs.values() if not j['deps']]) == 5   # five lane heads
    run_to_end(env, sl)
    assert sl.peak <= 20
    segs, st, status = make(env, sl)
    assert all(r['status'] == cs.DONE for r in status.values())
    assert len(sl.jobs) == 42                                          # nothing resubmitted


def test_trunk_seeded_once(env):
    sl = FakeSlurm()
    make(env, sl)
    t = env['tmp'] / 'runs' / f'seed{SEED}' / 'raw_random' / 'trunk' / 'raw_random' / f'seed{SEED}'
    assert (t / 'latest.txt').read_text().strip() == '0' and (t / '0' / 'model_config.json').is_file()


def test_duplicate_invocation_submits_nothing(env):
    sl = FakeSlurm()
    make(env, sl)
    sl.tick()
    make(env, sl)
    assert len(sl.jobs) == 42


def test_concurrent_submitter_refused(env):
    env['state'].mkdir()
    with cs.Lock(env['state'] / 'submit.lock'):
        with pytest.raises(SystemExit, match='refusing'):
            with cs.Lock(env['state'] / 'submit.lock'):
                pass


def test_partial_submission_recovery(env):
    sl = FakeSlurm()
    segs = cs.load_segments(env['cfg_dir'], SEED, cs.SETTING_NAMES)
    cs.assign_lanes(segs, 5, 22.8)
    victim = sorted(segs.values(), key=lambda s: (s.est_start_h, s.lane, s.order))[7]
    sl.crash_after_sbatch = victim.job_name
    with pytest.raises(KeyboardInterrupt):
        make(env, sl)
    assert len(sl.jobs) == 8
    st = json.loads((env['state'] / 'state.json').read_text())
    assert st['segments'][victim.name]['attempts'][-1]['job_id'] is None   # intent written, id lost
    make(env, sl)
    assert len(sl.jobs) == 42                                               # adopted, not duplicated
    names = [j['name'] for j in sl.jobs.values()]
    assert len(names) == len(set(names))
    run_to_end(env, sl)
    assert sl.peak <= 20


def test_interrupted_submission_without_job_is_resubmitted(env):
    sl = FakeSlurm()
    make(env, sl)
    st = cs.State(env['state'] / 'state.json')
    # simulate: intent recorded for a not-yet-submitted segment, crash before sbatch ran
    first = next(n for n in st.data['segments'])
    sl.jobs.clear()
    for n in st.data['segments']:
        st.data['segments'][n]['attempts'] = []
    st.data['segments'][first]['attempts'] = [{'job_id': None, 'intent_at': '2026-10-16T00:00:00'}]
    st.save()
    make(env, sl)
    assert len(sl.jobs) == 42


def test_requeue_keeps_lane(env):
    sl = FakeSlurm()
    make(env, sl)
    sl.tick()
    running = [j['name'] for j in sl.jobs.values() if j['state'] == 'RUNNING']
    sl.requeue(running[0])
    make(env, sl)
    assert len(sl.jobs) == 42
    run_to_end(env, sl)
    assert sl.peak <= 20


def test_failed_upstream_halts_then_retry(env):
    target = 'kys_raw_random_seed42_trunk1'
    sl = FakeSlurm(fail={target})
    make(env, sl)
    run_to_end(env, sl)
    segs, st, status = make(env, sl)
    assert status['raw_random_seed42_trunk1']['status'] == cs.FAILED
    assert all(r['status'] != cs.DONE for n, r in status.items() if n.startswith('raw_random_'))
    n_before = len(sl.jobs)
    assert n_before == 42                                         # halted: nothing resubmitted without --retry
    killed = [n for n, r in status.items() if r['status'] == cs.DEPKILLED]
    assert killed                                                 # dependents were cancelled by kill_invalid_depend
    with pytest.raises(SystemExit):
        make(env, sl, retry={'raw_random_seed42_ep1'})            # not FAILED -> refused
    make(env, sl, retry={'raw_random_seed42_trunk1'})
    run_to_end(env, sl)
    assert sl.peak <= 20
    segs, st, status = make(env, sl)
    assert all(r['status'] == cs.DONE for r in status.values())


def test_foreign_job_with_planned_name_refused(env):
    sl = FakeSlurm()
    sl.jobs['999'] = {'name': 'kys_raw_random_seed42_trunk1', 'deps': [], 'state': 'PENDING', 'start': 'Unknown',
                      'cfg': '', 'gpus': 4}
    with pytest.raises(SystemExit, match='did not submit'):
        make(env, sl)


def test_other_project_gpu_job_blocks_grid(env):
    sl = FakeSlurm()
    sl.jobs['998'] = {'name': 'kys_smoke20', 'deps': [], 'state': 'RUNNING', 'start': 'x', 'cfg': '', 'gpus': 4}
    with pytest.raises(SystemExit, match='other project GPU jobs'):
        make(env, sl)


def test_manual_cancel_is_not_auto_resubmitted(env):
    sl = FakeSlurm()
    make(env, sl)
    jid = next(i for i, j in sl.jobs.items() if j['name'] == 'kys_raw_random_seed42_trunk1')
    sl.cancel([jid])
    segs, st, status = make(env, sl)
    assert status['raw_random_seed42_trunk1']['status'] == cs.FAILED
    assert len(sl.jobs) == 42


def test_gpu_parsing():
    assert cs._gpus('cpu=32,mem=1G,node=1,gres/gpu=4,gres/gpu:h100=4') == 4
    assert cs._gpus('gres/gpu:h100:4') == 4 and cs._gpus('gres/gpu:4') == 4 and cs._gpus('N/A') == 0


# --- parity with plan_submit.py (needs the real filled configs) -----------------------------------------

REPO = Path(__file__).resolve().parents[2]
FILLED = REPO / 'configs' / f'1.5B-baseline-seed{SEED}' / 'filled'


@pytest.mark.skipif(not (FILLED / f'raw_random_seed{SEED}_trunk1.env').is_file(), reason='filled configs not rendered')
def test_sbatch_options_match_plan_submit(tmp_path):
    prof = yaml.safe_load((REPO / 'deploy/clusters.yaml').read_text())
    c = prof['clusters'][cs.DEFAULT_CLUSTER]
    out = FILLED / f'submit_seed{SEED}.sh'
    backup = out.read_bytes() if out.exists() else None
    subprocess.run([sys.executable, REPO / 'tools/kys_raw/plan_submit.py', '--cluster', cs.DEFAULT_CLUSTER, '--seed',
                    str(SEED), '--settings', 'raw_random'], check=True, capture_output=True)
    script = out.read_text()
    if backup is None:
        out.unlink()
    else:
        out.write_bytes(backup)
    segs = cs.load_segments(FILLED, SEED, ['raw_random'])
    for s in segs.values():
        line = next(ln for ln in script.splitlines() if f'--job-name=kys_{s.name} ' in ln or f"--job-name=kys_{s.name}'" in ln)
        theirs = {o.strip("'") for o in re.findall(r"(--[a-z-]+=\S+)", line) if not o.startswith('--dependency')
                  and not o.startswith('--output')}
        ours = {o for o in cs.sbatch_opts(c, s, tmp_path) if not o.startswith('--output')}
        assert ours == theirs, (s.name, ours ^ theirs)
