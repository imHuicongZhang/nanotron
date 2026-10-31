"""LR continuity of PORTABLE-rendered configs across resumes, requeues and branches, all four settings (CPU only).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_portable_3b_lr.py

Mirrors the trainer per segment with nanotron's own code from this checkout (lr_scheduler_builder via lrsched,
save_lr_scheduler / load_lr_scheduler from nanotron.serialize.optimizer, i.e. the corrected LR resume):
  build optimizer -> (resume) optimizer.load_state_dict -> lr_scheduler_builder -> (resume) load_lr_scheduler;
  per update: record the group lr, optimizer.step(), lr_scheduler.step(); save every checkpoint_interval and at
  train_steps. Configs: kys3b render of an operator deployment unlike Skipjack's (dp 16 x mbs 4 x accum 16), the
  effective configs prepare would run: trunk1 stops after update 20 and continues with startup.resume_config (the
  startup validation / a requeue past step 0), trunk2 is killed after update 7000 and requeued, trunk3..6 resume
  latest, ep1..3 resume trunk/8583, /17166, /25749, and ep2 is restarted once from its source.
Expected: update n uses peak * lambda(n - 1) of its segment's config (one continuous scheduler), every checkpoint <S>
stores last_epoch == S, the trunk never decays, branch update B+1 is at peak and update H uses peak / D (never 0).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as tdist
import yaml

from conftest import CODE, operator_dict, write
import kys3b as K
import lrsched
import science_3b as S
import startup

sys.path.insert(0, str(CODE / 'src'))
from nanotron.serialize import optimizer as SER  # noqa: E402


def _pc():
    if not tdist.is_initialized():
        tdist.init_process_group('gloo', init_method='tcp://127.0.0.1:0', rank=0, world_size=1)
    w = tdist.group.WORLD
    return SimpleNamespace(pp_pg=w, dp_pg=w, tp_pg=w, ep_pg=w, dp_cp_pg=w, expert_parallel_size=1)


class _AdamW(torch.optim.AdamW):
    def get_base_optimizer(self):
        return self


def run_chain(site, setting: str, tmp: Path):
    build, LRArgs = lrsched.builder()
    pc, used, saved, cfgs = _pc(), {}, {}, {}

    def segment(label, c, src, first, stop=None):
        a, steps, ck = LRArgs(**c['optimizer']['learning_rate_scheduler']), c['tokens']['train_steps'], c['checkpoints']
        z = c['optimizer']['zero_stage']
        p = torch.nn.Parameter(torch.zeros(4))
        opt = _AdamW([p], lr=a.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        if src is not None and ck['load_optimizer']:
            opt.load_state_dict(torch.load(src / 'optimizer.pt'))
        sch = build(opt, a, total_training_steps=steps)
        if src is not None and ck['load_lr_scheduler']:
            SER.load_lr_scheduler(lr_scheduler=sch, is_zero=z, parallel_context=pc, root_folder=src)
        out = tmp / label
        lrs = []
        for it in range(first, (stop or steps) + 1):
            lrs.append(opt.param_groups[0]['lr'])
            p.grad = torch.ones_like(p)
            opt.step()
            sch.step()
            if it % ck['checkpoint_interval'] == 0 or it == steps or it == stop:
                d = out / str(it)
                d.mkdir(parents=True, exist_ok=True)
                torch.save(opt.state_dict(), d / 'optimizer.pt')
                SER.save_lr_scheduler(sch, is_zero=z, parallel_context=pc, root_folder=d)
                saved[(label, it)] = sch.last_epoch
        cfgs[label] = c
        used[label] = (first, np.array(lrs))
        return out

    cfg = {k: yaml.safe_load(site.cfg_path(setting, k).read_text()) for k in S.KINDS}
    a, _ = S.STARTUP_STEPS
    resume_p = tmp / 'trunk1.resume.yaml'
    startup.resume_config(site.cfg_path(setting, 'trunk1'), resume_p)
    segment('trunk1_startup', cfg['trunk1'], None, 1, stop=a)                       # weights-only from the init
    prev = segment('trunk1', yaml.safe_load(resume_p.read_text()), tmp / 'trunk1_startup' / str(a), a + 1)
    last = S.TRUNK_ENDS[0]
    trunk_dirs = {S.TRUNK_ENDS[0]: prev}
    for i, end in enumerate(S.TRUNK_ENDS[1:], 2):
        k = f'trunk{i}'
        if k == 'trunk2':
            killed = segment('trunk2_killed', cfg[k], prev / str(last), last + 1, stop=7000)
            prev, last = killed, 7000
        prev = segment(k, cfg[k], prev / str(last), last + 1)
        last = end
        trunk_dirs[end] = prev
    for k, b in zip(S.BRANCH_KINDS, S.BRANCH_POINTS):
        if k == 'ep2':
            segment('ep2_killed', cfg[k], trunk_dirs[b] / str(b), b + 1, stop=b + 500)   # restart re-runs from B
        segment(k, cfg[k], trunk_dirs[b] / str(b), b + 1)
    return used, saved, cfgs


def expected(cfg, first, n):
    peak, lam = lrsched.lr_lambda(cfg)
    return np.array([peak * lam(u - 1) for u in range(first, first + n)])


@pytest.fixture(scope='module')
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp('lr_site')
    s = K.Site(K.load_operator(write(operator_dict(root), root / 'operator.yaml')))
    K.render(s)
    return s


@pytest.fixture(scope='module', params=S.GRID_SETTINGS)
def chain(request, site, tmp_path_factory):
    return request.param, run_chain(site, request.param, tmp_path_factory.mktemp(f'lr_{request.param}'))


def test_every_update_matches_uninterrupted(chain):
    setting, (used, _, cfgs) = chain
    for seg, (first, lrs) in used.items():
        want = expected(cfgs[seg], first, len(lrs))
        bad = np.flatnonzero(np.abs(lrs - want) > 1e-15)
        assert bad.size == 0, f'{setting} {seg}: update {first + bad[0]} used {lrs[bad[0]]:.6e}, uninterrupted {want[bad[0]]:.6e}'


def test_saved_scheduler_index_equals_step(chain):
    _, (_, saved, _) = chain
    assert {k: v for k, v in saved.items() if v != k[1]} == {}


def test_trunk_never_decays_and_branch_boundaries(chain):
    setting, (used, _, _) = chain
    trunk = np.concatenate([used[k][1] for k in ('trunk1_startup', 'trunk1', 'trunk2_killed', 'trunk2', 'trunk3',
                                                 'trunk4', 'trunk5', 'trunk6')])
    assert len(trunk) == S.TRUNK_ENDS[-1] and np.all(trunk[S.WARMUP_STEPS:] == S.PEAK_LR)
    assert trunk[0] == 0.0 and trunk[S.WARMUP_STEPS - 1] < S.PEAK_LR
    for k, b, h in zip(S.BRANCH_KINDS, S.BRANCH_POINTS, S.HORIZONS):
        first, lrs = used[k]
        d = h - b
        assert first == b + 1 and len(lrs) == d
        assert lrs[0] == S.PEAK_LR and lrs[1] < S.PEAK_LR
        assert lrs[-1] == pytest.approx(S.PEAK_LR / d, rel=1e-12) and np.all(lrs > 0)
    assert np.array_equal(used['ep2_killed'][1], used['ep2'][1][:500])          # a branch restart repeats exactly


def test_detects_the_unfixed_resume(site, tmp_path, monkeypatch):
    """Negative control: with load_lr_scheduler as shipped before the fix (an extra _initial_step on every resume), the
    same chain must break continuity, so the checks above can fail."""
    def upstream(lr_scheduler, is_zero, parallel_context, root_folder):
        lr_scheduler.load_state_dict(torch.load(Path(root_folder) / 'lr_scheduler' /
                                                SER.lr_scheduler_filename(parallel_context, is_zero)))
        lr_scheduler._initial_step()
    monkeypatch.setattr(SER, 'load_lr_scheduler', upstream)
    used, saved, cfgs = run_chain(site, 'quality-base', tmp_path)
    off = [seg for seg, (first, lrs) in used.items() if np.any(np.abs(lrs - expected(cfgs[seg], first, len(lrs))) > 1e-15)]
    assert off and 'ep3' in off and any(v != k[1] for k, v in saved.items())
