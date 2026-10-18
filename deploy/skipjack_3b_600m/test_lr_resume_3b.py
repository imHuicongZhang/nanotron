"""CPU regression test: a resumed 3B run applies exactly the LRs of one uninterrupted run (trunk, requeue, branches).

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_lr_resume_3b.py
    $P deploy/skipjack_3b_600m/test_lr_resume_3b.py --report <out.json>     # upstream vs this worktree, at boundaries

Mirrors the trainer per segment, using the real rendered 3B configs and nanotron's own save/load functions:
  build optimizer -> (resume) optimizer.load_state_dict -> lr_scheduler_builder -> (resume) load_lr_scheduler
  (trainer.py:208-238); per update: record the group lr, optimizer.step(), lr_scheduler.step() (trainer.py:726, 731);
  save optimizer + save_lr_scheduler every checkpoint_interval and at train_steps; the next trunk segment resumes from
  the last save, a branch from trunk/<B>. Trunk 2 is additionally interrupted after update 7000 and requeued.
Expected: update n uses peak * lambda(n-1) of its segment's config (a continuous scheduler), and every checkpoint <S>
stores last_epoch == S. lr_scheduler_builder is extracted from helpers.py (that module imports flash-attn, GPU-only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as tdist
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import registry_3b as R  # noqa: E402
from lr_schedule_report import _builder  # noqa: E402
from nanotron.serialize import optimizer as S  # noqa: E402

CFG = str(R.REPO / 'configs' / '3B-rewrite-600m' / f'seed{R.SEED}' / '3b_quality_first_seed42_{}.yaml')
REQUEUE = ('trunk2', 7000)  # trunk2 dies after update 7000 and restarts from checkpoint 7000


def upstream_load_lr_scheduler(lr_scheduler, is_zero, parallel_context, root_folder):
    """load_lr_scheduler as shipped at bc131392 (serialize/optimizer.py:360-370), for the report."""
    lr_scheduler.load_state_dict(torch.load(Path(root_folder) / 'lr_scheduler' / S.lr_scheduler_filename(parallel_context, is_zero)))
    lr_scheduler._initial_step()


def _pc():
    if not tdist.is_initialized():
        tdist.init_process_group('gloo', init_method='tcp://127.0.0.1:0', rank=0, world_size=1)
    w = tdist.group.WORLD
    return SimpleNamespace(pp_pg=w, dp_pg=w, tp_pg=w, ep_pg=w, dp_cp_pg=w, expert_parallel_size=1)


class _AdamW(torch.optim.AdamW):
    def get_base_optimizer(self):
        return self


def run_chain(tmp: Path, load=S.load_lr_scheduler):
    """Returns ({segment: (first update, lrs used)}, {(segment, step): saved last_epoch}, {segment: config})."""
    build, LRArgs = _builder()
    pc, used, saved, cfgs = _pc(), {}, {}, {}

    def segment(kind, src, first, stop=None):
        c = yaml.safe_load(open(CFG.format(kind)))
        a, steps, ck = LRArgs(**c['optimizer']['learning_rate_scheduler']), c['tokens']['train_steps'], c['checkpoints']
        z = c['optimizer']['zero_stage']
        p = torch.nn.Parameter(torch.zeros(4))
        opt = _AdamW([p], lr=a.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        if src is not None and ck['load_optimizer']:
            opt.load_state_dict(torch.load(src / 'optimizer.pt'))
        sch = build(opt, a, total_training_steps=steps)
        if src is not None and ck['load_lr_scheduler']:
            load(lr_scheduler=sch, is_zero=z, parallel_context=pc, root_folder=src)
        out = tmp / kind
        lrs = []
        for it in range(first, (stop or steps) + 1):
            lrs.append(opt.param_groups[0]['lr'])
            p.grad = torch.ones_like(p)
            opt.step()
            sch.step()
            if it % ck['checkpoint_interval'] == 0 or it == steps:
                d = out / str(it)
                d.mkdir(parents=True, exist_ok=True)
                torch.save(opt.state_dict(), d / 'optimizer.pt')
                S.save_lr_scheduler(sch, is_zero=z, parallel_context=pc, root_folder=d)
                saved[(kind, it)] = sch.last_epoch
        cfgs[kind] = c
        return first, np.array(lrs)

    prev, last = None, 0
    for i, end in enumerate(R.TRUNK_ENDS, 1):
        kind = f'trunk{i}'
        if kind == REQUEUE[0]:
            used[kind + '_killed'] = segment(kind, prev, last + 1, stop=REQUEUE[1])
            prev, last = tmp / kind / str(REQUEUE[1]), REQUEUE[1]
        used[kind] = segment(kind, prev, last + 1)
        prev, last = tmp / kind / str(end), end
    for k, b in enumerate(R.BRANCH_POINTS, 1):
        src = next(tmp / f'trunk{i}' / str(b) for i, e in enumerate(R.TRUNK_ENDS, 1) if e == b)
        used[f'ep{k}'] = segment(f'ep{k}', src, b + 1)
    return used, saved, cfgs


def expected(cfg, first, n):
    build, LRArgs = _builder()
    a = LRArgs(**cfg['optimizer']['learning_rate_scheduler'])
    lam = build(_AdamW([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate), a, cfg['tokens']['train_steps']).lr_lambdas[0]
    return np.array([a.learning_rate * lam(u - 1) for u in range(first, first + n)])


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    return run_chain(tmp_path_factory.mktemp('lr_resume'))


def test_every_update_matches_uninterrupted(chain):
    used, _, cfgs = chain
    for seg, (first, lrs) in used.items():
        want = expected(cfgs[seg.removesuffix('_killed')], first, len(lrs))
        bad = np.flatnonzero(np.abs(lrs - want) > 1e-15)
        assert bad.size == 0, f'{seg}: update {first + bad[0]} used {lrs[bad[0]]:.6e}, uninterrupted {want[bad[0]]:.6e}'


def test_saved_scheduler_index_equals_step(chain):
    _, saved, _ = chain
    assert {k: v for k, v in saved.items() if v != k[1]} == {}


def test_branch_boundaries(chain):
    used, _, _ = chain
    trunk = np.concatenate([v[1] for k, v in used.items() if k.startswith('trunk')])  # in order; index 0 = update 1
    assert len(trunk) == R.TRUNK_ENDS[-1] and np.all(trunk[R.WARMUP_STEPS:] == R.PEAK_LR)  # no trunk dip
    for k, (b, h) in enumerate(zip(R.BRANCH_POINTS, R.HORIZONS), 1):
        first, lrs = used[f'ep{k}']
        d = h - b
        assert first == b + 1 and len(lrs) == d
        assert lrs[0] == R.PEAK_LR and lrs[1] < R.PEAK_LR             # update B+1 at peak, decay from B+2
        assert lrs[-1] == pytest.approx(R.PEAK_LR / d, rel=1e-12)      # update H uses lambda(H-1) = peak/D, never 0
        assert np.all(lrs > 0)


def report(out: Path):
    import tempfile
    rows = {}
    for name, fn in (('upstream_bc131392', upstream_load_lr_scheduler), ('worktree', S.load_lr_scheduler)):
        with tempfile.TemporaryDirectory() as t:
            used, saved, cfgs = run_chain(Path(t), fn)
        lr_at = {}
        for seg, (first, lrs) in used.items():                     # trunk2_killed covers 4293-7000, trunk2 7001-8583
            for j, v in enumerate(lrs):
                lr_at.setdefault('trunk' if seg.startswith('trunk') else seg, {})[first + j] = float(v)
        probe = {'trunk': [999, 1000, 1001, 7000, 7001, 8583, 25745, 25746, 25749]}
        for k, (b, h) in enumerate(zip(R.BRANCH_POINTS, R.HORIZONS), 1):
            probe[f'ep{k}'] = [b + 1, b + 2, h - 6, h - 5, h - 4, h - 3, h - 2, h - 1, h]
        rows[name] = {'saved_last_epoch': {f'{k[0]}/{k[1]}': v for k, v in saved.items() if k[1] in R.TRUNK_ENDS or k[1] == REQUEUE[1]},
                      'lr_used_by_update': {s: {u: lr_at[s][u] for u in us} for s, us in probe.items()},
                      'updates_at_lr0': {s: sum(1 for v in lr_at[s].values() if v == 0) for s in lr_at}}
    for s, us in rows['worktree']['lr_used_by_update'].items():
        for u in us:
            print(f"{s:6s} update {u:6d}  upstream {rows['upstream_bc131392']['lr_used_by_update'][s][u]:.6e}  "
                  f"worktree {rows['worktree']['lr_used_by_update'][s][u]:.6e}")
    out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path, required=True)
    report(ap.parse_args().report)
