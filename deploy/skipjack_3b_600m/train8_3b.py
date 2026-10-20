#!/usr/bin/env python
"""Dedicated 8-GPU submitter for the 3B rewrite-600m grid (owner authorization 2026-10-20): the only way to put 3B GPU
work on the queue. Supersedes capped_submit_3b.py (shared 20-GPU budget after the baseline handoff), which is retired.

    P=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python
    $P deploy/skipjack_3b_600m/train8_3b.py plan      [--s-per-it 32.2]   # lane table, estimated finals (pure)
    $P deploy/skipjack_3b_600m/train8_3b.py test-only                      # sbatch --test-only x 28 (queues nothing)
    $P deploy/skipjack_3b_600m/train8_3b.py submit    [--retry <segment>]  # idempotent reconcile + submit
    $P deploy/skipjack_3b_600m/train8_3b.py status

Budget: 28 jobs (bounded startup validation + 27 segments) in TWO serial `afterok` lanes x 4 H100 (one node, DP=4,
TP=1, PP=1), so Slurm's dependency engine keeps at most 8 GPUs allocated, with no live orchestrator needed. Jobs are
named rw3b_gpu_*: not kys_, so the 1.5B baselines' tool (which refuses to submit while another kys_ job is live) and
its separate 20-GPU budget are untouched. The QoS limits (jhu2: 32 GPUs per user and per account) still apply.

Order: breadth-first by horizon tier, so both lanes stay busy and the three settings advance together:
  tier 0: trunk1-2 of all settings, then ep1 (20B finals); tier 1: trunk3-4, ep2 (40B); tier 2: trunk5-6, ep3 (60B).
Lane 0 starts with the startup validation (quality-first steps 0 -> 30, real progress), followed by quality-first
trunk1 (resumes trunk/30); lane 1 starts with diversity-oriented trunk1 at once. Every scientific edge is kept
(trunk_k -> trunk_k+1, trunk2/4/6 -> ep1/2/3) and attached as afterok in addition to the lane edge.

Restart/duplicate safety, DONE/FAILED/DEP_KILLED handling, intent records and adoption are deploy/skipjack_1p5b/
capped_submit.py's reconcile (imported unchanged from this worktree). A FAILED segment halts its lane until it is
inspected and named in --retry. The startup job is DONE only when its report passed (marker under reports/train8).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
import capped_submit as cs  # noqa: E402  (the 1.5B tool's reconcile/state/lock, unchanged)
import registry_3b as R  # noqa: E402

CFG_DIR = R.REPO / 'configs' / '3B-rewrite-600m' / f'seed{R.SEED}'
SEGMENT_SBATCH = HERE / 'segment_3b.sbatch'
STARTUP_SBATCH = HERE / 'startup_3b.sbatch'
STATE_DIR = R.TRAIN8_STATE
STARTUP_DIR = R.TRAIN8_DIR / 'startup'
STARTUP_WORK = STARTUP_DIR / 'run'
STARTUP_PASSED = STARTUP_DIR / 'PASSED'        # PASSED/model_config.json written only by a passing startup job
SEGMENT_TIME = R.SITE['slurm']['time']          # 2-00:00:00
STARTUP_TIME = '04:00:00'
DESIGN_S_PER_IT = 32.2                          # DESIGN.md estimate; replaced by the measured value in reports
TIER = {'startup': -1, 'trunk1': 0, 'trunk2': 0, 'ep1': 0, 'trunk3': 1, 'trunk4': 1, 'ep2': 1,
        'trunk5': 2, 'trunk6': 2, 'ep3': 2}

# the reused module counts and names only this grid's GPU jobs
cs.PROJECT_PREFIX = R.TRAIN8_PREFIX
cs.GPU_CEILING = R.TRAIN8_GPU_CEILING
assert cs.GPUS_PER_SEGMENT == R.GPUS_PER_SEGMENT and Path(cs.REPO) == R.REPO
assert not R.TRAIN8_PREFIX.startswith('kys_')


@dataclass
class Seg8(cs.Segment):
    script: Path = SEGMENT_SBATCH
    time: str = SEGMENT_TIME
    args: list = field(default_factory=list)

    @property
    def job_name(self) -> str:
        return R.TRAIN8_PREFIX + self.name

    @property
    def final(self) -> Path:
        return STARTUP_PASSED if self.kind == 'startup' else self.ckpt / str(self.final_step)


def load_segments() -> dict[str, Seg8]:
    segs = {}
    qf1 = R.seg_name('quality-first', 'trunk1')
    segs[R.STARTUP] = Seg8(name=R.STARTUP, setting='quality-first', kind='startup', cfg=CFG_DIR / f'{qf1}.yaml',
                           ckpt=R.ckpt_dir('quality-first', 'trunk1'), final_step=R.STARTUP_STEPS[-1],
                           steps=R.STARTUP_STEPS[-1] * 2 + 30,      # 30 + reference 30 steps + ~4 launches
                           script=STARTUP_SBATCH, time=STARTUP_TIME, args=[str(STARTUP_WORK)])
    for s in R.SETTINGS:
        for k in R.KINDS:
            name = R.seg_name(s, k)
            cfg = CFG_DIR / f'{name}.yaml'
            c = yaml.safe_load(cfg.read_text())
            assert c['tokens']['train_steps'] == R.END_STEP[k] and c['parallelism']['dp'] * c['parallelism']['tp'] \
                * c['parallelism']['pp'] == R.GPUS_PER_SEGMENT
            dep = R.seg_name(s, R.DEPENDS[k]) if R.DEPENDS[k] else (R.STARTUP if name == qf1 else None)
            segs[name] = Seg8(name=name, setting=s, kind=k, cfg=cfg, ckpt=Path(c['checkpoints']['checkpoints_path']),
                              final_step=R.END_STEP[k], steps=R.END_STEP[k] - R.FIRST_STEP[k], science_dep=dep,
                              args=[str(cfg)])
    return segs


def assign_breadth_first(segs: dict[str, Seg8], lanes: int, s_per_it: float) -> float:
    """Greedy list scheduling: a free lane takes the ready segment with the lowest (tier, first step, branch-after-trunk,
    setting): the least-advanced trunk of the current tier goes first, so the three settings stay level. Of the orders
    simulated at 32.2 s/it (tier only; first step only; these), this one gives the shortest makespan (438 h vs 452) and
    the earliest last 20B final (135 h vs 165) without delaying the 40B ones."""
    dur = {n: s.steps * s_per_it / 3600 + cs.OVERHEAD_H for n, s in segs.items()}
    first = {**R.FIRST_STEP, 'startup': -1}
    key = {n: (TIER[s.kind], first[s.kind], s.kind.startswith('ep'), R.SETTINGS.index(s.setting)) for n, s in segs.items()}
    free, last, length = [0.0] * lanes, [None] * lanes, [0] * lanes
    end: dict[str, float] = {}
    todo = set(segs)
    while todo:
        li = min(range(lanes), key=lambda i: (free[i], i))
        t = free[li]
        ready = [n for n in todo if segs[n].science_dep is None or end.get(segs[n].science_dep, 1e18) <= t]
        if not ready:
            free[li] = min(end[segs[n].science_dep] for n in todo if segs[n].science_dep in end)
            continue
        n = min(ready, key=lambda x: key[x])
        s = segs[n]
        s.lane, s.order, s.lane_prev, s.est_start_h, s.est_end_h = li, length[li], last[li], t, t + dur[n]
        end[n] = s.est_end_h
        free[li], last[li], length[li] = s.est_end_h, n, length[li] + 1
        todo.remove(n)
    return max(end.values())


def check_graph(segs: dict[str, Seg8], lanes: int):
    """Every segment in one lane; lanes are chains; edges acyclic; scientific edges exactly the registry's."""
    errs = []
    by_lane: dict[int, list[Seg8]] = {}
    for s in segs.values():
        by_lane.setdefault(s.lane, []).append(s)
    if set(by_lane) - set(range(lanes)):
        errs.append(f'lanes {sorted(by_lane)} outside 0..{lanes - 1}')
    for li, ss in by_lane.items():
        ss.sort(key=lambda s: s.order)
        if [s.order for s in ss] != list(range(len(ss))) or (ss and ss[0].lane_prev):
            errs.append(f'lane {li}: bad order/head')
        errs += [f'lane {li}: {b.name}.lane_prev={b.lane_prev}' for a, b in zip(ss, ss[1:]) if b.lane_prev != a.name]
    qf1 = R.seg_name('quality-first', 'trunk1')
    for s in segs.values():
        if s.kind == 'startup':
            want = None
        else:
            want = R.seg_name(s.setting, R.DEPENDS[s.kind]) if R.DEPENDS[s.kind] else (R.STARTUP if s.name == qf1 else None)
        if s.science_dep != want:
            errs.append(f'{s.name}: science_dep {s.science_dep}, expected {want}')
    seen: dict[str, int] = {}

    def visit(n):
        if seen.get(n) == 1:
            errs.append(f'cycle through {n}')
        if seen.get(n):
            return
        seen[n] = 1
        for d in (segs[n].lane_prev, segs[n].science_dep):
            if d:
                visit(d)
        seen[n] = 2
    for n in segs:
        visit(n)
    if cs.max_concurrency(segs) > lanes:
        errs.append('simulated concurrency exceeds the lane count')
    if errs:
        raise SystemExit('plan invariant violated:\n  ' + '\n  '.join(errs))


def site_for(s: Seg8) -> dict:
    return {**R.SITE, 'slurm': {**R.SITE['slurm'], 'time': s.time}}


def validate_time(s_per_it: float):
    have = cs.slurm_seconds(SEGMENT_TIME)
    need = max(R.END_STEP[k] - R.FIRST_STEP[k] for k in R.KINDS) * s_per_it
    if have < 1.1 * need + 1800:
        sys.exit(f'time {SEGMENT_TIME} too short at {s_per_it} s/it (longest segment needs {(1.1 * need + 1800) / 3600:.1f} h)')


def finals_table(segs: dict[str, Seg8], t0: float = 0.0) -> dict:
    return {s: {f'{20 * (i + 1)}B': round(t0 + segs[R.seg_name(s, f'ep{i + 1}')].est_end_h, 1) for i in range(3)}
            for s in R.SETTINGS}


def print_plan(segs, lanes, makespan):
    print(f'{"lane":>4s} {"#":>2s} {"segment":44s} {"steps":>6s} {"start h":>8s} {"end h":>8s}  after (lane / science)')
    for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
        print(f'{s.lane:4d} {s.order:2d} {s.name:44s} {s.steps:6d} {s.est_start_h:8.1f} {s.est_end_h:8.1f}  '
              f'{s.lane_prev or "-"} / {s.science_dep or "-"}')
    print(f'\n{len(segs)} jobs in {lanes} lanes; run-time makespan {makespan:.1f} h ({makespan / 24:.1f} d), no queue wait; '
          f'simulated peak {cs.max_concurrency(segs)} jobs = {cs.max_concurrency(segs) * R.GPUS_PER_SEGMENT} GPUs '
          f'(ceiling {R.TRAIN8_GPU_CEILING})')
    print('finals (h after the first start):', json.dumps(finals_table(segs)))


def submit(segs, st, slurm, retry: set[str], dry: bool = False) -> dict:
    queue = slurm.queue()
    mine = {s.job_name for s in segs.values()}
    others = [j for j in queue if j['name'].startswith(R.TRAIN8_PREFIX) and j['name'] not in mine]
    if others:
        sys.exit('rw3b_gpu_ jobs outside this plan are live (they would share the 8-GPU budget):\n  '
                 + '\n  '.join(f'{j["id"]} {j["name"]} {j["state"]}' for j in others))
    status = cs.reconcile(segs, st, slurm, queue)
    if not dry:
        st.save()
    for r in retry:
        if r not in segs:
            sys.exit(f'--retry {r}: not a segment of this plan')
        if status[r]['status'] not in (cs.FAILED, cs.UNKNOWN):
            sys.exit(f'--retry {r}: status is {status[r]["status"]}, only FAILED/UNKNOWN segments can be retried')
    submitted, halted = [], []
    for s in sorted(segs.values(), key=lambda s: (s.est_start_h, s.lane, s.order)):
        r = status[s.name]
        if r['status'] in (cs.DONE, cs.LIVE_S):
            continue
        if r['status'] in (cs.FAILED, cs.UNKNOWN) and s.name not in retry:
            halted.append(f'{s.name}: {r["status"]} job {r["job_id"]} ({r["state"]}) — lane {s.lane} halted; '
                          f'inspect the log, then rerun with --retry {s.name}')
            continue
        deps, blocked = [], None
        for u in (s.lane_prev, s.science_dep):
            if not u:
                continue
            us = status[u]['status']
            if us == cs.LIVE_S:
                deps.append(status[u]['job_id'])
            elif us != cs.DONE:
                blocked = f'{u} is {us}'
        if blocked:
            halted.append(f'{s.name}: not submitted, upstream {blocked}')
            continue
        opts = cs.sbatch_opts(site_for(s), s, R.LOG_DIR)
        if s.kind == 'startup':
            opts.append('--no-requeue')
        if deps:
            opts.append('--dependency=afterok:' + ':'.join(sorted(set(deps), key=lambda d: (len(d), d))))
        argv = [*opts, str(s.script), *s.args]
        if dry:
            print('would submit', s.job_name, ' '.join(o for o in opts if o.startswith('--dependency')))
            status[s.name] = {'status': cs.LIVE_S, 'job_id': f'dry-{s.name}', 'state': 'PENDING', 'reason': 'dry'}
            continue
        st.seg(s.name)['attempts'].append({'job_id': None, 'intent_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                           'deps': deps, 'retry': s.name in retry})
        st.save()
        try:
            jid = slurm.sbatch(argv)
        except RuntimeError as e:
            st.seg(s.name)['attempts'].pop()
            st.save()
            sys.exit(f'sbatch failed for {s.job_name}: {e}\n(nothing recorded for it; rerun submit to reconcile and continue)')
        st.current(s.name)['job_id'] = jid
        st.note(f'submitted {s.name} as {jid} after {deps or "nothing"}')
        st.save()
        status[s.name] = {'status': cs.LIVE_S, 'job_id': jid, 'state': 'PENDING', 'reason': 'new'}
        submitted.append((s.name, jid, deps))
    for name, jid, deps in submitted:
        print(f'submitted {name}: {jid} afterok {",".join(deps) or "-"}')
    for h in halted:
        print('HALTED', h)
    return status


def preflight():
    r = subprocess.run([sys.executable, HERE / 'assert_invariants_3b.py', '--all', '--corpus'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'assert_invariants_3b --all --corpus failed:\n{r.stdout[-1500:]}{r.stderr[-800:]}')
    missing = [s for s in R.SETTINGS if not (R.REPORTS / 'tokenize' / s / 'epoch_stats.json').is_file()]
    if missing:
        sys.exit(f'tokenization not verified for {missing}')
    if not (R.TRAIN8_DIR / 'RESOURCE_AUTHORIZATION.json').is_file():
        sys.exit(f'no {R.TRAIN8_DIR}/RESOURCE_AUTHORIZATION.json (the owner\'s 8-GPU training authorization record)')


def main(argv=None, slurm=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['plan', 'test-only', 'submit', 'status'])
    ap.add_argument('--s-per-it', type=float, default=DESIGN_S_PER_IT, help='planning estimate only (lane order)')
    ap.add_argument('--retry', default='')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    slurm = slurm or cs.Slurm()
    lanes = R.TRAIN8_LANES
    segs = load_segments()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with cs.Lock(STATE_DIR / 'submit.lock'):
        st = cs.State(STATE_DIR / 'state.json')
        if st.data['plan']:
            p = st.data['plan']
            if p['segments'].keys() != segs.keys() or p['lanes'] != lanes:
                sys.exit(f'{STATE_DIR}/state.json holds a plan for other segments/lanes; refusing to re-plan a live grid')
            for n, s in segs.items():
                for k in ('lane', 'order', 'lane_prev', 'est_start_h', 'est_end_h'):
                    setattr(s, k, p['segments'][n][k])
            makespan = max(s.est_end_h for s in segs.values())
        else:
            makespan = assign_breadth_first(segs, lanes, args.s_per_it)
        check_graph(segs, lanes)

        if args.command == 'plan':
            validate_time(args.s_per_it)
            print_plan(segs, lanes, makespan)
            return 0
        if args.command == 'status':
            status = cs.reconcile(segs, st, slurm, slurm.queue())
            st.save()
            cs.print_status(segs, status)
            return 0
        if args.command == 'test-only':
            bad = 0
            for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
                rc, msg = slurm.test_only([*cs.sbatch_opts(site_for(s), s, R.LOG_DIR), str(s.script), *s.args])
                bad += rc != 0
                print(f'  {"OK  " if rc == 0 else "FAIL"} lane {s.lane} {s.job_name}: {msg}')
            print(f'sbatch --test-only: {len(segs) - bad}/{len(segs)} accepted (dependencies omitted; nothing queued)')
            return 1 if bad else 0

        validate_time(st.data['plan']['s_per_it'] if st.data['plan'] else args.s_per_it)
        if not args.dry_run:
            preflight()
        R.LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not st.data['plan']:
            st.data['plan'] = {'lanes': lanes, 'seed': R.SEED, 's_per_it': args.s_per_it, 'layout': R.DEFAULT_LAYOUT,
                               'gpu_ceiling': R.TRAIN8_GPU_CEILING, 'prefix': R.TRAIN8_PREFIX, 'order': 'breadth-first',
                               'repo_head': subprocess.run(['git', '-C', R.REPO, 'rev-parse', 'HEAD'], capture_output=True,
                                                           text=True).stdout.strip(),
                               'segments': {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev,
                                                'science_dep': s.science_dep, 'est_start_h': s.est_start_h,
                                                'est_end_h': s.est_end_h} for n, s in segs.items()}}
            if not args.dry_run:
                st.save()
        submit(segs, st, slurm, {x for x in args.retry.split(',') if x}, dry=args.dry_run)
        if not args.dry_run:
            st.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
