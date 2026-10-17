#!/usr/bin/env python
"""Project-wide GPU-capped, restart-safe submitter for the raw-baseline grid on skipjack.

    python deploy/skipjack_1p5b/capped_submit.py plan      --seed 42 --s-per-it 22.8
    python deploy/skipjack_1p5b/capped_submit.py test-only --seed 42 --s-per-it 22.8
    python deploy/skipjack_1p5b/capped_submit.py submit    --seed 42 --s-per-it 22.8
    python deploy/skipjack_1p5b/capped_submit.py status    --seed 42
    python deploy/skipjack_1p5b/capped_submit.py submit    --seed 42 --s-per-it 22.8 --retry raw_random_seed42_trunk2
    python deploy/skipjack_1p5b/capped_submit.py usage                       # project GPU allocation now
    python deploy/skipjack_1p5b/capped_submit.py guard --need-gpus 4         # for smoke/eval jobs

Why this exists. tools/kys_raw/plan_submit.py submits a seed's 42 segments with only the scientific
afterok graph, so 7 trunks start at once and branches overlap later trunks: up to 14 x 4 = 56 GPUs.
This project may hold at most 20 H100s (five 4-GPU segments) at any moment, including smoke tests and
evaluation, and the account's QoS (jhu2, gres/gpu=32) is shared with other experiments, so the cap
cannot be delegated to it and the account-wide limit must not change.

Mechanism: LANES serial scheduling lanes (default 5). Every segment of the seed is placed in exactly one
lane, and every segment carries `afterok` on its lane predecessor in addition to its scientific
upstream (trunk1 -> trunk2 -> trunk3, trunkN -> epN). A lane therefore never has two jobs that can
run at the same time, so the project never holds more than LANES x 4 GPUs. This is enforced by
Slurm's dependency engine, persists across this script's restarts, survives requeue (the job id is
kept, so afterok still holds) and holds no allocated GPU while waiting. Placement is greedy list
scheduling on the critical path (remaining trunk work first), so lanes interleave chains instead of
leaving one lane with two whole chains; the scientific graph and each branch's source step are
unchanged.

Restart and duplicate safety:
  * single submitter: an exclusive flock on <state-dir>/submit.lock (verified to work on weka);
  * every sbatch is preceded by an `intent` record and followed by the job id, written atomically
    to <state-dir>/state.json; a crash in between is recovered by adopting the one job with that
    exact name found in squeue/sacct (two -> refuse);
  * before anything is submitted the saved state is reconciled with squeue, sacct and the
    checkpoints on disk; a planned job name that is live under an unrecorded id is adopted when the
    segment has no live job and refused as a duplicate otherwise;
  * a segment is DONE only when <checkpoints_path>/<train_steps>/model_config.json exists and no job
    of it is live; COMPLETED in sacct without that file is a failure;
  * a FAILED segment (anything terminal that is not DONE, except a dependency kill) halts its lane:
    nothing after it in the lane, and nothing downstream of it scientifically, is submitted until
    it is named in --retry. The cluster runs kill_invalid_depend, so its pending dependents are
    cancelled by Slurm; those are recognised as dependency kills and resubmitted automatically once
    their upstream is live or done again.
  * afterok is attached only to LIVE upstream jobs (MinJobAge is 30 s, so finished ids disappear from
    the controller); a finished upstream is checked on disk instead.
  * the grid is refused if any other project GPU job (job name prefix kys_, e.g. another seed, a
    smoke test or an evaluation) is live, so seeds never mix and the budget is never shared.

Each job is deploy/slurm/kys_segment.sbatch with the same sbatch options plan_submit.py generates
(test_capped_submit.py checks parity), so its in-job guards are unchanged: skip when the final
checkpoint exists, assert_invariants.py --check-resume before torchrun, fail without a final.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools' / 'kys_raw'))
from plan_submit import slurm_seconds  # noqa: E402
from registry import DEPENDS, FIRST_STEP, KINDS, SETTING_NAMES, parse_settings  # noqa: E402

SBATCH = REPO / 'deploy' / 'slurm' / 'kys_segment.sbatch'
PROJECT_PREFIX = 'kys_'
GPU_CEILING = 20
GPUS_PER_SEGMENT = 4
DEFAULT_CLUSTER = 'skipjack_1p5b_baselines'
DEFAULT_STATE_ROOT = Path('/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/reports/capped_submit')
DEFAULT_LOG_DIR = Path('/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/logs/slurm')
# per-segment fixed cost on top of steps x s/it: startup, data index, checkpoint writes
OVERHEAD_H = 0.5
# squeue states in which a job holds, or may at any moment acquire, an allocation
LIVE = {'PENDING', 'RUNNING', 'REQUEUED', 'REQUEUE_HOLD', 'REQUEUE_FED', 'RESIZING', 'SUSPENDED',
        'COMPLETING', 'CONFIGURING', 'SIGNALING', 'STAGE_OUT', 'SPECIAL_EXIT', 'STOPPED', 'RESV_DEL_HOLD',
        'PREEMPTED'}
HOLDS_GPUS = {'RUNNING', 'COMPLETING', 'CONFIGURING', 'SIGNALING', 'STAGE_OUT', 'SUSPENDED', 'STOPPED',
              'RESIZING'}


# ---------------------------------------------------------------------------------------------------
# Slurm access (replaced by a fake in test_capped_submit.py)

def _gpus(tres: str) -> int:
    """GPU count of a TRES string: 'gres/gpu=4,...' (alloc) or 'gres/gpu:h100:4' (per node)."""
    m = re.search(r'gres/gpu=(\d+)', tres or '')
    if m:
        return int(m.group(1))
    m = re.search(r'gres[/:]gpu(?::[^:,]+)?:(\d+)', tres or '')
    return int(m.group(1)) if m else 0


class Slurm:
    def __init__(self, user: str | None = None):
        self.user = user or os.environ['USER']

    def _run(self, argv: list[str]) -> str:
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'{" ".join(argv[:3])} failed ({r.returncode}): {r.stderr.strip()[-500:]}')
        return r.stdout

    def sbatch(self, argv: list[str]) -> str:
        out = self._run(['sbatch', '--parsable', *argv]).strip().splitlines()
        return out[-1].split(';')[0].strip()

    def test_only(self, argv: list[str]) -> tuple[int, str]:
        r = subprocess.run(['sbatch', '--test-only', *argv], capture_output=True, text=True)
        msg = ' '.join(ln for ln in (r.stdout + r.stderr).splitlines()
                       if 'sbatch' in ln.lower() or 'error' in ln.lower())
        return r.returncode, msg.strip()

    def queue(self) -> list[dict]:
        """All of the user's live jobs: id, name, state, reason, gpus (allocated if running, else requested)."""
        out = self._run(['squeue', '-h', '-u', self.user, '-O',
                         'JobID:|,Name:|,State:|,Reason:|,tres-per-node:|,NumNodes:|,tres-alloc:|'])
        jobs = []
        for ln in out.splitlines():
            p = ln.split('|')
            if len(p) < 7:
                continue
            jid, name, state, reason, per_node, nodes, alloc = (x.strip() for x in p[:7])
            gpus = _gpus(alloc) if state in HOLDS_GPUS and _gpus(alloc) else _gpus(per_node) * int(nodes or 1)
            jobs.append({'id': jid, 'name': name, 'state': state, 'reason': reason, 'gpus': gpus})
        return jobs

    def accounting(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        out = self._run(['sacct', '-X', '-n', '-P', '-j', ','.join(ids), '-o', 'JobIDRaw,JobName,State,Start,End'])
        res = {}
        for ln in out.splitlines():
            p = ln.split('|')
            if len(p) >= 5:
                res[p[0]] = {'name': p[1], 'state': p[2].split()[0], 'start': p[3], 'end': p[4]}
        return res

    def by_name(self, name: str, since: str) -> list[dict]:
        out = self._run(['sacct', '-X', '-n', '-P', '-u', self.user, '-S', since, '--name', name,
                         '-o', 'JobIDRaw,JobName,State,Start,End'])
        return [{'id': p[0], 'name': p[1], 'state': p[2].split()[0]}
                for p in (ln.split('|') for ln in out.splitlines()) if len(p) >= 5 and p[1] == name]

    def cancel(self, ids: list[str]):
        if ids:
            self._run(['scancel', *ids])


# ---------------------------------------------------------------------------------------------------
# plan

@dataclass
class Segment:
    name: str          # <setting>_seed<S>_<kind>
    setting: str
    kind: str
    cfg: Path
    ckpt: Path         # checkpoints_path
    final_step: int
    steps: int
    lane: int = -1
    order: int = -1    # position inside the lane
    lane_prev: str | None = None
    science_dep: str | None = None
    est_start_h: float = 0.0
    est_end_h: float = 0.0

    @property
    def job_name(self) -> str:
        return f'kys_{self.name}'

    @property
    def final(self) -> Path:
        return self.ckpt / str(self.final_step)


def load_segments(cfg_dir: Path, seed: int, settings: list[str]) -> dict[str, Segment]:
    segs = {}
    for s in settings:
        for k in KINDS:
            name = f'{s}_seed{seed}_{k}'
            cfg = cfg_dir / f'{name}.yaml'
            c = yaml.safe_load(cfg.read_text())
            steps = c['tokens']['train_steps']
            segs[name] = Segment(name=name, setting=s, kind=k, cfg=cfg, ckpt=Path(c['checkpoints']['checkpoints_path']),
                                 final_step=steps, steps=steps - FIRST_STEP[k],
                                 science_dep=f'{s}_seed{seed}_{DEPENDS[k]}' if DEPENDS[k] else None)
    return segs


def assign_lanes(segs: dict[str, Segment], lanes: int, s_per_it: float) -> float:
    """Greedy list scheduling into `lanes` serial lanes; returns the estimated makespan in hours.

    Priority = longest remaining path to the end of the segment's chain (trunk work first), ties broken by
    registry order. A lane takes the highest-priority segment whose scientific upstream has finished in the
    simulation; if none is ready it waits for the next finish. Every edge (lane predecessor and scientific
    upstream) therefore points from an earlier to a later simulated start, so the graph is acyclic.
    """
    dur = {n: s.steps * s_per_it / 3600 + OVERHEAD_H for n, s in segs.items()}
    children: dict[str, list[str]] = {n: [] for n in segs}
    for n, s in segs.items():
        if s.science_dep:
            children[s.science_dep].append(n)

    def tail(n):
        return dur[n] + max((tail(c) for c in children[n]), default=0.0)

    rank = {n: i for i, n in enumerate(segs)}
    prio = {n: tail(n) for n in segs}
    lane_free = [0.0] * lanes
    lane_last: list[str | None] = [None] * lanes
    lane_len = [0] * lanes
    end: dict[str, float] = {}
    todo = set(segs)
    while todo:
        li = min(range(lanes), key=lambda i: (lane_free[i], i))
        t = lane_free[li]
        ready = [n for n in todo if segs[n].science_dep is None or end.get(segs[n].science_dep, 1e18) <= t]
        if not ready:
            nxt = min(end[segs[n].science_dep] for n in todo if segs[n].science_dep in end)
            lane_free[li] = max(t, nxt)
            continue
        n = max(ready, key=lambda x: (prio[x], -rank[x]))
        s = segs[n]
        s.lane, s.order, s.lane_prev = li, lane_len[li], lane_last[li]
        s.est_start_h, s.est_end_h = t, t + dur[n]
        end[n] = s.est_end_h
        lane_free[li], lane_last[li], lane_len[li] = s.est_end_h, n, lane_len[li] + 1
        todo.remove(n)
    return max(end.values())


def max_concurrency(segs: dict[str, Segment]) -> int:
    ev = sorted([(s.est_start_h, 1) for s in segs.values()] + [(s.est_end_h, -1) for s in segs.values()],
                key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def check_graph(segs: dict[str, Segment], lanes: int):
    """Invariants of a plan: every segment in one lane, lanes are chains, edges acyclic, science preserved."""
    errs = []
    by_lane: dict[int, list[Segment]] = {}
    for s in segs.values():
        if not 0 <= s.lane < lanes:
            errs.append(f'{s.name}: lane {s.lane}')
        by_lane.setdefault(s.lane, []).append(s)
    for li, ss in by_lane.items():
        ss.sort(key=lambda s: s.order)
        if [s.order for s in ss] != list(range(len(ss))):
            errs.append(f'lane {li}: orders {[s.order for s in ss]}')
        for a, b in zip(ss, ss[1:]):
            if b.lane_prev != a.name:
                errs.append(f'lane {li}: {b.name}.lane_prev={b.lane_prev}, expected {a.name}')
        if ss and ss[0].lane_prev is not None:
            errs.append(f'lane {li}: head {ss[0].name} has lane_prev')
    for s in segs.values():
        want = DEPENDS[s.kind]
        if (s.science_dep is None) != (want is None) or (want and not s.science_dep.endswith('_' + want)):
            errs.append(f'{s.name}: science_dep {s.science_dep}')
    # acyclicity over lane + science edges
    state: dict[str, int] = {}

    def visit(n):
        if state.get(n) == 1:
            errs.append(f'cycle through {n}')
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in (segs[n].lane_prev, segs[n].science_dep):
            if d:
                visit(d)
        state[n] = 2
    for n in segs:
        visit(n)
    if errs:
        raise SystemExit('plan invariant violated:\n  ' + '\n  '.join(errs))


# ---------------------------------------------------------------------------------------------------
# state

class State:
    """<state-dir>/state.json: the plan (lane placement) and every submission attempt per segment."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {'plan': None, 'segments': {}, 'log': []}

    def seg(self, name: str) -> dict:
        return self.data['segments'].setdefault(name, {'attempts': []})

    def current(self, name: str) -> dict | None:
        a = self.seg(name)['attempts']
        return a[-1] if a else None

    def note(self, msg: str):
        self.data['log'].append(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}')

    def save(self):
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


class Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.fh = open(self.path, 'a+')
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.seek(0)
            raise SystemExit(f'another capped_submit.py holds {self.path} ({self.fh.read().strip()}); refusing')
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f'pid {os.getpid()} host {os.uname().nodename} since {time.strftime("%F %T")}\n')
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


# ---------------------------------------------------------------------------------------------------
# reconcile

DONE, LIVE_S, FAILED, DEPKILLED, UNSUBMITTED, UNKNOWN = 'DONE', 'LIVE', 'FAILED', 'DEP_KILLED', 'UNSUBMITTED', 'UNKNOWN'


def reconcile(segs: dict[str, Segment], st: State, slurm, queue: list[dict]) -> dict[str, dict]:
    """Status per segment from saved attempts, squeue, sacct and the disk. Adopts orphans; refuses duplicates."""
    live_by_name: dict[str, list[dict]] = {}
    for j in queue:
        live_by_name.setdefault(j['name'], []).append(j)
    recorded = {a['job_id'] for n in segs for a in st.seg(n)['attempts'] if a.get('job_id')}
    acct = slurm.accounting(sorted(recorded - {j['id'] for j in queue}))
    out = {}
    for n, s in segs.items():
        live = live_by_name.get(s.job_name, [])
        cur = st.current(n)
        # crash between intent and job id: adopt the single job of that name submitted since the intent
        if cur and not cur.get('job_id'):
            found = {j['id']: j for j in live}
            for j in slurm.by_name(s.job_name, cur['intent_at']):
                found.setdefault(j['id'], j)
            if len(found) > 1:
                raise SystemExit(f'{s.job_name}: {len(found)} jobs {sorted(found)} after an interrupted submission; '
                                 f'resolve by hand (scancel the extra) before continuing')
            if found:
                cur['job_id'] = next(iter(found))
                cur['adopted'] = True
                st.note(f'adopted {cur["job_id"]} for {n} (interrupted submission)')
            else:
                st.seg(n)['attempts'].pop()
                st.note(f'{n}: interrupted submission left no job; intent dropped')
                cur = st.current(n)
        ids = {a['job_id'] for a in st.seg(n)['attempts'] if a.get('job_id')}
        unknown_live = [j for j in live if j['id'] not in ids]
        if unknown_live or len(live) > 1:
            # not submitted by this tool (or a second copy): its dependencies are unknown, so it may run outside
            # the lanes. Refuse rather than adopt; only an interrupted submission (above) is adopted.
            raise SystemExit(f'{s.job_name} is live as {[j["id"] for j in live]}, recorded ids {sorted(ids)}: a job this '
                             f'tool did not submit, or a duplicate; resolve by hand (scancel) before continuing')
        done_on_disk = (s.final / 'model_config.json').is_file()
        if live:
            out[n] = {'status': LIVE_S, 'job_id': live[0]['id'], 'state': live[0]['state'], 'reason': live[0]['reason']}
        elif done_on_disk:
            out[n] = {'status': DONE, 'job_id': cur and cur.get('job_id'), 'state': 'final on disk'}
        elif cur is None:
            out[n] = {'status': UNSUBMITTED, 'job_id': None, 'state': '-'}
        else:
            a = acct.get(cur['job_id'])
            if a is None:
                out[n] = {'status': UNKNOWN, 'job_id': cur['job_id'], 'state': 'not in squeue or sacct'}
            else:
                never_ran = a['start'] in ('None', 'Unknown', '')
                out[n] = {'status': DEPKILLED if a['state'] == 'CANCELLED' and never_ran else FAILED,
                          'job_id': cur['job_id'], 'state': a['state'] + ('' if a['state'] != 'COMPLETED' else ' without final')}
        out[n]['retry_ok'] = False
    # a cancelled-before-start job is a dependency kill only if one of its upstreams really failed
    for n, s in segs.items():
        if out[n]['status'] == DEPKILLED:
            ups = [d for d in (s.lane_prev, s.science_dep) if d]
            if not any(out[u]['status'] in (FAILED, DEPKILLED) or _had_failure(st, u) for u in ups):
                out[n]['status'] = FAILED
                out[n]['state'] += ' (cancelled before start, no failed upstream: manual cancel?)'
    return out


def _had_failure(st: State, name: str) -> bool:
    return len(st.seg(name)['attempts']) > 1


# ---------------------------------------------------------------------------------------------------
# commands

def sbatch_opts(c: dict, s: Segment, log_dir: Path) -> list[str]:
    """Identical to tools/kys_raw/plan_submit.py's per-segment options (except --output's directory)."""
    sl = c['slurm']
    exports = {'KYS_REPO': c.get('repo_dir'), 'KYS_ENV_ACTIVATE': c.get('env_activate'),
               'KYS_PYTHON_INCLUDE': c.get('python_include')}
    export_arg = ','.join(['ALL'] + [f'{k}={v}' for k, v in exports.items() if v])
    opts = [f'--job-name={s.job_name}', f'--partition={sl["partition"]}', f'--gres={sl["gres"]}',
            f'--nodes={sl.get("nodes") or 1}', f'--cpus-per-task={sl["cpus_per_task"]}', f'--time={sl["time"]}',
            f'--output={log_dir}/%x.%j.log', f'--export={export_arg}']
    opts += [f'--account={sl["account"]}'] if sl.get('account') else []
    opts += [f'--qos={sl["qos"]}'] if sl.get('qos') else []
    return opts


def validate_profile(c: dict, cluster: str, s_per_it: float | None, lanes: int):
    sl = c.get('slurm') or {}
    for k in ('partition', 'gres', 'cpus_per_task', 'time'):
        if sl.get(k) is None:
            sys.exit(f'{cluster}.slurm.{k} is null')
    if _gpus('gres/' + sl['gres'] if not sl['gres'].startswith('gres') else sl['gres']) != GPUS_PER_SEGMENT \
            or int(sl.get('nodes') or 1) != 1 or c['dp'] * c['tp'] * c['pp'] != GPUS_PER_SEGMENT:
        sys.exit(f'{cluster}: every segment must be exactly 1 node x {GPUS_PER_SEGMENT} GPUs '
                 f'(gres {sl["gres"]}, nodes {sl.get("nodes")}, dp*tp*pp {c["dp"] * c["tp"] * c["pp"]})')
    if lanes * GPUS_PER_SEGMENT > GPU_CEILING:
        sys.exit(f'{lanes} lanes x {GPUS_PER_SEGMENT} GPUs exceeds the project ceiling of {GPU_CEILING}')
    have = slurm_seconds(sl['time'])
    if have > 3 * 86400:
        sys.exit(f'{cluster}.slurm.time {sl["time"]} exceeds the 3-day site limit')
    if s_per_it:  # same rule as plan_submit.py
        need = max(1500, 14305 - FIRST_STEP['ep3']) * s_per_it
        if have < 1.1 * need + 1800:
            sys.exit(f'{cluster}.slurm.time {sl["time"]} too short at {s_per_it} s/it '
                     f'(need >= {(1.1 * need + 1800) / 3600:.1f} h)')
    if c.get('repo_dir') and Path(c['repo_dir']).resolve() != REPO:
        sys.exit(f'{cluster}.repo_dir is {c["repo_dir"]}, but this tool lives in {REPO}; refusing to mix checkouts')


def check_configs(segs: dict[str, Segment]):
    for s in segs.values():
        r = subprocess.run([sys.executable, REPO / 'tools/assert_invariants.py', '--config', s.cfg, '--skip-corpus',
                            '--skip-env'], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{s.cfg} fails assert_invariants.py:\n{r.stdout[-600:]}{r.stderr[-600:]}')


def seed_trunk(c: dict, seed: int, s: Segment):
    """plan_submit.py's trunk seeding: init -> <trunk>/0 + latest.txt, only if latest.txt is absent."""
    trunk = s.ckpt
    if (trunk / 'latest.txt').is_file():
        return False
    if trunk.exists() and any(p.is_dir() and p.name.isdigit() for p in trunk.iterdir()):
        sys.exit(f'{trunk} holds step directories but no latest.txt; refusing to seed over a progressed trunk')
    init = Path(c['init_root']) / f'_init_1.5B_seed{seed}' / '0'
    for f in ('model_config.json', 'optimizer/optimizer_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt',
              'lr_scheduler/lr_scheduler_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt'):
        if not (init / f).is_file() or (init / f).stat().st_size == 0:
            sys.exit(f'init {init} is incomplete: missing {f}')
    n = sum(1 for p in init.rglob('*') if p.is_file())
    if n != 180:
        sys.exit(f'init {init} has {n} files, expected 180')
    trunk.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-a', str(init), str(trunk / '0')], check=True)
    (trunk / 'latest.txt').write_text('0\n')
    return True


def other_project_jobs(queue: list[dict], segs: dict[str, Segment]) -> list[dict]:
    mine = {s.job_name for s in segs.values()}
    return [j for j in queue if j['name'].startswith(PROJECT_PREFIX) and j['name'] not in mine]


def project_gpus(queue: list[dict]) -> tuple[int, int]:
    """(GPUs allocated now, GPUs requested by live-but-not-running jobs) for project jobs."""
    held = sum(j['gpus'] for j in queue if j['name'].startswith(PROJECT_PREFIX) and j['state'] in HOLDS_GPUS)
    waiting = sum(j['gpus'] for j in queue if j['name'].startswith(PROJECT_PREFIX) and j['state'] not in HOLDS_GPUS)
    return held, waiting


def print_plan(segs: dict[str, Segment], lanes: int, makespan: float):
    print(f'{"lane":>4s} {"#":>2s} {"segment":44s} {"steps":>6s} {"start h":>8s} {"end h":>8s}  after (lane / science)')
    for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
        print(f'{s.lane:4d} {s.order:2d} {s.name:44s} {s.steps:6d} {s.est_start_h:8.1f} {s.est_end_h:8.1f}  '
              f'{s.lane_prev or "-"} / {s.science_dep or "-"}')
    print(f'\n{len(segs)} segments in {lanes} lanes; estimated makespan {makespan:.1f} h ({makespan / 24:.1f} d); '
          f'simulated peak concurrency {max_concurrency(segs)} segments = {max_concurrency(segs) * GPUS_PER_SEGMENT} GPUs '
          f'(ceiling {GPU_CEILING})')


def print_status(segs, status):
    print(f'{"lane":>4s} {"#":>2s} {"segment":44s} {"status":12s} {"job":>10s}  state')
    for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
        r = status[s.name]
        print(f'{s.lane:4d} {s.order:2d} {s.name:44s} {r["status"]:12s} {str(r["job_id"] or "-"):>10s}  '
              f'{r["state"]}{" (" + r["reason"] + ")" if r.get("reason") not in (None, "None", "") else ""}')
    counts = {}
    for r in status.values():
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print('\n' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))


def submit(segs, st: State, slurm, c, seed, log_dir: Path, retry: set[str], dry: bool = False) -> dict:
    queue = slurm.queue()
    others = [j for j in other_project_jobs(queue, segs) if j['state'] in LIVE]
    if others:
        sys.exit('other project GPU jobs are live; the grid shares the 20-GPU budget with nothing and seeds never mix:\n  '
                 + '\n  '.join(f'{j["id"]} {j["name"]} {j["state"]}' for j in others))
    status = reconcile(segs, st, slurm, queue)
    if not dry:
        st.save()
    for r in retry:
        if r not in segs:
            sys.exit(f'--retry {r}: not a segment of this plan')
        if status[r]['status'] not in (FAILED, UNKNOWN):
            sys.exit(f'--retry {r}: status is {status[r]["status"]}, only FAILED/UNKNOWN segments can be retried')
    order = sorted(segs.values(), key=lambda s: (s.est_start_h, s.lane, s.order))
    submitted, halted = [], []
    for s in order:
        r = status[s.name]
        if r['status'] in (DONE, LIVE_S):
            continue
        if r['status'] in (FAILED, UNKNOWN) and s.name not in retry:
            halted.append(f'{s.name}: {r["status"]} job {r["job_id"]} ({r["state"]}) — lane {s.lane} halted; '
                          f'inspect the log, then rerun with --retry {s.name}')
            continue
        deps, blocked = [], None
        for u in (s.lane_prev, s.science_dep):
            if not u:
                continue
            us = status[u]['status']
            if us == LIVE_S:
                deps.append(status[u]['job_id'])
            elif us != DONE:
                blocked = f'{u} is {us}'
        if blocked:
            halted.append(f'{s.name}: not submitted, upstream {blocked}')
            continue
        if s.kind == 'trunk1' and not dry:
            if seed_trunk(c, seed, s):
                st.note(f'seeded {s.ckpt}/0 from init')
        opts = sbatch_opts(c, s, log_dir)
        if deps:
            opts.append('--dependency=afterok:' + ':'.join(sorted(set(deps), key=lambda d: (len(d), d))))
        argv = [*opts, str(SBATCH), str(s.cfg)]
        if dry:
            print('would submit', s.job_name, ' '.join(o for o in opts if o.startswith('--dependency')))
            status[s.name] = {'status': LIVE_S, 'job_id': f'dry-{s.name}', 'state': 'PENDING', 'reason': 'dry'}
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
        status[s.name] = {'status': LIVE_S, 'job_id': jid, 'state': 'PENDING', 'reason': 'new'}
        submitted.append((s.name, jid, deps))
    for name, jid, deps in submitted:
        print(f'submitted {name}: {jid} afterok {",".join(deps) or "-"}')
    for h in halted:
        print('HALTED', h)
    return status


def main(argv=None, slurm=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['plan', 'test-only', 'submit', 'status', 'usage', 'guard'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--cluster', default=DEFAULT_CLUSTER)
    ap.add_argument('--profile', type=Path, default=REPO / 'deploy' / 'clusters.yaml')
    ap.add_argument('--configs', type=Path, help='default: configs/1.5B-baseline-seed<S>/filled')
    ap.add_argument('--settings', default=','.join(SETTING_NAMES))
    ap.add_argument('--lanes', type=int, default=GPU_CEILING // GPUS_PER_SEGMENT)
    ap.add_argument('--s-per-it', type=float, help='measured s/step; required for plan/test-only/submit')
    ap.add_argument('--state-dir', type=Path, help=f'default: {DEFAULT_STATE_ROOT}/seed<S>')
    ap.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument('--retry', default='', help='comma-separated FAILED segments to resubmit (submit only)')
    ap.add_argument('--need-gpus', type=int, help='guard: GPUs a non-training project job would add')
    ap.add_argument('--dry-run', action='store_true', help='submit: reconcile and print, submit nothing')
    args = ap.parse_args(argv)
    slurm = slurm or Slurm()

    if args.command in ('usage', 'guard'):
        q = slurm.queue()
        held, waiting = project_gpus(q)
        for j in q:
            if j['name'].startswith(PROJECT_PREFIX):
                print(f'{j["id"]:>10s} {j["name"]:50s} {j["state"]:12s} gpus={j["gpus"]}')
        print(f'project GPUs allocated now: {held} / {GPU_CEILING}; requested by waiting project jobs: {waiting}')
        if args.command == 'guard':
            need = args.need_gpus or 0
            if held + waiting + need > GPU_CEILING:
                sys.exit(f'guard: {held} held + {waiting} waiting + {need} new > {GPU_CEILING}; refusing')
            print(f'guard: OK to add {need} GPUs')
        return 0

    if args.seed is None:
        sys.exit('--seed is required')
    prof = yaml.safe_load(args.profile.read_text())
    c = prof['clusters'][args.cluster]
    if prof.get('baseline_seed_assignment', {}).get(args.seed) != args.cluster:
        sys.exit(f'baseline_seed_assignment[{args.seed}] is not {args.cluster}')
    settings = parse_settings(args.settings)
    cfg_dir = args.configs or REPO / 'configs' / f'1.5B-baseline-seed{args.seed}' / 'filled'
    segs = load_segments(cfg_dir, args.seed, settings)
    state_dir = args.state_dir or DEFAULT_STATE_ROOT / f'seed{args.seed}'
    state_dir.mkdir(parents=True, exist_ok=True)

    prev = args.seed - 1
    if args.command == 'submit' and prev in prof.get('baseline_seed_assignment', {}) \
            and not (DEFAULT_STATE_ROOT / f'seed{prev}' / 'SEED_COMPLETE').is_file() and args.state_dir is None:
        sys.exit(f'seed {prev} has no SEED_COMPLETE marker (all 21 finals verified, uploaded, evaluated, GPUs released); '
                 f'seeds run one at a time')
    if args.command != 'status' and not args.s_per_it:
        sys.exit('--s-per-it (measured) is required')
    validate_profile(c, args.cluster, args.s_per_it, args.lanes)

    with Lock(state_dir / 'submit.lock'):
        st = State(state_dir / 'state.json')
        if st.data['plan']:
            p = st.data['plan']
            if p['segments'].keys() != segs.keys() or p['lanes'] != args.lanes:
                sys.exit(f'{state_dir}/state.json holds a plan for other segments/lanes; refusing to re-plan a live grid')
            for n, s in segs.items():
                s.lane, s.order, s.lane_prev = p['segments'][n]['lane'], p['segments'][n]['order'], p['segments'][n]['lane_prev']
                s.est_start_h, s.est_end_h = p['segments'][n]['est_start_h'], p['segments'][n]['est_end_h']
            makespan = max(s.est_end_h for s in segs.values())
        else:
            makespan = assign_lanes(segs, args.lanes, args.s_per_it or 22.8)
        check_graph(segs, args.lanes)
        if max_concurrency(segs) > args.lanes:
            sys.exit('simulated concurrency exceeds the lane count')

        if args.command == 'plan':
            print_plan(segs, args.lanes, makespan)
            (state_dir / 'plan_preview.json').write_text(json.dumps(
                {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev, 'science_dep': s.science_dep,
                     'est_start_h': round(s.est_start_h, 2), 'est_end_h': round(s.est_end_h, 2)} for n, s in segs.items()},
                indent=1))
            return 0
        if args.command == 'status':
            status = reconcile(segs, st, slurm, slurm.queue())
            st.save()
            print_status(segs, status)
            held, waiting = project_gpus(slurm.queue())
            print(f'project GPUs allocated now: {held} / {GPU_CEILING} (waiting jobs request {waiting})')
            return 0
        check_configs(segs)
        if args.command == 'test-only':
            bad = 0
            for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
                rc, msg = slurm.test_only([*sbatch_opts(c, s, args.log_dir), str(SBATCH), str(s.cfg)])
                bad += rc != 0
                print(f'  {"OK  " if rc == 0 else "FAIL"} lane {s.lane} {s.name}: {msg}')
            if bad:
                sys.exit(f'sbatch --test-only rejected {bad} segment(s)')
            print(f'sbatch --test-only accepted all {len(segs)} segments (dependencies omitted)')
            return 0
        # submit
        args.log_dir.mkdir(parents=True, exist_ok=True)
        if not st.data['plan']:
            st.data['plan'] = {'lanes': args.lanes, 'seed': args.seed, 's_per_it': args.s_per_it, 'cluster': args.cluster,
                               'repo_head': subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True,
                                                           text=True).stdout.strip(),
                               'segments': {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev,
                                                'science_dep': s.science_dep, 'est_start_h': s.est_start_h,
                                                'est_end_h': s.est_end_h} for n, s in segs.items()}}
            if not args.dry_run:
                st.save()
        retry = {r for r in args.retry.split(',') if r}
        submit(segs, st, slurm, c, args.seed, args.log_dir, retry, dry=args.dry_run)
        if not args.dry_run:
            st.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
