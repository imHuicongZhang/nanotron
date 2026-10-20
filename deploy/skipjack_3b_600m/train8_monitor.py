#!/usr/bin/env python
"""One tick of the durable 3B training monitor (run by train8_monitor.sbatch every 30 min; CPU only).

    $P deploy/skipjack_3b_600m/train8_monitor.py tick [--dry-run]
    $P deploy/skipjack_3b_600m/train8_monitor.py status          # last tick + STATUS.md

Each tick:
  1. another monitor queued -> exit (one chain); <TRAIN8_DIR>/STOP exists -> exit without a successor;
  2. queue the successor first (--begin=now+30min, --dependency=singleton), so a crash in this tick never ends the chain;
  3. train8_3b.py submit: idempotent reconcile; re-submits dependency-killed jobs; never retries a FAILED segment
     (a failure halts its lane until a human inspects it and runs submit --retry <segment>);
  4. health of every RUNNING rw3b_gpu_ segment from its log: finite loss and grad_norm, logged lr == nanotron's scheduler
     for that segment's config, consumed_tokens == step x 2,097,152. A demonstrated failure scancels that job (its lane and
     scientific dependents then stop through afterok) and writes alerts/<time>.json; nothing is resubmitted;
  5. throughput (median s/it of the last 50 steps per running segment) and an ETA for every 20B/40B/60B final from the
     saved lane plan, at the measured s/it, for queue waits of 0 h and of QUEUE_WAIT_H per job start;
  6. when all 28 jobs are DONE: write COMPLETE and stop the chain.
State: <TRAIN8_DIR>/monitor_state.json, status.json, STATUS.md; log logs/slurm/rw3b_train8_monitor.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry_3b as R  # noqa: E402

NAME = 'rw3b_train8_monitor'
INTERVAL = '30minutes'
D = R.TRAIN8_DIR
LOG = R.LOG_DIR / f'{NAME}.log'
STATE = D / 'monitor_state.json'
QUEUE_WAIT_H = 12.0          # scenario: mean wait per job start on the h100 queue (unmeasured; see STATUS.md)
DEADLINE = dt.datetime(2026, 11, 5, 0, 0)   # America/New_York (the cluster's local time)


def sh(argv, dry=False):
    if dry and argv[0] in ('sbatch', 'scancel'):
        print('[dry-run]', ' '.join(map(str, argv)))
        return subprocess.CompletedProcess(argv, 0, 'dry', '')
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True)


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {'ticks': [], 'alerts': [], 'throughput': {}}


def gpu_jobs() -> list[dict]:
    out = sh(['squeue', '-h', '-u', os.environ['USER'], '-O', 'JobID:|,Name:|,State:|,Reason:|,tres-per-node:|,StartTime:|']).stdout
    jobs = []
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split('|')]
        if len(p) >= 6 and p[1].startswith(R.TRAIN8_PREFIX):
            jobs.append({'id': p[0], 'name': p[1], 'state': p[2], 'reason': p[3], 'tres': p[4], 'start': p[5]})
    return jobs


def health(job: dict, segs: dict) -> dict:
    import startup_3b as S
    seg = segs.get(job['name'][len(R.TRAIN8_PREFIX):])
    if seg is None or seg.kind == 'startup':
        return {'skipped': 'startup validates itself'} if seg else {'skipped': 'not in plan'}
    log = R.LOG_DIR / f'{job["name"]}.{job["id"]}.log'
    steps = S.parse_log(log.read_text(errors='replace')) if log.is_file() else []
    probs = S.step_problems(steps, seg.cfg, seg.name)
    tail = [x['s'] for x in steps[-52:]][2:]
    return {'segment': seg.name, 'log': str(log), 'steps_logged': len(steps), 'last': steps[-1] if steps else None,
            's_per_it': statistics.median(tail) if len(tail) >= 10 else None, 'problems': probs}


def project(segs: dict, plan: dict, done: set, running: dict, s_per_it: float, wait_h: float, now: float,
            slurm_start: dict | None = None) -> dict:
    """End time (epoch s) of every job: lanes in plan order, each start after its lane predecessor and scientific
    upstream, plus wait_h for every job that has not started yet; a pending job with a Slurm start estimate starts no
    earlier than that estimate (and gets no extra wait)."""
    slurm_start = slurm_start or {}
    end = {}
    order = sorted(segs, key=lambda n: (plan['segments'][n]['lane'], plan['segments'][n]['order']))
    lanes = {}
    for n in order:
        lanes.setdefault(plan['segments'][n]['lane'], []).append(n)
    idx = {li: 0 for li in lanes}
    free = {li: now for li in lanes}
    while any(idx[li] < len(ns) for li, ns in lanes.items()):
        progressed = False
        for li, ns in lanes.items():
            if idx[li] >= len(ns):
                continue
            n = ns[idx[li]]
            s = segs[n]
            if n in done:
                end[n] = now
            elif n in running:
                end[n] = now + max(0, s.final_step - running[n]) * s_per_it + 600
            else:
                if s.science_dep and s.science_dep not in end:
                    continue
                start = max(free[li], end.get(s.science_dep, now))
                start = max(start, slurm_start[n]) if n in slurm_start else start + wait_h * 3600
                end[n] = start + s.steps * s_per_it + 1800
            free[li] = end[n]
            idx[li] += 1
            progressed = True
        if not progressed:
            raise RuntimeError('projection deadlock (plan graph)')
    return end


def fmt(t: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(t))


def tick(dry: bool) -> int:
    D.mkdir(parents=True, exist_ok=True)
    R.LOG_DIR.mkdir(parents=True, exist_ok=True)
    me = os.environ.get('SLURM_JOB_ID', '')
    now = time.time()
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    others = [j for j in sh(['squeue', '-h', '-u', os.environ['USER'], '-n', NAME, '-o', '%i']).stdout.split() if j != me]
    if others:
        print(f'[monitor] {stamp} another monitor is queued ({others}); exiting without a successor')
        return 0
    if (D / 'STOP').exists():
        print(f'[monitor] {stamp} {D}/STOP present; chain stopped')
        return 0
    st = load()
    rec = {'at': stamp, 'job': me, 'head': sh(['git', '-C', R.REPO, 'rev-parse', 'HEAD']).stdout.strip()}
    nxt = sh(['sbatch', '--parsable', f'--begin=now+{INTERVAL}', '--dependency=singleton', f'--output={LOG}',
              HERE / 'train8_monitor.sbatch'], dry)
    rec['successor'] = nxt.stdout.strip().split(';')[0] if nxt.returncode == 0 else f'SBATCH FAILED: {nxt.stderr.strip()}'

    r = sh([sys.executable, HERE / 'train8_3b.py', 'submit', *(['--dry-run'] if dry else [])])
    rec['submit'] = {'rc': r.returncode, 'out': (r.stdout + r.stderr).strip()[-2500:]}

    sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
    import train8_3b as T
    segs = T.load_segments()
    plan = json.loads((T.STATE_DIR / 'state.json').read_text())['plan'] if (T.STATE_DIR / 'state.json').is_file() else None
    jobs = gpu_jobs()
    run_jobs = [j for j in jobs if j['state'] == 'RUNNING']
    rec['gpu_jobs'] = jobs
    rec['gpus_running'] = 4 * len(run_jobs)
    if rec['gpus_running'] > R.TRAIN8_GPU_CEILING:
        a = {'at': stamp, 'kind': 'ceiling', 'detail': f'{len(run_jobs)} rw3b_gpu_ jobs running > {R.TRAIN8_LANES}'}
        st['alerts'].append(a)
    rec['health'] = {}
    running_step = {}
    for j in run_jobs:
        h = health(j, segs)
        rec['health'][j['name']] = h
        if j['name'] == R.TRAIN8_PREFIX + R.STARTUP:
            running_step[R.STARTUP] = 0
        if h.get('s_per_it'):
            st['throughput'][h['segment']] = {'s_per_it': h['s_per_it'], 'at': stamp, 'last_step': h['last']['step']}
        if h.get('last'):
            running_step[h['segment']] = h['last']['step']
        elif 'segment' in h:
            running_step[h['segment']] = T.R.FIRST_STEP[segs[h['segment']].kind]
        if h.get('problems'):
            a = {'at': stamp, 'kind': 'health', 'job': j['id'], 'name': j['name'], 'problems': h['problems'],
                 'action': 'scancel (demonstrated failure; lane halted; not resubmitted)'}
            sh(['scancel', j['id']], dry)
            st['alerts'].append(a)
            (D / 'alerts').mkdir(exist_ok=True)
            if not dry:
                (D / 'alerts' / f'{stamp}_{j["name"]}.json').write_text(json.dumps(a, indent=1))
    startup_rep = T.STARTUP_WORK / 'startup_report.json'
    if startup_rep.is_file():
        sr = json.loads(startup_rep.read_text())
        rec['startup'] = {'ok': sr['ok'], 'problems': sr['problems'], 's_per_it': sr.get('s_per_it_median_production_layout'),
                          'peak_gib': sr.get('peak_gib'), 'resume_vs_uninterrupted': sr.get('resume_vs_uninterrupted')}

    # status of all 28 jobs (disk + squeue), throughput, projection
    done = {n for n, s in segs.items() if (s.final / 'model_config.json').is_file() and not any(
        j['name'] == s.job_name for j in jobs)}
    meas = [v['s_per_it'] for v in st['throughput'].values()]
    s_meas = statistics.median(meas) if meas else (rec.get('startup') or {}).get('s_per_it')
    rec['s_per_it_measured'] = s_meas
    rec['done'] = sorted(done)
    proj = {}
    starts = {}
    for j in jobs:
        if j['state'] == 'PENDING' and j['start'][:2] == '20':
            try:
                starts[j['name'][len(R.TRAIN8_PREFIX):]] = time.mktime(time.strptime(j['start'], '%Y-%m-%dT%H:%M:%S'))
            except ValueError:
                pass
    if plan:
        for label, spi, wait in (('measured_no_wait', s_meas, 0.0), (f'measured_wait{QUEUE_WAIT_H:g}h', s_meas, QUEUE_WAIT_H),
                                 ('design_32.2_no_wait', T.DESIGN_S_PER_IT, 0.0)):
            if not spi:
                continue
            e = project(segs, plan, done, running_step, spi, wait, now, starts)
            proj[label] = {s: {f'{20 * (i + 1)}B': fmt(e[R.seg_name(s, f'ep{i + 1}')]) for i in range(3)} for s in R.SETTINGS}
            proj[label]['all'] = fmt(max(e.values()))
            proj[label]['meets_deadline'] = max(e.values()) <= DEADLINE.timestamp()
    rec['projection'] = proj
    rec['slurm_start_estimates'] = {n: fmt(t) for n, t in starts.items()}
    st['ticks'] = st['ticks'][-500:] + [rec]
    st['last'] = rec
    if len(done) == len(segs):
        (D / 'COMPLETE').write_text(json.dumps({'at': stamp, 'segments': len(segs)}, indent=1))
        if nxt.returncode == 0:
            sh(['scancel', rec['successor']], dry)
        rec['action'] = 'all 28 jobs DONE; COMPLETE written; chain stopped'
    if not dry:
        STATE.write_text(json.dumps(st, indent=1))
        (D / 'status.json').write_text(json.dumps(rec, indent=1))
        write_md(rec, st)
    print(json.dumps({k: v for k, v in rec.items() if k != 'gpu_jobs'}, indent=1))
    return 0


def write_md(rec: dict, st: dict):
    L = [f'# 3B training (8-GPU budget) — monitor tick {rec["at"]} (job {rec["job"]}, next {rec["successor"]})', '',
         f'GPUs running (rw3b_gpu_*): **{rec["gpus_running"]}** / {R.TRAIN8_GPU_CEILING}. '
         f'Measured s/it: **{rec["s_per_it_measured"]}**. DONE: {len(rec["done"])}/28.', '',
         '| job | name | state | reason / start estimate |', '|---|---|---|---|']
    L += [f'| {j["id"]} | {j["name"]} | {j["state"]} | {j["reason"]} / {j["start"]} |' for j in rec['gpu_jobs']]
    if rec.get('startup'):
        L += ['', f'Startup validation: ok={rec["startup"]["ok"]}, s/it {rec["startup"]["s_per_it"]}, peak '
              f'{rec["startup"]["peak_gib"]} GiB, problems {rec["startup"]["problems"]}']
    for n, h in rec['health'].items():
        L.append(f'- {n}: {h}' if 'skipped' in h else f'- {n}: step {h["last"] and h["last"]["step"]}, loss '
                 f'{h["last"] and h["last"]["loss"]}, lr {h["last"] and h["last"]["lr"]}, s/it {h["s_per_it"]}, problems {h["problems"]}')
    for label, p in rec['projection'].items():
        L += ['', f'**Projection {label}** (all finals {p["all"]}; meets 2026-11-05 00:00: {p["meets_deadline"]})', '',
              '| setting | 20B | 40B | 60B |', '|---|---|---|---|']
        L += [f'| {s} | {p[s]["20B"]} | {p[s]["40B"]} | {p[s]["60B"]} |' for s in R.SETTINGS]
    if st['alerts']:
        L += ['', '## ALERTS', ''] + [f'- {json.dumps(a)}' for a in st['alerts'][-10:]]
    L += ['', f'submit rc {rec["submit"]["rc"]}:', '```', rec['submit']['out'][-1500:], '```']
    (D / 'STATUS.md').write_text('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['tick', 'status'])
    ap.add_argument('--dry-run', action='store_true', help='no sbatch/scancel, submit --dry-run, no state write')
    args = ap.parse_args(argv)
    if args.command == 'status':
        print((D / 'STATUS.md').read_text() if (D / 'STATUS.md').is_file() else 'no STATUS.md yet')
        return 0
    return tick(args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
