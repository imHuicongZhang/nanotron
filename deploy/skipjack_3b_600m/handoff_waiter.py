#!/usr/bin/env python
"""One tick of the durable baseline -> 3B handoff waiter (run by handoff_waiter.sbatch; never submits training).

    python deploy/skipjack_3b_600m/handoff_waiter.py tick [--dry-run]    # what the Slurm job runs
    python deploy/skipjack_3b_600m/handoff_waiter.py status              # print the persistent state

Each tick, in this order:
  1. another waiter queued -> exit (one chain); <reports>/handoff/STOP exists -> exit without a successor;
  2. queue the successor first (--begin=now+INTERVAL, --dependency=singleton), so a crash in this tick never ends the chain;
  3. smoke already submitted -> follow that job: still live -> done; ended -> record smoke_finished.json, cancel the
     successor, stop. A failed smoke is never resubmitted here (the owner decides);
  4. otherwise handoff_gate.check(require_approval, scope=smoke). It must pass on CONFIRM consecutive ticks (>= one
     INTERVAL apart), so a momentarily empty queue cannot trigger it. Then capped_submit_3b.py smoke, which re-checks the
     gate and the cap and requires tokenization verified + the LR resume regression test passing; a refusal is logged
     and the waiter keeps waiting.
State: <reports>/handoff/waiter_state.json (every tick, appended), last_check.json, smoke_submitted.json, smoke_finished.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff_gate  # noqa: E402
import registry_3b as R  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = 'rw3b_handoff_waiter'
INTERVAL = '2hours'
CONFIRM = 2
H = R.REPORTS / 'handoff'
STATE = H / 'waiter_state.json'
SUBMITTED = H / 'smoke_submitted.json'
FINISHED = H / 'smoke_finished.json'
LOG = R.LOG_DIR / f'{NAME}.log'
LIVE = {'PENDING', 'RUNNING', 'REQUEUED', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUE_HOLD', 'REQUEUE_FED'}


def sh(argv, dry=False):
    if dry and argv[0] in ('sbatch', 'scancel'):
        print('[dry-run]', ' '.join(map(str, argv)))
        return subprocess.CompletedProcess(argv, 0, '0\n', '')
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True)


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {'consecutive_passes': 0, 'ticks': []}


def save(st: dict) -> None:
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def job_state(jid: str) -> str:
    q = sh(['squeue', '-h', '-j', jid, '-o', '%T']).stdout.strip()
    if q:
        return q.split()[0]
    a = sh(['sacct', '-n', '-X', '-P', '-j', jid, '-o', 'State']).stdout.strip()
    return a.split()[0] if a else 'UNKNOWN'


def tick(dry: bool) -> int:
    H.mkdir(parents=True, exist_ok=True)
    R.LOG_DIR.mkdir(parents=True, exist_ok=True)
    me = os.environ.get('SLURM_JOB_ID', '')
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    others = [j for j in sh(['squeue', '-h', '-u', os.environ['USER'], '-n', NAME, '-o', '%i']).stdout.split() if j != me]
    if others:
        print(f'[waiter] {now} another waiter is queued ({others}); exiting without a successor')
        return 0
    if (H / 'STOP').exists():
        print(f'[waiter] {now} {H}/STOP present; chain stopped')
        return 0
    st = load()
    rec = {'at': now, 'job': me, 'head': sh(['git', '-C', R.REPO, 'rev-parse', 'HEAD']).stdout.strip()}
    nxt = sh(['sbatch', '--parsable', f'--begin=now+{INTERVAL}', '--dependency=singleton', f'--output={LOG}',
              HERE / 'handoff_waiter.sbatch'], dry)
    rec['successor'] = nxt.stdout.strip().split(';')[0] if nxt.returncode == 0 else f'SBATCH FAILED: {nxt.stderr.strip()}'

    if SUBMITTED.is_file():
        sub = json.loads(SUBMITTED.read_text())
        state = job_state(sub['job'])
        rec['smoke'] = {'job': sub['job'], 'state': state}
        if state not in LIVE:
            report = Path(sub['workdir']) / 'smoke_report.json'
            fin = {'job': sub['job'], 'state': state, 'workdir': sub['workdir'], 'checked_at': now,
                   'smoke_report': str(report) if report.is_file() else None,
                   'problems': json.loads(report.read_text())['problems'] if report.is_file() else ['no smoke_report.json']}
            FINISHED.write_text(json.dumps(fin, indent=1))
            if nxt.returncode == 0:
                sh(['scancel', rec['successor']], dry)
            rec['action'] = f'smoke ended {state}; recorded {FINISHED}; chain stopped (full training is never submitted here)'
    else:
        ev = handoff_gate.check(require_approval=True, scope='smoke')
        (H / 'last_check.json').write_text(json.dumps(ev, indent=1))
        st['consecutive_passes'] = st['consecutive_passes'] + 1 if ev['handoff'] else 0
        rec['gate'] = {'handoff': ev['handoff'], 'missing': ev['missing'], 'consecutive_passes': st['consecutive_passes']}
        if ev['handoff'] and st['consecutive_passes'] >= CONFIRM:
            if dry:
                rec['action'] = '[dry-run] would run capped_submit_3b.py smoke'
            else:
                r = sh([sys.executable, HERE / 'capped_submit_3b.py', 'smoke'])
                m = re.search(r'submitted smoke (\d+); workdir (\S+)', r.stdout)
                if r.returncode == 0 and m:
                    SUBMITTED.write_text(json.dumps({'job': m.group(1), 'workdir': m.group(2), 'submitted_at': now,
                                                     'by': f'{NAME} {me}', 'gate': ev}, indent=1))
                    rec['action'] = f'smoke submitted: job {m.group(1)}'
                else:
                    rec['action'] = f'smoke refused/failed (keep waiting): {(r.stdout + r.stderr).strip()[-800:]}'
        else:
            rec['action'] = 'held' if not ev['handoff'] else f'gate passed {st["consecutive_passes"]}/{CONFIRM}; confirming next tick'
    st['ticks'].append(rec)
    st['last'] = rec
    if not dry:
        save(st)
    print(json.dumps(rec, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['tick', 'status'])
    ap.add_argument('--dry-run', action='store_true', help='no sbatch/scancel, no smoke, no state write')
    args = ap.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(load().get('last'), indent=1))
        for f in (SUBMITTED, FINISHED):
            print(f, f.read_text() if f.is_file() else '(absent)')
        return 0
    return tick(args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
