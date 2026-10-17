#!/usr/bin/env python
"""Read-only gate: has the 1.5B raw-baseline workflow released its 20-H100 budget for good?

    python deploy/skipjack_3b_600m/handoff_gate.py check [--json out.json]     # exit 0 = handoff reached
    python deploy/skipjack_3b_600m/handoff_gate.py check --require-approval     # also needs the owner's LAUNCH_APPROVED

The baseline workflow (deploy/skipjack_1p5b/, active checkout ../nanotron-kys) runs seeds one at a time; a seed is
finished only when its operator writes <1.5B state root>/seed<S>/SEED_COMPLETE after its 21 finals are verified,
uploaded and evaluated and its GPU jobs have ended (capped_submit.py refuses the next seed without it). An empty queue,
or one finished seed, is therefore NOT the handoff. The gate requires ALL of:

  1. the seed list: every seed that deploy/clusters.yaml `baseline_seed_assignment` assigns to skipjack_1p5b_baselines
     (read from the ACTIVE checkout, so a seed added later is picked up), and at least {42, 43, 44};
  2. seed<S>/SEED_COMPLETE exists for every one of those seeds;
  3. seed<S>/state.json exists and every one of its 21 branch finals (ep1-3 x 7 settings) has model_config.json on disk;
  4. no live job of this user whose name starts with `kys_` other than this grid's `kys_3b_` jobs (training retries,
     smoke tests and evaluation all carry the prefix; deploy/skipjack_1p5b/README.md);
  5. with --require-approval: <3B reports>/LAUNCH_APPROVED exists (written by the owner, never by a tool).

Nothing here writes to the baseline's files, submits, or cancels. Missing evidence is reported, never assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry_3b as R  # noqa: E402

BASELINE_REPO = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys')            # read only
BASELINE_PROFILE = BASELINE_REPO / 'deploy' / 'clusters.yaml'
BASELINE_CLUSTER = 'skipjack_1p5b_baselines'
BASELINE_STATE_ROOT = Path('/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/reports/capped_submit')
MIN_SEEDS = {42, 43, 44}
APPROVAL = R.REPORTS / 'LAUNCH_APPROVED'


def live_jobs() -> list[dict]:
    out = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j|%T'],
                         capture_output=True, text=True, check=True).stdout
    return [dict(zip(('id', 'name', 'state'), ln.split('|'))) for ln in out.splitlines() if ln.count('|') == 2]


def check(require_approval: bool) -> dict:
    ev = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'missing': [], 'seeds': {}}
    prof = yaml.safe_load(BASELINE_PROFILE.read_text())
    seeds = sorted(s for s, c in (prof.get('baseline_seed_assignment') or {}).items() if c == BASELINE_CLUSTER)
    ev['baseline_seeds'] = seeds
    if not MIN_SEEDS <= set(seeds):
        ev['missing'].append(f'baseline_seed_assignment lists {seeds}, expected at least {sorted(MIN_SEEDS)}')
    for s in seeds:
        d = BASELINE_STATE_ROOT / f'seed{s}'
        rec = {'SEED_COMPLETE': (d / 'SEED_COMPLETE').is_file(), 'state_json': (d / 'state.json').is_file()}
        if not rec['SEED_COMPLETE']:
            ev['missing'].append(f'seed {s}: no {d}/SEED_COMPLETE')
        if not rec['state_json']:
            ev['missing'].append(f'seed {s}: no {d}/state.json (seed never submitted)')
        finals, have = 0, 0
        for cfg in sorted((BASELINE_REPO / 'configs' / f'1.5B-baseline-seed{s}' / 'filled').glob('*_ep[123].yaml')):
            c = yaml.safe_load(cfg.read_text())
            finals += 1
            have += (Path(c['checkpoints']['checkpoints_path']) / str(c['tokens']['train_steps']) / 'model_config.json').is_file()
        rec['finals_on_disk'] = f'{have}/{finals}'
        if finals != 21 or have != finals:
            ev['missing'].append(f'seed {s}: {have}/{finals} branch finals on disk (expected 21/21)')
        ev['seeds'][s] = rec
    jobs = live_jobs()
    foreign = [j for j in jobs if j['name'].startswith('kys_') and not j['name'].startswith(R.JOB_PREFIX)]
    ev['live_baseline_jobs'] = len(foreign)
    if foreign:
        ev['missing'].append(f'{len(foreign)} live kys_ baseline job(s), e.g. {foreign[0]["id"]} {foreign[0]["name"]} {foreign[0]["state"]}')
    ev['approval'] = APPROVAL.is_file()
    if require_approval and not ev['approval']:
        ev['missing'].append(f'no owner approval marker {APPROVAL}')
    ev['handoff'] = not ev['missing']
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['check'])
    ap.add_argument('--require-approval', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    ev = check(args.require_approval)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))
    print('HANDOFF REACHED' if ev['handoff'] else f'HELD: {len(ev["missing"])} condition(s) unmet')
    return 0 if ev['handoff'] else 3


if __name__ == '__main__':
    sys.exit(main())
