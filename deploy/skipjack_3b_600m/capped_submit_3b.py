#!/usr/bin/env python
"""20-GPU-capped, restart-safe submitter for the 3B rewrite-600m grid (27 segments, seed 42) — the only way to
put this grid's GPU work on the queue.

    P=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python
    $P deploy/skipjack_3b_600m/capped_submit_3b.py plan      --s-per-it 31     # lane table, makespan, peak (pure)
    $P deploy/skipjack_3b_600m/capped_submit_3b.py test-only --s-per-it 31     # sbatch --test-only x 27 (queues nothing)
    $P deploy/skipjack_3b_600m/capped_submit_3b.py usage                       # project (kys_*) GPUs right now
    $P deploy/skipjack_3b_600m/capped_submit_3b.py smoke                       # gated: the 4-GPU smoke job
    $P deploy/skipjack_3b_600m/capped_submit_3b.py guard --need-gpus 1         # gated: before any 3B eval job
    $P deploy/skipjack_3b_600m/capped_submit_3b.py submit    --s-per-it <smoke> # gated: the grid (idempotent)
    $P deploy/skipjack_3b_600m/capped_submit_3b.py status

Mechanism: the reconcile / lane / lock / state / duplicate-adoption logic is deploy/skipjack_1p5b/capped_submit.py's,
imported unchanged from THIS worktree's copy (bc131392, the revision the baselines run); only the grid-specific parts
are swapped here: the 27-segment graph (registry_3b), the segment script (segment_3b.sbatch), trunk seeding from the
3B init, and the gates below. Five serial `afterok` lanes x 4 GPUs = at most 20 GPUs, enforced by Slurm.

Gates (all re-evaluated on every smoke / guard / submit call, so nothing armed today can fire early):
  * handoff_gate.check(require_approval=True): every baseline seed has SEED_COMPLETE, all 63 baseline finals are on disk,
    no non-3B kys_ job is live, and the owner wrote LAUNCH_APPROVED;
  * the cap: kys_* GPUs held + waiting + requested <= 20 (both grids carry the kys_ prefix, so both tools see all of it);
  * smoke additionally (smoke_prerequisites): all three corpora tokenized and verified (assert_invariants_3b --all
    --corpus, epoch_stats.json present) and the CPU LR resume regression test passes; the smoke gate accepts the owner's
    SMOKE_AUTHORIZED in place of LAUNCH_APPROVED (smoke only);
  * submit additionally: SMOKE_ACCEPTED (owner-written JSON naming the layout and measured s/it) matches the rendered
    configs, assert_invariants_3b --all --corpus passes, and the 3B init matches its manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
import capped_submit as cs  # noqa: E402  (the 1.5B tool, unchanged; its logic is reused, its grid is not)
import handoff_gate  # noqa: E402
import registry_3b as R  # noqa: E402

CFG_DIR = R.REPO / 'configs' / '3B-rewrite-600m' / f'seed{R.SEED}'
SEGMENT_SBATCH = HERE / 'segment_3b.sbatch'
SMOKE_SBATCH = HERE / 'smoke_3b.sbatch'
STATE_DIR = R.STATE_ROOT / f'seed{R.SEED}'
SMOKE_ACCEPTED = R.REPORTS / 'SMOKE_ACCEPTED'
PYTEST_PATH = R.TRAIN_ROOT / 'tools' / 'pytest_pkgs'   # pytest is not in the training env; installed --target here
INIT = R.INIT_ROOT / f'_init_3B_seed{R.SEED}' / '0'
INIT_MANIFEST = R.INIT_ROOT / f'_init_3B_seed{R.SEED}' / 'init_manifest.json'

# grid-specific globals of the reused module
cs.DEPENDS = R.DEPENDS            # check_graph: science edges
cs.SBATCH = SEGMENT_SBATCH        # submit / test-only: the job script
assert cs.PROJECT_PREFIX == 'kys_' and cs.GPU_CEILING == R.GPU_CEILING == 20 and cs.GPUS_PER_SEGMENT == R.GPUS_PER_SEGMENT
assert Path(cs.REPO) == R.REPO, f'capped_submit imported from {cs.REPO}, not the 3B worktree'


def load_segments() -> dict[str, cs.Segment]:
    segs = {}
    for s in R.SETTINGS:
        for k in R.KINDS:
            name = R.seg_name(s, k)
            cfg = CFG_DIR / f'{name}.yaml'
            c = yaml.safe_load(cfg.read_text())
            assert c['tokens']['train_steps'] == R.END_STEP[k]
            segs[name] = cs.Segment(name=name, setting=s, kind=k, cfg=cfg, ckpt=Path(c['checkpoints']['checkpoints_path']),
                                    final_step=R.END_STEP[k], steps=R.END_STEP[k] - R.FIRST_STEP[k],
                                    science_dep=R.seg_name(s, R.DEPENDS[k]) if R.DEPENDS[k] else None)
            assert segs[name].job_name.startswith(R.JOB_PREFIX)
    return segs


def init_digest(init: Path) -> dict:
    files = sorted(p for p in init.rglob('*') if p.is_file())
    h = hashlib.sha256()
    for p in files:
        if p.parts[-2:-1] != ('lr_scheduler',) and 'optimizer' not in p.parts:   # weights + config only
            h.update(p.relative_to(init).as_posix().encode())
            with open(p, 'rb') as f:
                while chunk := f.read(16 << 20):
                    h.update(chunk)
    return {'files': len(files), 'weights_sha256': h.hexdigest()}


def seed_trunk_3b(c: dict, seed: int, s: cs.Segment) -> bool:
    """Shared init -> <trunk>/0 + latest.txt (only if latest.txt is absent): identical weights for all three settings."""
    trunk = s.ckpt
    if (trunk / 'latest.txt').is_file():
        return False
    if trunk.exists() and any(p.is_dir() and p.name.isdigit() for p in trunk.iterdir()):
        sys.exit(f'{trunk} holds step directories but no latest.txt; refusing to seed over a progressed trunk')
    want = json.loads(INIT_MANIFEST.read_text())
    if init_digest(INIT) != {'files': want['files'], 'weights_sha256': want['weights_sha256']}:
        sys.exit(f'{INIT} does not match {INIT_MANIFEST}; refusing to seed')
    trunk.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-a', str(INIT), str(trunk / '0')], check=True)
    if init_digest(trunk / '0')['weights_sha256'] != want['weights_sha256']:
        sys.exit(f'{trunk}/0 differs from the init after copying')
    (trunk / 'latest.txt').write_text('0\n')
    return True


cs.seed_trunk = seed_trunk_3b


def validate_time(s_per_it: float):
    have = cs.slurm_seconds(R.SITE['slurm']['time'])
    need = max(R.END_STEP[k] - R.FIRST_STEP[k] for k in R.KINDS) * s_per_it
    if have < 1.1 * need + 1800:
        sys.exit(f'time {R.SITE["slurm"]["time"]} too short at {s_per_it} s/it (longest segment needs '
                 f'{(1.1 * need + 1800) / 3600:.1f} h); split the trunk further in registry_3b.TRUNK_ENDS')


def gate(need_gpus: int, slurm, cap: bool = True, scope: str = 'launch') -> None:
    """Handoff + owner approval; with cap, also kys_* GPUs held + waiting + need <= 20 (smoke, eval). The grid itself
    is capped by its lanes instead (its own pending lane jobs would otherwise count as 'waiting' on every re-run)."""
    ev = handoff_gate.check(require_approval=True, scope=scope)
    if not ev['handoff']:
        sys.exit('HELD — baseline handoff not reached:\n  ' + '\n  '.join(ev['missing']))
    if not cap:
        return
    held, waiting = cs.project_gpus(slurm.queue())
    if held + waiting + need_gpus > R.GPU_CEILING:
        sys.exit(f'cap: {held} held + {waiting} waiting + {need_gpus} new > {R.GPU_CEILING}; refusing')


def smoke_prerequisites() -> None:
    """Tokenized corpora verified, and the LR resume fix still demonstrated by its CPU regression test."""
    missing = [s for s in R.SETTINGS if not (R.REPORTS / 'tokenize' / s / 'epoch_stats.json').is_file()]
    if missing:
        sys.exit(f'tokenization not finished/verified for {missing} (no reports/tokenize/<s>/epoch_stats.json)')
    r = subprocess.run([sys.executable, HERE / 'assert_invariants_3b.py', '--all', '--corpus'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'assert_invariants_3b --all --corpus failed:\n{r.stdout[-1500:]}')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore',
                        str(HERE / 'test_lr_resume_3b.py')], capture_output=True, text=True,
                       env={**__import__('os').environ, 'PYTHONPATH': str(PYTEST_PATH)})
    if r.returncode:
        sys.exit(f'LR resume regression test failed:\n{r.stdout[-1500:]}')


def smoke_acceptance() -> dict:
    if not SMOKE_ACCEPTED.is_file():
        sys.exit(f'no {SMOKE_ACCEPTED}: the owner writes it after reviewing the smoke report '
                 '(JSON: {"layout": "A"|"B", "s_per_it": <measured>, "smoke_job": <id>, "accepted_by": ..., "date": ...})')
    acc = json.loads(SMOKE_ACCEPTED.read_text())
    envs = {ln.split('=', 1)[1] for p in CFG_DIR.glob('*.env') for ln in p.read_text().splitlines()
            if ln.startswith('export KYS3B_LAYOUT=')}
    if envs != {acc['layout']}:
        sys.exit(f'SMOKE_ACCEPTED layout {acc["layout"]} but configs are rendered with {sorted(envs)}; re-render')
    return acc


def main(argv=None, slurm=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['plan', 'test-only', 'usage', 'smoke', 'guard', 'submit', 'status'])
    ap.add_argument('--lanes', type=int, default=R.GPU_CEILING // R.GPUS_PER_SEGMENT)
    ap.add_argument('--s-per-it', type=float)
    ap.add_argument('--need-gpus', type=int, default=0)
    ap.add_argument('--retry', default='')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    slurm = slurm or cs.Slurm()
    if args.lanes * R.GPUS_PER_SEGMENT > R.GPU_CEILING:
        sys.exit(f'{args.lanes} lanes exceed the {R.GPU_CEILING}-GPU ceiling')

    if args.command == 'usage':
        return cs.main(['usage'], slurm=slurm)
    if args.command == 'guard':
        gate(args.need_gpus, slurm)
        print(f'guard: handoff reached and OK to add {args.need_gpus} GPUs (name the job {R.JOB_PREFIX}<what>)')
        return 0
    if args.command == 'smoke':
        gate(R.GPUS_PER_SEGMENT, slurm, scope='smoke')
        smoke_prerequisites()
        if any(j['name'].startswith(R.JOB_PREFIX) for j in slurm.queue()):
            sys.exit('a kys_3b_ job is already live; one smoke at a time, never alongside the grid')
        work = R.REPORTS / 'smoke' / time.strftime('%Y%m%d_%H%M%S')
        R.LOG_DIR.mkdir(parents=True, exist_ok=True)
        sl = R.SITE['slurm']
        jid = slurm.sbatch([f'--job-name={R.JOB_PREFIX}smoke', f'--partition={sl["partition"]}', f'--account={sl["account"]}',
                            f'--qos={sl["qos"]}', f'--gres={sl["gres"]}', '--nodes=1', f'--cpus-per-task={sl["cpus_per_task"]}',
                            '--time=04:00:00', f'--output={R.LOG_DIR}/%x.%j.log',
                            f'--export=ALL,KYS_REPO={R.REPO}', str(SMOKE_SBATCH), str(work)])
        print(f'submitted smoke {jid}; workdir {work}')
        return 0

    segs = load_segments()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with cs.Lock(STATE_DIR / 'submit.lock'):
        st = cs.State(STATE_DIR / 'state.json')
        if st.data['plan']:
            p = st.data['plan']
            if p['segments'].keys() != segs.keys() or p['lanes'] != args.lanes:
                sys.exit(f'{STATE_DIR}/state.json holds a plan for other segments/lanes; refusing to re-plan a live grid')
            for n, s in segs.items():
                for k in ('lane', 'order', 'lane_prev', 'est_start_h', 'est_end_h'):
                    setattr(s, k, p['segments'][n][k])
            makespan = max(s.est_end_h for s in segs.values())
        else:
            if args.command != 'status' and not args.s_per_it:
                sys.exit('--s-per-it is required')
            makespan = cs.assign_lanes(segs, args.lanes, args.s_per_it or 31.0)
        cs.check_graph(segs, args.lanes)
        if cs.max_concurrency(segs) > args.lanes:
            sys.exit('simulated concurrency exceeds the lane count')

        if args.command == 'plan':
            validate_time(args.s_per_it)
            cs.print_plan(segs, args.lanes, makespan)
            (STATE_DIR / 'plan_preview.json').write_text(json.dumps(
                {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev, 'science_dep': s.science_dep,
                     'steps': s.steps, 'est_start_h': round(s.est_start_h, 2), 'est_end_h': round(s.est_end_h, 2)}
                 for n, s in segs.items()}, indent=1))
            return 0
        if args.command == 'status':
            status = cs.reconcile(segs, st, slurm, slurm.queue())
            st.save()
            cs.print_status(segs, status)
            return 0
        if args.command == 'test-only':
            validate_time(args.s_per_it)
            bad = 0
            for s in sorted(segs.values(), key=lambda s: (s.lane, s.order)):
                rc, msg = slurm.test_only([*cs.sbatch_opts(R.SITE, s, R.LOG_DIR), str(SEGMENT_SBATCH), str(s.cfg)])
                bad += rc != 0
                print(f'  {"OK  " if rc == 0 else "FAIL"} lane {s.lane} {s.name}: {msg}')
            print(f'sbatch --test-only: {len(segs) - bad}/{len(segs)} accepted (dependencies omitted; nothing queued)')
            return 1 if bad else 0

        # submit
        acc = smoke_acceptance()
        if args.s_per_it != acc['s_per_it']:
            sys.exit(f'--s-per-it {args.s_per_it} != SMOKE_ACCEPTED s_per_it {acc["s_per_it"]}')
        validate_time(args.s_per_it)
        if not args.dry_run:
            gate(0, slurm, cap=False)   # cs.submit refuses while any non-3B kys_ job is live; lanes bound the grid at 20
        r = subprocess.run([sys.executable, HERE / 'assert_invariants_3b.py', '--all', '--corpus'], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'assert_invariants_3b --all --corpus failed:\n{r.stdout[-1500:]}')
        R.LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not st.data['plan']:
            st.data['plan'] = {'lanes': args.lanes, 'seed': R.SEED, 's_per_it': args.s_per_it, 'layout': acc['layout'],
                               'repo_head': subprocess.run(['git', '-C', R.REPO, 'rev-parse', 'HEAD'], capture_output=True,
                                                           text=True).stdout.strip(),
                               'segments': {n: {'lane': s.lane, 'order': s.order, 'lane_prev': s.lane_prev,
                                                'science_dep': s.science_dep, 'est_start_h': s.est_start_h,
                                                'est_end_h': s.est_end_h} for n, s in segs.items()}}
            if not args.dry_run:
                st.save()
        retry = {x for x in args.retry.split(',') if x}
        cs.submit(segs, st, slurm, R.SITE, R.SEED, R.LOG_DIR, retry, dry=args.dry_run)
        if not args.dry_run:
            st.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
