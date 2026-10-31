#!/usr/bin/env python
"""SKIPJACK-ONLY CLI for the startup validation, the shared init and trunk1 continuation (startup_3b.sbatch,
init_3b.sh, segment_3b.sbatch). The logic is deploy/portable_3b/startup.py; every derive/summarize command takes the
production config EXPLICITLY (no committed config is read implicitly), so a config rendered for another site yields
init/startup configs for that site. Other sites: deploy/portable_3b/kys3b.py (same functions).

    python deploy/skipjack_3b_600m/startup_3b.py derive      --config <trunk1.yaml> --workdir <fresh dir>
    python deploy/skipjack_3b_600m/startup_3b.py derive-init --config <trunk1.yaml> --out <init.yaml> --ckpt <scratch ckpt>
    python deploy/skipjack_3b_600m/startup_3b.py summarize   --config <trunk1.yaml> --workdir <dir> [--mem-budget-gib 74.5]
    python deploy/skipjack_3b_600m/startup_3b.py seed-trunk  --config <any segment.yaml whose folder starts from the init>
    python deploy/skipjack_3b_600m/startup_3b.py resume-config --config <trunk1.yaml> --out <yaml>
    python deploy/skipjack_3b_600m/startup_3b.py check-log   --config <seg.yaml> --log <seg log> [--json]

Phases (startup_3b.sbatch, one 4 x H100 node, all on quality-first's real data and real trunk folder):
  init        only if the shared init does not exist yet (init_3b.sh, under a lock shared with segment_3b.sbatch)
  s_trunk     the rendered production trunk1 config with train_steps 20: writes trunk/20 (real progress)
  s_resume    the same config with train_steps 30 and optimizer + LR state loaded: resumes trunk/20, writes trunk/30
  s_straight  30 uninterrupted steps from the init in a scratch folder (nothing saved): reference for the resume
The production trunk1 job then resumes trunk/30 (segment_3b.sbatch loads optimizer + LR state whenever latest > 0).
Checks: startup.summarize (finite loss/grad_norm, lr == scheduler, tokens == step x 2,097,152, resume vs
uninterrupted, trunk/20 + trunk/30, latest.txt 30, peak GPU memory <= --mem-budget-gib when given).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry_3b as R  # noqa: E402  (puts deploy/portable_3b on sys.path)
import startup as P  # noqa: E402

CFG_DIR = R.REPO / 'configs' / '3B-rewrite-600m' / f'seed{R.SEED}'
QF_TRUNK1 = CFG_DIR / f'{R.seg_name("quality-first", "trunk1")}.yaml'   # what the Skipjack scripts pass as --config
INIT_PARENT = R.INIT_ROOT / f'_init_3B_seed{R.SEED}'
PHASES = P.STARTUP_PHASES
SKIPJACK_MEM_BUDGET_GIB = 74.5     # 4 x H100 80GB (startup_3b.sbatch passes it)
parse_log = P.parse_log


def step_problems(steps: list[dict], sched_cfg: Path, label: str) -> list[str]:
    return P.step_problems(steps, P.load(sched_cfg), label)


def seed_trunk(cfg: Path) -> str:
    """Seed <checkpoints_path>/0 from the shared Skipjack init (digest-checked) if it has no latest.txt; never over progress."""
    import capped_submit_3b as C
    import initlib
    trunk = Path(P.load(cfg)['checkpoints']['checkpoints_path'])
    if not C.INIT_MANIFEST.is_file():
        sys.exit(f'no {C.INIT_MANIFEST}: create the init first (init_3b.sh)')
    return 'seeded' if initlib.seed_trunk(trunk, C.INIT, C.INIT_MANIFEST) else 'already seeded'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['derive', 'derive-init', 'summarize', 'seed-trunk', 'resume-config', 'check-log'])
    ap.add_argument('--config', type=Path, required=True, help='the production config the command reads or derives from')
    ap.add_argument('--workdir', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--log', type=Path)
    ap.add_argument('--mem-budget-gib', type=float)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    need = {'derive': ['workdir'], 'derive-init': ['out', 'ckpt'], 'summarize': ['workdir'], 'resume-config': ['out'],
            'check-log': ['log']}.get(args.command, [])
    missing = [f'--{n.replace("_", "-")}' for n in need if getattr(args, n) is None]
    if missing:
        ap.error(f'{args.command} needs {" ".join(missing)}')
    if args.command == 'derive':
        P.derive_startup(args.config, args.workdir)
        print(f'wrote {PHASES} to {args.workdir}/cfg from {args.config}')
    elif args.command == 'derive-init':
        print(json.dumps(P.derive_init(args.config, args.out, args.ckpt)))
    elif args.command == 'summarize':
        rep = P.summarize(args.config, args.workdir, args.mem_budget_gib)
        brief = {k: v for k, v in rep.items() if k != 'phases'}
        brief['phases'] = {ph: {k: v for k, v in p.items() if k != 'steps'} for ph, p in rep['phases'].items()}
        print(json.dumps(brief, indent=1))
        print('STARTUP OK' if rep['ok'] else f'STARTUP PROBLEMS: {len(rep["problems"])}')
        return 0 if rep['ok'] else 1
    elif args.command == 'seed-trunk':
        print(seed_trunk(args.config))
    elif args.command == 'resume-config':
        P.resume_config(args.config, args.out)
    elif args.command == 'check-log':
        r = P.check_log(args.config, args.log)
        print(json.dumps(r) if args.json else json.dumps(r, indent=1))
        return 1 if r['problems'] else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
