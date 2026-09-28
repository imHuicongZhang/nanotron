#!/usr/bin/env python
"""Write (and optionally run) the SLURM submission script for one seed of the raw baselines.

    python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 [--s-per-it 16.5] [--submit]

Reads the filled configs (7 settings x 6 segments = 42 per seed; tools/kys_raw/registry.py) in configs/1.5B-baseline-seed<S>/filled/ (tools/kys_raw/fill_placeholders.py)
and the cluster entry in deploy/clusters.yaml, re-runs tools/assert_invariants.py on each (placeholders,
batch, environment are checked; corpus and resume are checked by the launcher before every segment),
and writes configs/1.5B-baseline-seed<S>/filled/submit_seed<S>.sh containing:

  1. trunk seeding per setting: <ckpt_root>/seed<S>/<setting>/trunk/<setting>/seed<S>/0 <- the seed's
     init checkpoint, latest.txt = 0 (skipped when latest.txt already exists);
  2. a duplicate-submission guard: before ANY sbatch, the script refuses to run if a job with one of its
     job names (kys_<setting>_seed<S>_<segment>) is already pending or running for $USER (squeue);
  3. one sbatch call per segment with the dependency chain, each job id captured for its dependents:
        trunk1 -> trunk2 -> trunk3          (afterok)
        trunk1 -> ep1, trunk2 -> ep2, trunk3 -> ep3

Every job runs deploy/slurm/kys_segment.sbatch, which is requeue-safe: a restarted trunk resumes from
its latest checkpoint through latest.txt, a restarted branch re-runs from its branch point, and a
segment whose final checkpoint exists exits 0. afterok holds across a requeue (same job id).

Nothing is submitted unless --submit is given; by default the script is only written and printed.
--test-only additionally runs `sbatch --test-only` for every segment (dependencies omitted, since the
upstream job ids do not exist yet): SLURM validates partition/account/qos/gres/time without queueing anything.

What the guard does NOT cover: jobs under another user or job name, or a second cluster. A completed-segment
skip (the sbatch's final-checkpoint check) only protects a segment that has finished; two concurrent copies of
an unfinished trunk would write into the same checkpoints_path. Do not resubmit while jobs are queued.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DEPENDS, FIRST_STEP, KINDS, SETTING_NAMES, parse_settings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SBATCH = REPO / 'deploy' / 'slurm' / 'kys_segment.sbatch'


def slurm_seconds(t: str) -> int:
    """SLURM --time -> seconds. Formats (sbatch(1)): M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    t = str(t).strip()
    if '-' in t:
        d, rest = t.split('-', 1)
        p = [int(x) for x in rest.split(':')] + [0, 0]
        return int(d) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]
    p = [int(x) for x in t.split(':')]
    return {1: lambda: p[0] * 60, 2: lambda: p[0] * 60 + p[1], 3: lambda: p[0] * 3600 + p[1] * 60 + p[2]}[len(p)]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', default='marc-cluster')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--configs', type=Path, help='default: configs/1.5B-baseline-seed<S>/filled')
    ap.add_argument('--settings', default=','.join(SETTING_NAMES), help='comma-separated subset of registry.py')
    ap.add_argument('--s-per-it', type=float, help='measured seconds per optimizer step, for the wall-time table')
    ap.add_argument('--expected-mbs', type=int, help='forwarded to assert_invariants.py if mbs != 32')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--test-only', action='store_true',
                    help='run `sbatch --test-only` per segment (no dependency, nothing queued) and report')
    ap.add_argument('--profile', type=Path, default=REPO / 'deploy' / 'clusters.yaml')
    args = ap.parse_args()
    settings = parse_settings(args.settings)
    if args.submit and args.test_only:
        sys.exit('--submit and --test-only are exclusive')

    prof = yaml.safe_load(args.profile.read_text())
    c = prof['clusters'][args.cluster]
    sl = c.get('slurm') or {}
    for k in ('partition', 'gres', 'cpus_per_task', 'time'):
        if sl.get(k) is None:
            sys.exit(f'{args.cluster}.slurm.{k} is null in deploy/clusters.yaml')
    # A job must be able to make progress before the wall limit: a trunk only persists work at its 1500-step
    # checkpoint interval (or its segment end), and a branch writes nothing until its final step. With a
    # shorter limit the job requeues forever at the same step and never completes.
    if args.s_per_it:
        need = max(1500, 14305 - FIRST_STEP['ep3']) * args.s_per_it
        have = slurm_seconds(sl['time'])
        if have < 1.1 * need + 1800:
            sys.exit(f'{args.cluster}.slurm.time = {sl["time"]} ({have / 3600:.1f} h) is too short at {args.s_per_it} s/it: the '
                     f'longest stretch without a checkpoint (max of the 1500-step trunk interval and ep3\'s 1430 steps) '
                     f'takes {need / 3600:.1f} h; set at least {(1.1 * need + 1800) / 3600:.1f} h')
    cfg_dir = args.configs or REPO / 'configs' / f'1.5B-baseline-seed{args.seed}' / 'filled'
    init_root = c.get('init_root')
    log_dir = Path((c.get('wandb') or {}).get('dir') or cfg_dir).parent / 'slurm_logs'
    # The generated script runs SBATCH — this repo's copy, resolved from __file__ — while exporting
    # KYS_REPO=repo_dir for the job to use. Normally the same clone, because you generate on the
    # cluster you run on. When they differ, every job launches one clone's kys_segment.sbatch against
    # another clone's code, which is silent: SLURM accepts it, the jobs start, and the two trees can
    # be at different commits. Refuse instead of generating a script that mixes them.
    if c.get('repo_dir') and Path(c['repo_dir']).resolve() != REPO:
        sys.exit(f'{args.cluster}.repo_dir is {c["repo_dir"]}, but this plan_submit.py lives in {REPO}.\n'
                 f'The generated script would run {SBATCH} while exporting KYS_REPO={c["repo_dir"]}.\n'
                 f'Generate from the clone at repo_dir, or set repo_dir to {REPO}.')
    exports = {'KYS_REPO': c.get('repo_dir'), 'KYS_ENV_ACTIVATE': c.get('env_activate'),
               'KYS_PYTHON_INCLUDE': c.get('python_include')}
    export_arg = ','.join(['ALL'] + [f'{k}={v}' for k, v in exports.items() if v])

    names = [f'kys_{s}_seed{args.seed}_{k}' for s in settings for k in KINDS]
    lines = ['#!/usr/bin/env bash', f'# Generated by tools/kys_raw/plan_submit.py --cluster {args.cluster} --seed {args.seed} '
             f'--settings {",".join(settings)}',
             'set -euo pipefail', f'mkdir -p {shlex.quote(str(log_dir))}', '',
             '# Refuse to submit anything if one of these jobs is already pending or running (see plan_submit.py).',
             'queued=$(squeue -h -u "$USER" -o %j)',
             f'for n in {" ".join(names)}; do',
             '  if grep -qx "$n" <<<"$queued"; then echo "$n is already queued or running; refusing to submit duplicates"; exit 1; fi',
             'done', '']
    rows, tests = [], []
    for setting in settings:
        trunk_cfg = yaml.safe_load((cfg_dir / f'{setting}_seed{args.seed}_trunk1.yaml').read_text())
        trunk = Path(trunk_cfg['checkpoints']['checkpoints_path'])
        init = Path(init_root or '<init_root>') / f'_init_1.5B_seed{args.seed}' / '0'
        # The init hash check (tools/hash_init_checkpoint.py) covers model weights only, so also refuse to
        # seed from an incomplete download (180 files). trunk1 loads the init as weights only
        # (load_optimizer/load_lr_scheduler false), but a partial copy is a sign of a broken transfer.
        lines += [f'# ---- {setting} seed {args.seed}',
                  f'I={shlex.quote(str(init))}',
                  f'T={shlex.quote(str(trunk))}',
                  'if [[ ! -f "$T/latest.txt" ]]; then',
                  '  for f in model_config.json optimizer/optimizer_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt '
                  'lr_scheduler/lr_scheduler_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt; do',
                  '    [[ -s "$I/$f" ]] || { echo "init $I is incomplete: missing $f"; exit 1; }',
                  '  done',
                  '  n=$(find "$I" -type f | wc -l); [[ $n -eq 180 ]] || { echo "init $I has $n files, expected 180"; exit 1; }',
                  '  mkdir -p "$T" && cp -a "$I" "$T/0" && echo 0 > "$T/latest.txt"',
                  'fi']
        for kind in KINDS:
            name = f'{setting}_seed{args.seed}_{kind}'
            cfg = cfg_dir / f'{name}.yaml'
            chk = [sys.executable, REPO / 'tools/assert_invariants.py', '--config', cfg, '--skip-corpus', '--skip-env']
            if args.expected_mbs:
                chk += ['--expected-mbs', str(args.expected_mbs)]
            r = subprocess.run(chk, capture_output=True, text=True)
            if r.returncode:
                sys.exit(f'{cfg} fails assert_invariants.py:\n{r.stdout[-600:]}{r.stderr[-600:]}')
            steps = yaml.safe_load(cfg.read_text())['tokens']['train_steps'] - FIRST_STEP[kind]
            var = name.upper().replace('-', '_')
            opts = [f'--job-name=kys_{name}', f'--partition={sl["partition"]}', f'--gres={sl["gres"]}',
                    f'--nodes={sl.get("nodes") or 1}', f'--cpus-per-task={sl["cpus_per_task"]}', f'--time={sl["time"]}',
                    f'--output={log_dir}/%x.%j.log', f'--export={export_arg}']
            opts += [f'--account={sl["account"]}'] if sl.get('account') else []
            opts += [f'--qos={sl["qos"]}'] if sl.get('qos') else []
            if args.test_only:
                r = subprocess.run(['sbatch', '--test-only', *opts, str(SBATCH), str(cfg)], capture_output=True, text=True)
                msg = ' '.join(ln for ln in (r.stdout + r.stderr).splitlines() if 'sbatch' in ln.lower() or 'error' in ln.lower())
                tests.append((name, r.returncode, msg.strip()))
            dep = DEPENDS[kind]
            if dep:
                opts.append(f'--dependency=afterok:${{{f"{setting}_seed{args.seed}_{dep}".upper()}}}')
            lines.append(f'{var}=$(sbatch --parsable {" ".join(shlex.quote(o) if "$" not in o else o for o in opts)} '
                         f'{shlex.quote(str(SBATCH))} {shlex.quote(str(cfg))} | cut -d";" -f1)')
            lines.append(f'echo "{name}: ${var}"')
            est = f'{steps * args.s_per_it / 3600:.1f}' if args.s_per_it else '-'
            rows.append((name, steps, est, dep or '-'))
        lines.append('')

    out = cfg_dir / f'submit_seed{args.seed}.sh'
    out.write_text('\n'.join(lines) + '\n')
    out.chmod(0o755)
    print(f'{"job":42s} {"steps":>6s} {"est h":>6s} {"after":>7s}')
    for name, steps, est, dep in rows:
        print(f'{name:42s} {steps:6d} {est:>6s} {dep:>7s}')
    print(f'\nwrote {out} ({len(rows)} segments for {len(settings)} setting(s); every config passed assert_invariants.py)')
    if args.test_only:
        print('\nsbatch --test-only (dependencies omitted):')
        for name, rc, msg in tests:
            print(f'  {"OK  " if rc == 0 else "FAIL"} {name}: {msg}')
        if any(rc for _, rc, _ in tests):
            sys.exit('sbatch --test-only rejected at least one segment')
    if args.submit:
        subprocess.run(['bash', out], check=True)


if __name__ == '__main__':
    main()
