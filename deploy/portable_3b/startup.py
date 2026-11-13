"""Derived configs and log checks for the shared init, the bounded startup validation, and trunk1 continuation.

Every function takes the production config it derives from EXPLICITLY (a rendered trunk1 YAML) and copies it; only
the keys listed in INIT_OVERRIDES / startup_overrides() change. Nothing here reads a committed config or a site path,
so a config rendered for any site yields an init / startup config for that same site (regression:
test_portable_3b_init.py runs the CLI on a config with unrelated paths and W&B values).

  derive_init      1 step with save_initial_state; its step 0 becomes the shared init (initlib.finalize_init)
  derive_startup   s_trunk (steps 0->20, writes trunk/20), s_resume (20->30 with optimizer + LR state),
                   s_straight (0->30 uninterrupted in a scratch folder): resume-vs-uninterrupted reference
  summarize        checks of the three startup logs (finite, lr == scheduler, tokens == step x 2,097,152, resume
                   matches the uninterrupted run, trunk/20 and trunk/30 written, optional memory budget)
  resume_config    trunk1 past step 0: the same config with load_optimizer / load_lr_scheduler true
  check_log        health of a running/finished segment log
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml

import science_3b as S

STARTUP_PHASES = ['s_trunk', 's_resume', 's_straight']
INIT_RUN = f'3b_init_seed{S.SEED}'
STEP_RE = re.compile(r'iteration: (\d+) / \d+ .*?consumed_tokens: ([\d.]+)([KMBT]?) .*?time_per_iteration_ms: ([\d.]+)(K?) '
                     r'.*?grad_norm: ([\w.+-]+) \| lm_loss: ([\w.+-]+) \| lr: ([\w.+-]+)')
UNIT = {'': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}


class StartupError(SystemExit):
    pass


def load(cfg_path: Path) -> dict:
    return yaml.safe_load(Path(cfg_path).read_text())


def dump(c: dict, path: Path, header: str = '') -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(header + yaml.safe_dump(c, sort_keys=False))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require_trunk1(c: dict, where) -> str:
    run = c['general']['run']
    if not run.endswith('_trunk1') or c['checkpoints']['load_optimizer'] or c['checkpoints']['load_lr_scheduler']:
        raise StartupError(f'{where}: not a rendered trunk1 config (weights-only load from the init); got run {run!r}')
    for s in S.ALL_SETTINGS:
        if run == S.seg_name(s, 'trunk1'):
            return s
    raise StartupError(f'{where}: general.run {run!r} is not a trunk1 segment of the 3B grid')


def init_overrides(ckpt: Path) -> dict:
    """The only keys the init config changes, as dotted path -> value."""
    return {'general.run': INIT_RUN, 'tokens.train_steps': 1,
            'checkpoints.checkpoints_path': str(ckpt), 'checkpoints.resume_checkpoint_path': None,
            'checkpoints.load_optimizer': False, 'checkpoints.load_lr_scheduler': False,
            'checkpoints.checkpoint_interval': 100000, 'checkpoints.save_initial_state': True,
            'checkpoints.save_final_state': False}


def _apply(c: dict, overrides: dict) -> dict:
    c = copy.deepcopy(c)
    for dotted, v in overrides.items():
        *path, leaf = dotted.split('.')
        d = c
        for k in path:
            d = d[k]
        if leaf not in d:
            raise StartupError(f'override {dotted}: key absent from the production config')
        d[leaf] = v
    return c


def derive_init(cfg_path: Path, out: Path, ckpt: Path) -> dict:
    """The 1-step init config from the SUPPLIED production trunk1 config (model, tokenizer, data, parallelism,
    batch, optimizer, W&B project preserved); writes `out` and returns its record."""
    prod = load(cfg_path)
    setting = require_trunk1(prod, cfg_path)
    ov = init_overrides(Path(ckpt))
    c = _apply(prod, ov)
    rec = {'source_config': str(Path(cfg_path).resolve()), 'source_config_sha256': sha256_file(cfg_path),
           'source_setting': setting, 'overrides': ov}
    head = (f'# DERIVED by deploy/portable_3b/startup.py derive_init: shared seed-{S.SEED} init (1 step, save_initial_state)\n'
            f'# from {rec["source_config"]} (sha256 {rec["source_config_sha256"]})\n'
            f'# changed keys only: {", ".join(ov)}\n')
    dump(c, out, head)
    return rec


def startup_overrides(work: Path) -> dict:
    a, s = S.STARTUP_STEPS
    straight = str(Path(work) / 'ckpt' / 'straight')
    return {'s_trunk': {'tokens.train_steps': a},
            's_resume': {'tokens.train_steps': s, 'checkpoints.load_optimizer': True, 'checkpoints.load_lr_scheduler': True},
            's_straight': {'general.run': '3b_startup_straight_reference', 'tokens.train_steps': s,
                           'checkpoints.checkpoints_path': straight, 'checkpoints.resume_checkpoint_path': straight,
                           'checkpoints.checkpoint_interval': 100000, 'checkpoints.save_final_state': False}}


def derive_startup(cfg_path: Path, work: Path) -> dict:
    """<work>/cfg/{s_trunk,s_resume,s_straight}.yaml from the SUPPLIED production trunk1 config."""
    work = Path(work)
    if (work / 'cfg').exists():
        raise StartupError(f'{work}/cfg exists; use a fresh workdir')
    prod = load(cfg_path)
    require_trunk1(prod, cfg_path)
    ovs = startup_overrides(work)
    for ph in STARTUP_PHASES:
        dump(_apply(prod, ovs[ph]), work / 'cfg' / f'{ph}.yaml',
             f'# DERIVED by deploy/portable_3b/startup.py derive_startup ({ph}) from {Path(cfg_path).resolve()}\n')
    rec = {'production_config': str(Path(cfg_path).resolve()), 'production_config_sha256': sha256_file(cfg_path),
           'overrides': ovs}
    (work / 'startup_derive.json').write_text(json.dumps(rec, indent=1))
    return rec


def resume_config(cfg_path: Path, out: Path) -> None:
    """trunk1 restarted past step 0 must load optimizer + LR state + metadata (data position), like trunk2..6. The
    rendered trunk1 is weights-only (correct only from the step-0 init); only the two load flags change."""
    c = load(cfg_path)
    require_trunk1(c, cfg_path)
    dump(_apply(c, {'checkpoints.load_optimizer': True, 'checkpoints.load_lr_scheduler': True}), out,
         f'# DERIVED by deploy/portable_3b/startup.py resume_config from {Path(cfg_path).resolve()}: trunk1 continuing '
         f'past step 0 with optimizer + LR scheduler state\n')


def expected_lr(cfg: dict, step: int) -> float:
    """The lr nanotron logs at iteration `step` (lambda(step), the next update's lr) for that config."""
    import lrsched
    key = json.dumps(cfg['optimizer']['learning_rate_scheduler'], sort_keys=True) + str(cfg['tokens']['train_steps'])
    if key not in _LAMBDAS:
        _LAMBDAS[key] = lrsched.lr_lambda(cfg)
    peak, lam = _LAMBDAS[key]
    return peak * lam(step)


_LAMBDAS: dict = {}


def parse_log(text: str) -> list[dict]:
    steps = []
    for m in STEP_RE.finditer(text):
        ms = float(m.group(4)) * (1000 if m.group(5) else 1)
        steps.append({'step': int(m.group(1)), 'tokens': float(m.group(2)) * UNIT[m.group(3)], 's': ms / 1000,
                      'grad_norm': float(m.group(6)), 'loss': float(m.group(7)), 'lr': float(m.group(8))})
    return steps


def step_problems(steps: list[dict], sched_cfg: dict, label: str) -> list[str]:
    out = []
    for st in steps:
        if not (math.isfinite(st['loss']) and math.isfinite(st['grad_norm'])):
            out.append(f'{label}: step {st["step"]} non-finite loss {st["loss"]} / grad_norm {st["grad_norm"]}')
            break
        want = expected_lr(sched_cfg, st['step'])
        if abs(st['lr'] - want) > max(5e-3 * want, 1e-12):   # logged with 3 significant digits
            out.append(f'{label}: step {st["step"]} logged lr {st["lr"]:.4g}, scheduler says {want:.4g}')
            break
        if abs(st['tokens'] - st['step'] * S.TOKENS_PER_STEP) > 5e-3 * st['step'] * S.TOKENS_PER_STEP:
            out.append(f'{label}: step {st["step"]} consumed_tokens {st["tokens"]:.4g} != step x {S.TOKENS_PER_STEP}')
            break
    return out


def peak_gib(nvsmi: Path):
    if not nvsmi.is_file():
        return None
    vals = [int(r.split(',')[1]) for r in nvsmi.read_text().splitlines()
            if r.count(',') >= 2 and r.split(',')[1].strip().isdigit()]
    return round(max(vals) / 1024, 2) if vals else None


def summarize(cfg_path: Path, work: Path, mem_budget_gib: float | None = None) -> dict:
    """Checks of <work>/logs/<phase>.log against the SUPPLIED production config; trunk folder = its checkpoints_path.
    mem_budget_gib is the operator's per-GPU limit (optional; peaks from <phase>.nvsmi.csv are reported either way)."""
    work = Path(work)
    prod = load(cfg_path)
    require_trunk1(prod, cfg_path)
    rep = {'workdir': str(work), 'production_config': str(Path(cfg_path).resolve()), 'phases': {}, 'problems': [],
           'mem_budget_gib': mem_budget_gib}
    a, s = S.STARTUP_STEPS
    trunk = Path(prod['checkpoints']['checkpoints_path'])
    for ph in STARTUP_PHASES:
        log = work / 'logs' / f'{ph}.log'
        if not log.is_file():
            rep['problems'].append(f'{ph}: no log')
            continue
        steps = parse_log(log.read_text(errors='replace'))
        rep['problems'] += step_problems(steps, prod, ph)
        timed = [x['s'] for x in steps[2:]]
        pk = peak_gib(work / 'logs' / f'{ph}.nvsmi.csv')
        if pk is not None and mem_budget_gib is not None and pk > mem_budget_gib:
            rep['problems'].append(f'{ph}: peak {pk} GiB above the {mem_budget_gib} GiB budget')
        rep['phases'][ph] = {'steps_logged': len(steps), 'first_step': steps[0]['step'] if steps else None,
                             'last_step': steps[-1]['step'] if steps else None,
                             's_per_it_median': statistics.median(timed) if timed else None, 'peak_gib': pk, 'steps': steps}
    P = rep['phases']
    want_n = {'s_trunk': (1, a), 's_resume': (a + 1, s), 's_straight': (1, s)}
    for ph, (f, l) in want_n.items():
        if ph in P and (P[ph]['first_step'], P[ph]['last_step'], P[ph]['steps_logged']) != (f, l, l - f + 1):
            rep['problems'].append(f'{ph}: logged steps {P[ph]["first_step"]}..{P[ph]["last_step"]} '
                                   f'({P[ph]["steps_logged"]}), expected {f}..{l}')
    if all(ph in P for ph in STARTUP_PHASES):
        by = {ph: {x['step']: x for x in P[ph]['steps']} for ph in STARTUP_PHASES}
        noise = [abs(by['s_trunk'][n]['loss'] - by['s_straight'][n]['loss']) for n in by['s_trunk'] if n in by['s_straight']]
        diff = [abs(by['s_resume'][n]['loss'] - by['s_straight'][n]['loss']) for n in by['s_resume'] if n in by['s_straight']]
        lr_eq = all(by['s_resume'][n]['lr'] == by['s_straight'][n]['lr'] for n in by['s_resume'] if n in by['s_straight'])
        tok_eq = all(by['s_resume'][n]['tokens'] == by['s_straight'][n]['tokens'] for n in by['s_resume'] if n in by['s_straight'])
        losses = [x['loss'] for x in by['s_straight'].values()]
        unit = max(10 ** (math.floor(math.log10(v)) - 2) for v in losses) if losses else 0
        tol = max(0.02, unit, 3 * max(noise, default=0))
        rep['resume_vs_uninterrupted'] = {'max_abs_loss_diff_steps_1_20': max(noise, default=None),
                                          'max_abs_loss_diff_steps_21_30': max(diff, default=None), 'tolerance': tol,
                                          'lr_identical': lr_eq, 'consumed_tokens_identical': tok_eq}
        if len(diff) != s - a or not lr_eq or not tok_eq or max(diff) > tol:
            rep['problems'].append(f'resume differs from the uninterrupted run: {rep["resume_vs_uninterrupted"]}')
        timed = [x['s'] for x in P['s_trunk']['steps'][2:]] + [x['s'] for x in P['s_resume']['steps'][2:]]
        rep['s_per_it_median_production_layout'] = statistics.median(timed) if timed else None
    for n in (a, s):
        if not (trunk / str(n) / 'model_config.json').is_file():
            rep['problems'].append(f'missing checkpoint {trunk}/{n}')
    lt = trunk / 'latest.txt'
    if not lt.is_file() or lt.read_text().strip() != str(s):
        rep['problems'].append(f'{lt} is {lt.read_text().strip() if lt.is_file() else "missing"}, expected {s}')
    rep['peak_gib'] = max((p['peak_gib'] or 0) for p in P.values()) if P else None
    rep['ok'] = not rep['problems']
    (work / 'startup_report.json').write_text(json.dumps(rep, indent=1))
    return rep


def check_log(cfg_path: Path, log: Path) -> dict:
    """Health of a running/finished production segment log: finite, lr == scheduler, tokens == step x T, s/it."""
    steps = parse_log(Path(log).read_text(errors='replace')) if Path(log).is_file() else []
    probs = step_problems(steps, load(cfg_path), Path(log).name)
    tail = [x['s'] for x in steps[-50:]][2:] if len(steps) > 2 else []
    return {'log': str(log), 'steps_logged': len(steps), 'last': steps[-1] if steps else None,
            's_per_it_median_last50': statistics.median(tail) if tail else None, 'problems': probs}
