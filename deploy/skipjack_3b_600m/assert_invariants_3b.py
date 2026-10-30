#!/usr/bin/env python
"""Refuse a 3B rewrite-600m config that differs from what registry_3b.py specifies.

    python deploy/skipjack_3b_600m/assert_invariants_3b.py --config <seg>.yaml [--corpus] [--check-resume]
    python deploy/skipjack_3b_600m/assert_invariants_3b.py --all [--corpus]       # every config of the live grid (SETTINGS)
    python deploy/skipjack_3b_600m/assert_invariants_3b.py --prepared [--corpus]  # the prepared settings' configs
    python deploy/skipjack_3b_600m/assert_invariants_3b.py --grid [--corpus]      # all four settings (live + prepared)

Checks:
  * the config equals generate_configs_3b.build() for its (setting, kind) under the layout recorded in its .env:
    model, batch (1024 x 2048), LR schedule, optimizer, steps, resume source, every path — field for field;
  * every path is inside a 3B namespace and none is under a 1.5B or 100M root (registry_3b.FORBIDDEN_PREFIXES);
  * all rendered configs share one layout (micro_batch_size must be identical across settings: Loss.forward's
    masked_mean is per micro-batch, so a different mbs is a different objective; tools/assert_invariants.py);
  * --corpus: the tokenized folder holds .ds shards whose .ds.metadata name the training tokenizer directory and
    whose token totals sum exactly to the manifest's corpus_train_tokens minus the empty-text documents
    (registry_3b.TOKENIZED_TOKENS; one EOS per non-empty document, no BOS), and datatrove read rows - empty documents;
  * --check-resume (in-job, before torchrun): trunk1 has the seeded init at trunk/0, later trunk segments have a
    latest.txt at or past their first step, a branch's source step directory exists.
Exit 0 = OK.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_configs_3b as G  # noqa: E402
import registry_3b as R  # noqa: E402

CFG_DIR = R.cfg_dir(R.SETTINGS[0])          # the live grid; --all checks exactly this directory


def config_paths(settings) -> list[Path]:
    """Every rendered config of `settings`, from their cfg_dir; refuses a missing or extra config."""
    paths = []
    for d in sorted({R.cfg_dir(s) for s in settings}):
        mine = [s for s in settings if R.cfg_dir(s) == d]
        want = sorted(d / f'{R.seg_name(s, k)}.yaml' for s in mine for k in R.KINDS)
        have = sorted(d.glob('*.yaml'))
        if have != want:
            sys.exit(f'{d}: {len(have)} configs, expected exactly {len(want)} for {mine}')
        paths += have
    return paths


def identify(cfg: dict) -> tuple[str, str]:
    run = cfg['general']['run']
    for s in R.GRID_SETTINGS:
        for k in R.KINDS:
            if run == R.seg_name(s, k):
                return s, k
    raise SystemExit(f'general.run {run!r} is not a segment of the 3B grid')


def layout_of(cfg_path: Path) -> str:
    env = cfg_path.with_suffix('.env')
    for ln in env.read_text().splitlines() if env.is_file() else []:
        if ln.startswith('export KYS3B_LAYOUT='):
            return ln.split('=', 1)[1].strip()
    raise SystemExit(f'{env}: missing or has no KYS3B_LAYOUT')


def _paths(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _paths(v)
    elif isinstance(x, list):
        for v in x:
            yield from _paths(v)
    elif isinstance(x, str) and x.startswith('/'):
        yield x


def check_config(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    setting, kind = identify(cfg)
    lay = layout_of(path)
    want = G.build(setting, kind, R.LAYOUTS[lay])
    errs = []

    def diff(a, b, where=''):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                diff(a.get(k, '<missing>'), b.get(k, '<missing>'), f'{where}.{k}')
        elif a != b:
            errs.append(f'{path.name}: {where} = {a!r}, registry says {b!r}')
    diff(cfg, want)
    env = path.with_suffix('.env').read_text()
    if env != G.env_text(setting, kind, lay, R.LAYOUTS[lay]):
        errs.append(f'{path.with_suffix(".env").name}: differs from generate_configs_3b.env_text()')
    base = yaml.safe_load(R.BASELINE_MODEL_CONFIG.read_text())['model']      # the 1.5B architecture, read only
    mine = dict(cfg['model'], model_config=dict(cfg['model']['model_config']))
    if mine['model_config'].pop('num_hidden_layers') != R.N_LAYERS:
        errs.append(f'{path.name}: num_hidden_layers is not {R.N_LAYERS}')
    base['model_config'].pop('num_hidden_layers')
    if mine != base:
        errs.append(f'{path.name}: model block differs from the 1.5B config beyond num_hidden_layers')
    for p in _paths(cfg):
        if any(p.startswith(f) for f in R.FORBIDDEN_PREFIXES):
            errs.append(f'{path.name}: path {p} is inside a 1.5B/100M namespace')
    t = cfg['tokens']
    if t['micro_batch_size'] * t['batch_accumulation_per_replica'] * cfg['parallelism']['dp'] * t['sequence_length'] != R.TOKENS_PER_STEP:
        errs.append(f'{path.name}: tokens/step != {R.TOKENS_PER_STEP}')
    return errs


def check_corpus(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    setting, _ = identify(cfg)
    folder = Path(cfg['data_stages'][0]['data']['dataset']['dataset_folder'][0])
    shards = sorted(folder.glob('*.ds'))
    if not shards:
        return [f'{folder}: no .ds shards (run tokenize_3b.sbatch first)']
    errs, total, toks = [], 0, set()
    for ds in shards:
        meta = Path(str(ds) + '.metadata')
        if not meta.is_file() or not Path(str(ds) + '.index').is_file():
            errs.append(f'{ds}: missing .metadata or .index')
            continue
        lines = meta.read_text().splitlines()
        toks.add(lines[0])
        n = int(lines[1])
        if ds.stat().st_size != 2 * n:
            errs.append(f'{ds}: size {ds.stat().st_size} != 2 x {n} tokens')
        total += n
    if toks != {f'{R.TOKENIZER}|2'}:
        errs.append(f'{folder}: metadata tokenizer strings {sorted(toks)} != {{{R.TOKENIZER}|2}} (run tools/fix_ds_metadata.py)')
    if len(shards) != R.CORPUS_PARTS[setting]:
        errs.append(f'{folder}: {len(shards)} shards, expected {R.CORPUS_PARTS[setting]} (one per shuffled part, in order)')
    if total != R.TOKENIZED_TOKENS[setting]:
        errs.append(f'{folder}: {total:,} tokens != manifest corpus_train_tokens {R.CORPUS_TRAIN_TOKENS[setting]:,} '
                    f'- {R.CORPUS_EMPTY_DOCS[setting]} empty-text documents = {R.TOKENIZED_TOKENS[setting]:,}')
    stats = R.REPORTS / 'tokenize' / setting / 'datatrove' / 'stats.json'
    if stats.is_file():
        read = json.loads(stats.read_text())[0]['stats']['documents']['total']
        if read != R.CORPUS_ROWS[setting] - R.CORPUS_EMPTY_DOCS[setting]:
            errs.append(f'{stats}: datatrove read {read:,} documents != rows {R.CORPUS_ROWS[setting]:,} - empty {R.CORPUS_EMPTY_DOCS[setting]}')
    return errs


def check_resume(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    setting, kind = identify(cfg)
    ck = cfg['checkpoints']
    if kind == 'trunk1':
        trunk = Path(ck['checkpoints_path'])
        if not (trunk / 'latest.txt').is_file():
            return [f'{trunk}: no latest.txt (the submitter seeds trunk/0 from {R.INIT_ROOT}/_init_3B_seed{R.SEED}/0)']
        return []
    if kind.startswith('trunk'):
        lt = Path(ck['resume_checkpoint_path']) / 'latest.txt'
        if not lt.is_file() or int(lt.read_text().strip()) < R.FIRST_STEP[kind]:
            return [f'{lt}: missing or before step {R.FIRST_STEP[kind]}']
        return []
    src = Path(ck['resume_checkpoint_path'])
    if src.name != str(R.FIRST_STEP[kind]) or not (src / 'model_config.json').is_file():
        return [f'{src}: branch source step {R.FIRST_STEP[kind]} missing (a branch must never start from random init)']
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', type=Path)
    ap.add_argument('--all', action='store_true', help=f'check every config in {CFG_DIR}')
    ap.add_argument('--prepared', action='store_true', help='check every config of registry_3b.PREPARED_SETTINGS')
    ap.add_argument('--grid', action='store_true', help='check every config of all four settings')
    ap.add_argument('--corpus', action='store_true')
    ap.add_argument('--check-resume', action='store_true')
    args = ap.parse_args(argv)
    paths = sorted(CFG_DIR.glob('*.yaml')) if args.all else [args.config]
    if args.all and len(paths) != len(R.SETTINGS) * len(R.KINDS):
        sys.exit(f'{CFG_DIR}: {len(paths)} configs, expected {len(R.SETTINGS) * len(R.KINDS)}')
    if args.prepared or args.grid:
        paths = config_paths(R.PREPARED_SETTINGS if args.prepared else R.GRID_SETTINGS)
    multi = args.all or args.prepared or args.grid
    errs = []
    for p in paths:
        errs += check_config(p)
        if args.corpus:
            errs += check_corpus(p)
        if args.check_resume:
            errs += check_resume(p)
    if multi:
        lays = {layout_of(p) for p in paths} | ({layout_of(sorted(CFG_DIR.glob('*.yaml'))[0])} if args.prepared else set())
        if len(lays) != 1:
            errs.append(f'configs mix layouts {sorted(lays)}; one layout for the whole grid')
    if errs:
        print('INVARIANTS VIOLATED:\n  ' + '\n  '.join(dict.fromkeys(errs)))
        return 1
    print(f'OK: {len(paths)} config(s) match registry_3b'
          + (' + corpus totals' if args.corpus else '') + (' + resume sources' if args.check_resume else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
