#!/usr/bin/env python
"""STEP 2 (wrap) — assemble ~5B rewritten tokens with a RANDOM distill supplement.

WRAP is the quality-AGNOSTIC baseline. Its supplement MUST be selected by seeded RANDOM draw,
NOT by any quality score — sorting by fasttext-ranking-v2 would inject a quality signal and
contaminate the Arm-2-vs-Arm-4 ("does source quality matter?") comparison.

  2a. ALL status==2 wrap docs (after Step 1 cleanup); total_wrap_tokens = sum(rewritten_tokens
      + 1); per-style breakdown.
  2b. GUARD: if total_wrap_tokens >= 5B (expected ~4B, never >=5B) -> STOP, write nothing.
  2c. Else keep ALL wrap status==2; fill the gap from distill status==2 by RANDOM order
      (np.random.default_rng(42).permutation — NO quality sort), cumulative (rewritten_tokens
      + 1), last doc whole. Same doc_id may appear in both passes (kept). If distill cannot
      fill the gap -> write what exists, report the shortfall prominently, and STOP (no pad).
  2d. source_prompt = "wrap_{style}" (wrap rows) / "distill" (supplement rows).
  2e. Save -> pretrain/wrap/rewritten/{wrap,distill}_NNNNN.parquet (atomic).
  2f/2g. Per-source distribution + cross-pass coverage. 2h. _assembly_manifest.json.

distill/ is currently EMPTY. Default: hard-stop unless --allow-partial-distill (records the
partial coverage; with an empty distill the result is wrap-only and will fall short of 5B).
The seed-42 RNG here is independent of Step 3's shuffle RNG (separate script / instance).
CPU-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pp_io import atomic_write_table, paired_wiki_status

# ----------------------------------------------------------------------------- paths / constants
BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B/wrap')
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

SETTING = 'wrap'
TARGET = 5_000_000_000
RANDOM_SEED = 42                      # supplement draw; independent of Step 3's shuffle seed
STYLES = ['easy', 'hard', 'wiki', 'qa']
STYLE2CODE = {s: i for i, s in enumerate(STYLES)}
SHARD_RE = re.compile(r'part_(\d+)\.parquet$')
V2_FT = 'fasttext-ranking-v2'

KEEP_WRAP = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2',
             'status', 'wrap_style', V2_FT, 'topic', 'url', 'metadata']
KEEP_DISTILL = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2',
                'status', V2_FT, 'topic', 'url', 'metadata']


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


def list_shards(subdir):
    out = []
    for p in sorted(glob.glob(str(BASE / subdir / 'part_*.parquet'))):
        out.append((int(SHARD_RE.search(os.path.basename(p)).group(1)), p))
    return out


# ----------------------------------------------------------------------------- numeric pass
def _read_numeric(job):
    subdir, k, path = job
    cols = ['doc_id', 'status', 'rewritten_tokens', 'tokens-llama2']
    has_style = subdir == 'rewritten'
    if has_style:
        cols.append('wrap_style')
    t = pq.read_table(path, columns=cols, use_threads=False)
    n = t.num_rows
    if has_style:
        styles = t.column('wrap_style').to_pylist()
        code = np.fromiter((STYLE2CODE.get(s, -1) for s in styles), count=n, dtype=np.int8)
    else:
        code = np.full(n, -1, dtype=np.int8)
    return dict(
        k=k, n=n,
        doc_id=t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64),
        status=t.column('status').to_numpy(zero_copy_only=False).astype(np.int8),
        rtok=t.column('rewritten_tokens').to_numpy(zero_copy_only=False).astype(np.int64),
        code=code,
        llama=t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64))


def collect_numeric(subdir, shards, workers):
    if not shards:
        return dict(present=[], pos={}, offs=np.zeros(1, np.int64), paths={},
                    doc_id=np.empty(0, np.int64), status=np.empty(0, np.int8),
                    rtok=np.empty(0, np.int64), code=np.empty(0, np.int8),
                    llama=np.empty(0, np.int64), len=np.empty(0, np.int64))
    res = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_read_numeric, (subdir, k, p)): k for k, p in shards}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'{subdir} numeric shard {k:05d} failed: {e!r}')
            res[r['k']] = r
            done += 1
            if done % 50 == 0:
                log(f'  numeric {subdir}: {done}/{len(shards)} ({time.time()-t0:.0f}s)')
    present = [k for k, _ in shards]
    ordered = [res[k] for k in present]
    rows = np.array([s['n'] for s in ordered], dtype=np.int64)
    offs = np.zeros(len(present) + 1, dtype=np.int64); offs[1:] = np.cumsum(rows)
    cat = lambda key: np.concatenate([s[key] for s in ordered])  # noqa: E731
    out = dict(present=present, pos={k: i for i, k in enumerate(present)}, offs=offs,
               paths={k: p for k, p in shards},
               doc_id=cat('doc_id'), status=cat('status'), rtok=cat('rtok'),
               code=cat('code'), llama=cat('llama'))
    out['len'] = out['rtok'] + 1
    return out


# ----------------------------------------------------------------------------- write pass
_WRITE = {}


def _write_shard(job):
    subdir, k, outprefix = job
    d = _WRITE[subdir]
    i = d['pos'][k]
    mask = d['keep'][d['offs'][i]:d['offs'][i + 1]]
    cnt = int(mask.sum())
    if cnt == 0:
        return (subdir, 0)
    t = pq.read_table(d['paths'][k], use_threads=False)
    sub = t.filter(pa.array(mask))
    if subdir == 'rewritten':
        out = sub.select(KEEP_WRAP)
        names = out.column_names
        names[names.index('rewritten')] = 'text'
        out = out.rename_columns(names)
        styles = sub.column('wrap_style').to_pylist()
        sp = pa.array([f'wrap_{s}' for s in styles], type=pa.large_string())
    else:
        out = sub.select(KEEP_DISTILL)
        names = out.column_names
        names[names.index('rewritten')] = 'text'
        out = out.rename_columns(names)
        sp = pa.array(['distill'] * cnt, type=pa.large_string())
    out = out.append_column('source_prompt', sp)

    outp = PRETRAIN / SETTING / 'rewritten' / f'{outprefix}_{k:05d}.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(out, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'{outprefix} shard {k}: rowcount mismatch')
    return (subdir, cnt)


def write_blocks(wrap, distill, keep_w, keep_d, workers):
    global _WRITE
    _WRITE = {
        'rewritten': dict(keep=keep_w, offs=wrap['offs'], pos=wrap['pos'], paths=wrap['paths']),
        'distill': dict(keep=keep_d, offs=distill['offs'], pos=distill['pos'],
                        paths=distill['paths']),
    }
    jobs = ([('rewritten', k, 'wrap') for k in wrap['present']]
            + [('distill', k, 'distill') for k in distill['present']])
    written = {'rewritten': 0, 'distill': 0}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_shard, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                subdir, cnt = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'write {j[2]} shard {j[1]:05d} failed: {e!r}')
            written[subdir] += cnt
            done += 1
            if done % 50 == 0:
                log(f'  write: {done}/{len(jobs)} ({time.time()-t0:.0f}s)')
    if written['rewritten'] != int(keep_w.sum()) or written['distill'] != int(keep_d.sum()):
        stop(f'written {written} != selected '
             f'(wrap={int(keep_w.sum())}, distill={int(keep_d.sum())})')
    return written


# ----------------------------------------------------------------------------- cross-pass coverage
def cross_pass(wrap, distill):
    if distill['doc_id'].size == 0:
        return dict(paired_docs=0, wrap_docs_without_distill=int(wrap['doc_id'].size),
                    status0_both=0, status0_both_tokens=0, status1_both=0,
                    status1_both_tokens=0, recovered_by_distill=0,
                    status2_wrap_not_distill=0,
                    unique_docs_with_any_status2=int((wrap['status'] == 2).sum()))
    w_for_d = paired_wiki_status(wrap, distill)   # assert per-shard equality; else doc_id join
    d = distill['status']; ll = distill['llama']

    def ct(mask): return int(mask.sum()), int(ll[mask].sum())
    s0n, s0t = ct((w_for_d == 0) & (d == 0))
    s1bn, s1bt = ct((w_for_d == 1) & (d == 1))
    recn, _ = ct((w_for_d == 1) & (d == 2))
    s2wn, _ = ct((w_for_d == 2) & ((d == 0) | (d == 1)))
    wrap_s2_total = int((wrap['status'] == 2).sum())
    extra = int(((d == 2) & (w_for_d != 2)).sum())
    return dict(paired_docs=int(distill['doc_id'].size),
                wrap_docs_without_distill=int(wrap['doc_id'].size - distill['doc_id'].size),
                status0_both=s0n, status0_both_tokens=s0t,
                status1_both=s1bn, status1_both_tokens=s1bt,
                recovered_by_distill=recn, status2_wrap_not_distill=s2wn,
                unique_docs_with_any_status2=wrap_s2_total + extra)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    ap.add_argument('--allow-partial-distill', action='store_true',
                    help='assemble even if distill has fewer shards than wrap (e.g. empty). '
                         'Default: hard-stop until distill is ready.')
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP2 (wrap) START: target={TARGET:,} workers={workers} random_seed={RANDOM_SEED}')

    wrap_shards = list_shards('rewritten')
    distill_shards = list_shards('distill')
    log(f'  wrap shards={len(wrap_shards)}  distill shards={len(distill_shards)}')
    if len(distill_shards) < len(wrap_shards):
        msg = f'distill is INCOMPLETE: {len(distill_shards)}/{len(wrap_shards)} shards present.'
        if not args.allow_partial_distill:
            stop(msg + ' Re-run with --allow-partial-distill to assemble on partial/empty '
                       'distill (will fall short of 5B), or wait for the distill pass.')
        log(f'  ⚠️ {msg} Proceeding (--allow-partial-distill).')

    log('=== numeric pass ===')
    wrap = collect_numeric('rewritten', wrap_shards, workers)
    distill = collect_numeric('distill', distill_shards, workers)

    # 2a wrap status==2 totals + per-style
    s2_w = wrap['status'] == 2
    total_wrap_tokens = int(wrap['len'][s2_w].sum())
    per_style = {}
    for s, c in STYLE2CODE.items():
        m = s2_w & (wrap['code'] == c)
        per_style[s] = dict(docs=int(m.sum()), tokens=int(wrap['len'][m].sum()))
    n_wrap = int(s2_w.sum())
    gap = TARGET - total_wrap_tokens
    log(f'2a: wrap status2 = {n_wrap:,} docs, {total_wrap_tokens:,} tok; gap to 5B = {gap:,}')
    for s in STYLES:
        d = per_style[s]
        log(f'    [{s}] docs={d["docs"]:,} tok={d["tokens"]:,} '
            f'({100.0*d["tokens"]/total_wrap_tokens:.1f}% of wrap)' if total_wrap_tokens else '')

    # 2b guard: never silently trim
    if total_wrap_tokens >= TARGET:
        stop(f'WRAP first pass unexpectedly reached/exceeded 5B — total_wrap_tokens='
             f'{total_wrap_tokens:,}; expected ~4B. Halting so the assembly strategy can be '
             f'reviewed. No output written.')

    # 2c keep all wrap; RANDOM distill supplement
    keep_w = s2_w.copy()
    keep_d = np.zeros(distill['doc_id'].size, dtype=bool)
    s2_d = np.flatnonzero(distill['status'] == 2)
    shortfall = False
    if s2_d.size and gap > 0:
        rng = np.random.default_rng(RANDOM_SEED)               # independent of Step 3 shuffle
        order = s2_d[rng.permutation(s2_d.size)]               # RANDOM order, NO quality sort
        c = np.cumsum(distill['len'][order])
        if c[-1] < gap:
            keep_d[order] = True
            tok_distill = int(c[-1]); shortfall = True
        else:
            cut = int(np.searchsorted(c, gap, side='left'))
            keep_d[order[:cut + 1]] = True
            tok_distill = int(c[cut])
    else:
        tok_distill = 0
        shortfall = gap > 0                                    # no distill docs to fill the gap

    n_distill = int(keep_d.sum())
    total = total_wrap_tokens + tok_distill
    overshoot = max(0, total - TARGET)
    dual = int(np.intersect1d(wrap['doc_id'][keep_w], distill['doc_id'][keep_d]).size)
    log(f'2c: distill supplement = {n_distill:,} docs, {tok_distill:,} tok (RANDOM seed '
        f'{RANDOM_SEED}); total = {total:,} (overshoot {overshoot:,})')

    cov = cross_pass(wrap, distill)

    # 2e write
    log('=== writing assembled rewritten shards ===')
    write_blocks(wrap, distill, keep_w, keep_d, workers)

    # per-source distribution of the final set
    per_source = {f'wrap_{s}': dict(docs=per_style[s]['docs'], tokens=per_style[s]['tokens'])
                  for s in STYLES}
    per_source['distill'] = dict(docs=n_distill, tokens=tok_distill)

    manifest = {
        'setting': SETTING,
        'distill_selection': 'seeded_random_seed42_no_quality_sort',
        'distill_shards_present': len(distill_shards),
        'wrap_shards_present': len(wrap_shards),
        'distill_complete': len(distill_shards) == len(wrap_shards),
        'total_docs': n_wrap + n_distill, 'total_tokens_plus_bos': int(total),
        'docs_from_wrap': n_wrap, 'tokens_from_wrap': total_wrap_tokens,
        'docs_from_wrap_per_style': {s: per_style[s]['docs'] for s in STYLES},
        'tokens_from_wrap_per_style': {s: per_style[s]['tokens'] for s in STYLES},
        'docs_from_distill': n_distill, 'tokens_from_distill': int(tok_distill),
        'dual_rewrite_doc_ids': dual,
        'target': TARGET,
        'gap_before_distill': int(TARGET - total_wrap_tokens),
        'overshoot': int(overshoot),
        'shortfall_vs_target': int(max(0, TARGET - total)),
    }
    (PRETRAIN / SETTING).mkdir(parents=True, exist_ok=True)
    (PRETRAIN / SETTING / '_assembly_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary = dict(manifest)
    summary.update(per_source=per_source, coverage=cov)
    (HERE / '_step2_wrap_summary.json').write_text(json.dumps(summary, indent=2))
    log('STEP2 DONE; wrote _assembly_manifest.json + _step2_wrap_summary.json')

    if shortfall:
        stop(f'SHORTFALL: total {total:,} < 5B by {TARGET-total:,} '
             f'(distill supplement insufficient — distill shards '
             f'{len(distill_shards)}/{len(wrap_shards)}). Wrote wrap+available-distill, did NOT '
             f'pad. Halting before Step 3 so a <10B corpus is not mixed unintentionally.')


if __name__ == '__main__':
    main()
