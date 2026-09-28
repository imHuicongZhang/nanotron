#!/usr/bin/env python
"""Build the 15B experiment's quality-base block: the TOP 10B tokens of REMAINING
ranked purely by DCLM-fasttext (fasttext-ranking-v2) DESC.

This reuses the EXACT selection logic of the existing 10B/quality-base block
(see select_10b.py). The ONLY two changes vs that block are:
  (1) token target: 5B -> 10B, and
  (2) exclusion set: REMAINING = all - val - shared-top-5B, loaded from disk.

Ranking = fasttext-ranking-v2 ALONE, DESC. Tiebreak = the recovered deterministic
random-permutation priority ASC (seed 42, SeedSequence child #1) -- NOT doc_id ASC.
Fill = (tokens-llama2 + 1), accumulate until cumsum first >= target, keep last doc
whole (no truncation).

Read-only on 6_merged_clean and on all experiments/ inputs. Writes ONLY under
experiments/train/15B/quality-base/. seed=42. Idempotent, atomic per-shard write.

Two phases:
  --phase report   PASS1 + load exclusions + STEP 0 sizes + first/last kept doc.
                   Writes NOTHING under OUTPUT. (default; the review gate.)
  --phase commit   recompute identical selection + write parquet shards +
                   doc_ids.npy + _manifest.json + SELECTION_REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- paths / constants
BASE   = Path('/scratch/bvandur1/zhuicon1/data_rewrite')
CLEAN  = BASE / '6_merged_clean'
EXP    = BASE / 'experiments'
VAL_DIR = EXP / 'val'
SHARED_DIR = EXP / 'train' / '10B' / 'shared-top-5B'
QB10_DIR   = EXP / 'train' / '10B' / 'quality-base'   # for info-only overlap report
OUTPUT     = EXP / 'train' / '15B' / 'quality-base'   # <- the ONLY write target
CODE_DIR = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/04_select')

# verbatim from select_10b.py
NSHARDS, ROWS_FULL, LAST_ROWS, N_EXPECT = 200, 500_000, 449_162, 99_949_162

QBASE15_TARGET = 10_000_000_000   # the ONLY changed target vs 10B/quality-base (was 5B)

VAL_SIZE = 50_000
SEED = 42

V2_FT = 'fasttext-ranking-v2'

# seed sequence children -- identical mapping to select_10b.py so `tie` is identical
_SS = np.random.SeedSequence(SEED)
_CH = _SS.spawn(8)
RNG_TIE = np.random.default_rng(_CH[1])   # child1: global tie-break priority for DESC ranking


def shard_path(i): return CLEAN / f'merged_clean_{i:05d}.parquet'
def shard_rows(i): return ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS
def shard_offset(i): return i * ROWS_FULL
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- PASS 1: load numeric cols
def _read_numeric(i):
    t = pq.read_table(shard_path(i),
                      columns=['doc_id', 'tokens-llama2', V2_FT],
                      use_threads=False)
    did = t.column('doc_id').to_numpy(zero_copy_only=False)
    tok = t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64) + 1  # +1 BOS
    ft = t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32)
    nulls = {c: t.column(c).null_count for c in ['tokens-llama2', V2_FT]}
    return i, t.num_rows, did, tok, ft, nulls


def pass1(workers):
    log(f'PASS1: loading doc_id/tokens-llama2/{V2_FT} across 200 shards with {workers} workers...')
    tok = np.empty(N_EXPECT, np.int64)
    ft = np.empty(N_EXPECT, np.float32)
    null_tot = Counter()
    t0 = time.time(); seen = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_read_numeric, i) for i in range(NSHARDS)]
        for fut in as_completed(futs):
            i, n, did, tk, a, nl = fut.result()
            if n != shard_rows(i): stop(f'shard {i}: {n} rows != {shard_rows(i)}')
            lo = shard_offset(i); hi = lo + n
            if not np.array_equal(did, np.arange(lo, hi, dtype=did.dtype)):
                stop(f'shard {i}: doc_id not contiguous')
            tok[lo:hi] = tk; ft[lo:hi] = a
            for k, v in nl.items(): null_tot[k] += v
            seen += 1
            if seen % 50 == 0: log(f'  PASS1 {seen}/200 ({time.time()-t0:.0f}s)')
    if any(null_tot.values()):
        stop(f'nulls found: {dict(null_tot)}')
    log(f'PASS1 OK ({time.time()-t0:.0f}s): N={N_EXPECT:,}, zero nulls.')
    return tok, ft


# ----------------------------------------------------------------------------- selection primitives (verbatim)
def order_desc(idxs, score, tie):
    """idxs sorted by score DESC, ties by random priority `tie` ASC (deterministic)."""
    return idxs[np.lexsort((tie[idxs], -score[idxs].astype(np.float64)))]


def fill_to(order, tok, target):
    """Accumulate (tok) along `order` until cumsum first >= target; keep last doc whole.
    Returns (selected_idx, total_tok, overshoot, filled_bool)."""
    if order.size == 0:
        return order, 0, -target, False
    c = np.cumsum(tok[order])
    if c[-1] < target:
        return order, int(c[-1]), int(c[-1] - target), False
    i = int(np.searchsorted(c, target, side='left'))
    return order[:i + 1], int(c[i]), int(c[i] - target), True


def mask_of(idx):
    m = np.zeros(N_EXPECT, bool); m[idx] = True; return m


# ----------------------------------------------------------------------------- exclusion loaders
def _doc_ids_from_dir(d, label):
    """Concatenate the `doc_id` column across all part_*.parquet shards in `d`."""
    parts = sorted(d.glob('part_*.parquet'))
    if not parts:
        stop(f'{label}: no part_*.parquet under {d}')
    chunks = []
    for p in parts:
        col = pq.read_table(p, columns=['doc_id'], use_threads=False).column('doc_id')
        chunks.append(col.to_numpy(zero_copy_only=False).astype(np.int64))
    ids = np.concatenate(chunks)
    return ids


def load_exclusions():
    val_idx = np.load(VAL_DIR / 'val_doc_ids.npy').astype(np.int64)
    if val_idx.size != VAL_SIZE:
        log(f'WARN: val size {val_idx.size:,} != expected {VAL_SIZE:,}')
    shared_idx = _doc_ids_from_dir(SHARED_DIR, 'shared-top-5B')
    # sanity: indices in range, unique
    for nm, a in (('val', val_idx), ('shared-top-5B', shared_idx)):
        if a.min() < 0 or a.max() >= N_EXPECT:
            stop(f'{nm}: doc_id out of range [0,{N_EXPECT})')
        if np.unique(a).size != a.size:
            stop(f'{nm}: doc_ids not unique')
    return val_idx, shared_idx


# ----------------------------------------------------------------------------- selection
def build_selection(tok, ft):
    tie = RNG_TIE.permutation(N_EXPECT).astype(np.int64)
    val_idx, shared_idx = load_exclusions()
    val_mask = mask_of(val_idx)
    shared_mask = mask_of(shared_idx)

    val_tok = int(tok[val_idx].sum())
    shared_tok = int(tok[shared_idx].sum())

    remaining_mask = (~val_mask) & (~shared_mask)
    remaining_idx = np.flatnonzero(remaining_mask)
    rem_tok = int(tok[remaining_idx].sum())

    log(f'STEP0 val:            {val_idx.size:,} docs, {val_tok:,} tok')
    log(f'STEP0 shared-top-5B:  {shared_idx.size:,} docs, {shared_tok:,} tok  (removed set (b))')
    log(f'STEP0 REMAINING:      {remaining_idx.size:,} docs, {rem_tok:,} tok '
        f'(= all - val - shared-top-5B)')

    order = order_desc(remaining_idx, ft, tie)
    qb_idx, qb_tok, qb_over, filled = fill_to(order, tok, QBASE15_TARGET)

    info = dict(val_idx=val_idx, shared_idx=shared_idx, val_mask=val_mask,
                shared_mask=shared_mask, remaining_idx=remaining_idx, rem_tok=rem_tok,
                val_tok=val_tok, shared_tok=shared_tok, order=order)

    if not filled:
        log(f'*** REMAINING cannot fill 10B. MAX achievable = {qb_tok:,} tok '
            f'over {qb_idx.size:,} docs (target {QBASE15_TARGET:,}).')
        stop('REMAINING total tokens < 10B target; not backfilling from excluded sets.')

    # first/last kept doc
    first_i = int(order[0]); last_i = int(qb_idx[-1])
    log(f'FIRST kept: doc_id={first_i}  {V2_FT}={float(ft[first_i]):.6f}')
    log(f'LAST  kept: doc_id={last_i}  {V2_FT}={float(ft[last_i]):.6f}')
    log(f'SELECTION: {qb_idx.size:,} docs, {qb_tok:,} tok, overshoot {qb_over:,} over {QBASE15_TARGET:,}')

    return qb_idx, qb_tok, qb_over, info


# ----------------------------------------------------------------------------- manifest / report
def write_manifest(qb_idx, qb_tok, qb_over, info, ft, tok):
    first_i = int(info['order'][0]); last_i = int(qb_idx[-1])
    m = dict(
        block='quality-base',
        target_tokens=int(QBASE15_TARGET),
        docs=int(qb_idx.size),
        train_tokens_sum=int(qb_tok),
        overshoot_tokens=int(qb_over),
        method='fasttext-v2 DESC, top-10B of REMAINING (all - val - shared-top-5B)',
        seed=SEED,
        pct_of_base=dict(base='remaining_tokens',
                         value=100.0 * qb_tok / info['rem_tok'] if info['rem_tok'] else None),
        ranking_column=V2_FT,
        tiebreak='deterministic random permutation priority ASC (seed 42, SeedSequence child #1)',
        first_kept=dict(doc_id=first_i, fasttext_ranking_v2=float(ft[first_i])),
        last_kept=dict(doc_id=last_i, fasttext_ranking_v2=float(ft[last_i])),
        exclusions=dict(val_docs=int(info['val_idx'].size),
                        shared_top_5B_docs=int(info['shared_idx'].size)),
    )
    return m


def stats_block(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                p99=float(np.percentile(a, 99)),
                min=float(a.min()), max=float(a.max()))


def write_final_report(manifest, qb_idx, info, ft, tok, overlap_tok):
    sel_ft = ft[qb_idx]
    sel_len = tok[qb_idx] - 1   # raw doc token length (tokens-llama2, without BOS)
    ftd = stats_block(sel_ft)
    lend = stats_block(sel_len)

    # disjointness asserts (must be 0)
    sel = qb_idx
    cap_val = int(np.intersect1d(sel, info['val_idx'], assume_unique=False).size)
    cap_shared = int(np.intersect1d(sel, info['shared_idx'], assume_unique=False).size)

    L = ['# 15B/quality-base selection — FINAL report', '',
         f'- seed={SEED}; ranking = {V2_FT} ALONE, DESC; budgets use (tokens-llama2 + 1).',
         f'- tiebreak = {manifest["tiebreak"]}.',
         f'- REMAINING = all - val - shared-top-5B = {info["remaining_idx"].size:,} docs, '
         f'{info["rem_tok"]:,} tok.', '',
         '## Selection',
         f'- docs: {manifest["docs"]:,}',
         f'- train_tokens_sum (Σ tokens-llama2+1): {manifest["train_tokens_sum"]:,}',
         f'- target: {manifest["target_tokens"]:,}  | overshoot: {manifest["overshoot_tokens"]:,}'
         f'  | achieved >= target: {manifest["train_tokens_sum"] >= manifest["target_tokens"]}',
         f'- first kept: doc_id={manifest["first_kept"]["doc_id"]} '
         f'({V2_FT}={manifest["first_kept"]["fasttext_ranking_v2"]:.6f})',
         f'- last kept:  doc_id={manifest["last_kept"]["doc_id"]} '
         f'({V2_FT}={manifest["last_kept"]["fasttext_ranking_v2"]:.6f})', '',
         f'## {V2_FT} over selection',
         f'- mean={ftd["mean"]:.6f} median={ftd["median"]:.6f} p10={ftd["p10"]:.6f} '
         f'p90={ftd["p90"]:.6f} min={ftd["min"]:.6f} max={ftd["max"]:.6f}', '',
         '## doc token length (tokens-llama2, raw)',
         f'- mean={lend["mean"]:.1f} median={lend["median"]:.1f} p90={lend["p90"]:.1f} '
         f'p99={lend["p99"]:.1f}', '',
         '## Disjointness (must be 0)',
         f'- selection ∩ val = {cap_val}',
         f'- selection ∩ shared-top-5B = {cap_shared}', '',
         '## Info only (overlap is BY DESIGN, not an assert)',
         f'- token overlap with 10B/quality-base = {overlap_tok:,} tok (expected ~5B: the new '
         f'block is a 10B prefix whose first 5B IS the existing quality-base).', '']

    if cap_val != 0 or cap_shared != 0:
        stop(f'disjointness violated: val∩={cap_val}, shared∩={cap_shared}')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'SELECTION_REPORT.md').write_text('\n'.join(L))
    # echo to stdout too
    log('FINAL REPORT:\n' + '\n'.join(L))
    return cap_val, cap_shared


def overlap_with_qb10(qb_idx, tok):
    """Info-only token overlap between this selection and 10B/quality-base."""
    qb10 = _doc_ids_from_dir(QB10_DIR, '10B/quality-base')
    inter = np.intersect1d(qb_idx, qb10, assume_unique=False)
    return int(tok[inter].sum()), int(inter.size), int(qb10.size)


# ----------------------------------------------------------------------------- PASS 2: write shards
_WMASK = None   # set before fork; workers inherit via COW


def _block_out(k):
    return OUTPUT / f'part_{k:05d}.parquet'


def _write_one(k):
    p = shard_path(k)
    lo = shard_offset(k); n = shard_rows(k)
    sel = _WMASK[lo:lo + n]
    cnt = int(sel.sum())
    if cnt == 0:
        return k, 0
    outp = _block_out(k)
    if outp.exists():
        try:
            if pq.ParquetFile(outp).metadata.num_rows == cnt:
                return k, cnt
        except Exception:
            pass
    t = pq.read_table(p, use_threads=False)
    sub = t.filter(pa.array(sel))
    tmp = outp.with_suffix(outp.suffix + '.tmp')
    pq.write_table(sub, tmp, compression='zstd')
    os.replace(tmp, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'shard {k}: rowcount mismatch')
    return k, cnt


def write_shards(mask, workers):
    log(f'PASS2: writing quality-base shards with {workers} workers...')
    global _WMASK
    _WMASK = mask
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_one, k): k for k in range(NSHARDS)}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                _, c = fut.result()
            except Exception as e:   # noqa: BLE001
                stop(f'PASS2 shard {k:05d} failed: {e!r}')
            total += c
            done += 1
            if done % 25 == 0: log(f'  PASS2 {done}/200 ({time.time()-t0:.0f}s)')
    log(f'PASS2 OK ({time.time()-t0:.0f}s): wrote {total:,} rows.')
    if total != int(mask.sum()):
        stop(f'written {total} != selected {int(mask.sum())}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['report', 'commit'], default='report')
    ap.add_argument('--write-mem-per-worker-gb', type=float, default=7.0)
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    write_workers = max(1, min(cpus, int((mem_gb - 16) // args.write_mem_per_worker_gb)))
    log(f'START phase={args.phase}: cpus={cpus} mem={mem_gb:.0f}G -> pass1 {cpus} workers, '
        f'write {write_workers} workers; seed={SEED}; target={QBASE15_TARGET:,}')

    t_start = time.time()
    tok, ft = pass1(cpus)
    qb_idx, qb_tok, qb_over, info = build_selection(tok, ft)

    if args.phase == 'report':
        log('phase=report: STEP 0 + selection computed; NOTHING written. '
            'Re-run with --phase commit to save.')
        log(f'DONE (report) in {time.time()-t_start:.0f}s.')
        return

    # ---- commit ----
    manifest = write_manifest(qb_idx, qb_tok, qb_over, info, ft, tok)
    overlap_tok, overlap_docs, qb10_docs = overlap_with_qb10(qb_idx, tok)
    manifest['overlap_with_10B_quality_base'] = dict(
        tokens=overlap_tok, docs=overlap_docs, of_10B_quality_base_docs=qb10_docs)

    # write parquet shards
    sel_mask = mask_of(qb_idx)
    write_shards(sel_mask, write_workers)

    # doc_ids.npy (sorted ascending; mirrors val_doc_ids.npy convention)
    # NOTE: write via a file handle so np.save does NOT append ".npy" to the tmp name.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc_ids = np.sort(qb_idx).astype(np.int64)
    tmp = OUTPUT / 'doc_ids.npy.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, doc_ids)
    os.replace(tmp, OUTPUT / 'doc_ids.npy')

    # _manifest.json (atomic)
    mtmp = OUTPUT / '_manifest.json.tmp'
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, OUTPUT / '_manifest.json')

    # final report + disjointness asserts
    write_final_report(manifest, qb_idx, info, ft, tok, overlap_tok)
    log(f'ALL DONE (commit) in {time.time()-t_start:.0f}s. Output -> {OUTPUT}')


if __name__ == '__main__':
    main()
