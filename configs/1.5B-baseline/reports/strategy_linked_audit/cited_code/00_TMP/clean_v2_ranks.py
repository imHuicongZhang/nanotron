#!/usr/bin/env python
"""Tie-aware GLOBAL percentile columns on the CLEAN merged dataset (6_merged_clean,
post-50k-removal). Append-only; operates ONLY on 6_merged_clean + a NEW flat dir.

New columns (float32, [0,1], higher = better):
    fasttext-ranking-v2, fineweb-edu-ranking-v2, modernbert-ranking-v2

Recipe (identical for all three, from the RAW score):
    pct = scipy.stats.rankdata(score, method="average") / N_actual
  * method="average": equal raw scores -> equal percentile (tie-aware; flattens
    fasttext's huge score==0 / score==1 clusters).
  * rankdata is ASCENDING and /N preserves it, so HIGHER raw score -> HIGHER pct
    (~1.0 best, ~0 worst). Same N divisor for all three.

Pure I/O + numpy/scipy; no model. Does NOT read/modify/delete the stale
4_scorers_full_ranked / 3_scorers_full_flat artifacts.

Phases: META (assert N==99,949,162) -> PASS1 (assemble 3 score arrays in doc_id
order) -> RANK (rankdata/N -> float32 npy in 6_merged_clean_flat/) -> VALIDATE
-> PASS2 (append 3 cols per shard, atomic, verify) -> report.
"""
from __future__ import annotations

import argparse
import gc
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
from scipy.stats import rankdata

# ----------------------------------------------------------------------------- paths / constants
BASE  = Path('/scratch/bvandur1/zhuicon1/data_rewrite')
CLEAN = BASE / '6_merged_clean'
FLAT  = BASE / '6_merged_clean_flat'            # NEW dir (do not reuse stale names)
REPORT = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/00_TMP/clean_v2_ranks_report.md')

NSHARDS    = 200
ROWS_FULL  = 500_000
LAST_ROWS  = 449_162
N_EXPECT   = 99_949_162                          # 199*500000 + 449162

SCORERS = ['fasttext', 'fineweb-edu', 'modernbert']
V2 = {'fasttext': 'fasttext-ranking-v2',
      'fineweb-edu': 'fineweb-edu-ranking-v2',
      'modernbert': 'modernbert-ranking-v2'}
PCT_NPY = {'fasttext':    FLAT / 'rank_fasttext_v2_pct.npy',
           'fineweb-edu': FLAT / 'rank_fineweb_v2_pct.npy',
           'modernbert':  FLAT / 'rank_modernbert_v2_pct.npy'}

ORIG_COLS = ['orig_doc_id', 'doc_id', 'text', 'url', 'metadata',
             'fasttext', 'fineweb-edu', 'modernbert', 'topic']
EXPECT_SCHEMA = ORIG_COLS + [V2['fasttext'], V2['fineweb-edu'], V2['modernbert']]

_RAW: dict[str, np.ndarray] = {}
_N: int = 0


def shard_path(i: int) -> Path:
    return CLEAN / f'merged_clean_{i:05d}.parquet'


def shard_rows(i: int) -> int:
    return ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS


def shard_offset(i: int) -> int:
    # shards 0..198 are 500k each, so offset is simply i*500000
    return i * ROWS_FULL


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def stop(m: str) -> 'NoReturn':
    print(f'\n*** STOP: {m}\n', flush=True)
    sys.exit(2)


def _init_worker() -> None:
    pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- META
def metadata_check() -> int:
    log('META: per-shard row counts...')
    tmps = sorted(str(p) for p in CLEAN.glob('*.tmp'))
    counts = []
    for i in range(NSHARDS):
        p = shard_path(i)
        if not p.exists():
            stop(f'missing shard {i:05d}: {p}')
        counts.append(pq.ParquetFile(p).metadata.num_rows)
    bad = [(i, c) for i, c in enumerate(counts) if c != shard_rows(i)]
    total = sum(counts)
    if tmps or bad or total != N_EXPECT:
        if tmps:
            print(f'  .tmp leftovers: {tmps}', flush=True)
        for i, c in bad:
            print(f'  shard {i:05d}: {c} rows (expected {shard_rows(i)})', flush=True)
        stop(f'row-count precondition failed (total={total:,}, expected {N_EXPECT:,}).')
    log(f'META OK: 200 shards (199x500k + {LAST_ROWS:,}), total {total:,}.')
    return total


# ----------------------------------------------------------------------------- PASS 1
def _read_scores(i: int):
    p = shard_path(i)
    t = pq.read_table(p, columns=SCORERS + ['doc_id'], use_threads=False)
    nrows = t.num_rows
    nulls = {c: t.column(c).null_count for c in SCORERS}
    did = t.column('doc_id').to_numpy(zero_copy_only=False)
    arrs = {c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for c in SCORERS}
    return i, nrows, nulls, did, arrs


def pass1_assemble(n_actual: int, workers: int) -> None:
    log(f'PASS1: assembling 3 score arrays in doc_id order with {workers} workers...')
    global _RAW
    _RAW = {c: np.empty(n_actual, dtype=np.float64) for c in SCORERS}
    null_tot = {c: 0 for c in SCORERS}
    seen = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_read_scores, i) for i in range(NSHARDS)]
        for fut in as_completed(futs):
            i, nrows, nulls, did, arrs = fut.result()
            if nrows != shard_rows(i):
                stop(f'shard {i:05d}: {nrows} rows != {shard_rows(i)}')
            lo = shard_offset(i)
            hi = lo + nrows
            # verify doc_id == contiguous range (proves doc_id-order placement)
            if not np.array_equal(did, np.arange(lo, hi, dtype=did.dtype)):
                stop(f'shard {i:05d}: doc_id not contiguous {lo}..{hi}')
            for c in SCORERS:
                _RAW[c][lo:hi] = arrs[c]
                null_tot[c] += nulls[c]
            seen += 1
            if seen % 50 == 0:
                log(f'  PASS1 {seen}/200 ({time.time()-t0:.0f}s)')
    if any(v > 0 for v in null_tot.values()):
        for c in SCORERS:
            print(f'  {c}: {null_tot[c]:,} nulls', flush=True)
        stop('raw score columns must have zero nulls.')
    for c in SCORERS:
        if _RAW[c].shape != (n_actual,):
            stop(f'{c} length {_RAW[c].shape} != ({n_actual},)')
    log(f'PASS1 OK: 3 arrays of length {n_actual:,}, zero nulls ({time.time()-t0:.0f}s).')


# ----------------------------------------------------------------------------- RANK COMPUTE
def _rankcompute(name: str):
    scores = _RAW[name]
    pct = (rankdata(scores, method='average') / _N).astype(np.float32)
    assert pct.min() > 0.0, f'{name}: min pct == 0'
    assert pct.max() <= 1.0 + 1e-6, f'{name}: max pct > 1'
    out = PCT_NPY[name]
    tmp = out.with_suffix(out.suffix + '.tmp')      # ends '.tmp' not '.npy'
    with open(tmp, 'wb') as fh:                      # file handle -> np.save won't append '.npy'
        np.save(fh, pct)
    os.replace(tmp, out)
    return name, {'min': float(pct.min()), 'max': float(pct.max()), 'len': int(pct.size)}


def rank_compute(n_actual: int, force: bool) -> None:
    global _N
    _N = n_actual
    FLAT.mkdir(parents=True, exist_ok=True)
    todo = []
    for name in SCORERS:
        out = PCT_NPY[name]
        if out.exists() and not force:
            a = np.load(out, mmap_mode='r')
            if a.shape == (n_actual,) and a.dtype == np.float32:
                log(f'RANK: {name} pct present ({out.name}), skipping.')
                continue
        todo.append(name)
    if not todo:
        log('RANK: all 3 pct present.')
        return
    log(f'RANK: computing {todo} (rankdata average / N={n_actual:,})...')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(3, len(todo)),
                             mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        for name, st in [fut.result() for fut in
                         [ex.submit(_rankcompute, n) for n in todo]]:
            log(f'  RANK {name}: min={st["min"]:.3e} max={st["max"]:.6f} len={st["len"]:,}')
    log(f'RANK OK ({time.time()-t0:.0f}s).')


# ----------------------------------------------------------------------------- VALIDATION
def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / d) if d else float('nan')


def validate(n_actual: int) -> list[str]:
    log('VALID: orientation / tie / monotonicity checks...')
    L = ['## Scorer-level validation (global, N = {:,})'.format(n_actual), '']
    L.append('| scorer | n_unique_score | n_unique_pct(f32) | min_pct | max_pct | '
             'v2(min raw) | v2(max raw) | corr(score,pct) | identical→identical | non-decreasing |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|')
    for name in SCORERS:
        score = _RAW[name]
        pct = np.asarray(np.load(PCT_NPY[name], mmap_mode='r'))
        order = np.argsort(score, kind='stable')
        s_sorted = score[order]; p_sorted = pct[order]
        is_new = np.empty(score.size, dtype=bool); is_new[0] = True
        np.not_equal(s_sorted[1:], s_sorted[:-1], out=is_new[1:])
        same = ~is_new[1:]
        ident_ok = bool(np.all(p_sorted[1:][same] == p_sorted[:-1][same]))
        assert ident_ok, f'{name}: identical raw score -> different pct.'
        pct_u = p_sorted[is_new]
        n_uscore = int(is_new.sum())
        diffs = np.diff(pct_u)
        nondecr = bool(np.all(diffs >= 0))
        assert nondecr, f'{name}: pct decreases as raw score increases.'
        n_strict = int(np.sum(diffs > 0))
        n_upct = int(np.unique(pct).size)
        assert n_upct <= n_uscore
        c_sp = _pearson(score, pct)
        assert c_sp > 0, f'{name}: corr(score,pct)={c_sp} !> 0'
        L.append('| {} | {:,} | {:,} | {:.3e} | {:.6f} | {:.6f} | {:.6f} | {:+.4f} | {} | {} ({:,} strict) |'
                 .format(name, n_uscore, n_upct, float(pct.min()), float(pct.max()),
                         float(pct_u[0]), float(pct_u[-1]), c_sp,
                         'yes' if ident_ok else 'NO', 'yes' if nondecr else 'NO', n_strict))
        del pct, order, s_sorted, p_sorted, is_new, same, pct_u, diffs
        gc.collect()

    L += ['', '## fasttext tie clusters (the fix) & coarseness', '']
    L.append('| scorer | raw value | cluster size | distinct v2 in cluster | v2 value |')
    L.append('|---|---:|---:|---:|---:|')
    for name in SCORERS:
        score = _RAW[name]
        pct = np.asarray(np.load(PCT_NPY[name], mmap_mode='r'))
        smin, smax = float(score.min()), float(score.max())
        for val in sorted({0.0, 1.0, smin, smax}):
            mask = (score == val)
            cnt = int(mask.sum())
            if cnt == 0:
                L.append(f'| {name} | {val:g} | 0 | — | (no rows) |')
                continue
            up = np.unique(pct[mask])
            v = f'{float(up[0]):.6f}' if up.size == 1 else f'{up.size} VALUES!'
            L.append(f'| {name} | {val:g} | {cnt:,} | {up.size} | {v} |')
        del score, pct; gc.collect()
    log('VALID OK.')
    return L


# ----------------------------------------------------------------------------- PASS 2
def _append_shard(i: int):
    p = shard_path(i)
    tmp = p.with_suffix(p.suffix + '.tmp')
    lo = shard_offset(i)
    hi = lo + shard_rows(i)
    sl = {n: np.asarray(np.load(PCT_NPY[n], mmap_mode='r')[lo:hi], dtype=np.float32) for n in SCORERS}

    pf = pq.ParquetFile(p)
    names = list(pf.schema_arrow.names)
    had = any(V2[n] in names for n in SCORERS)

    if names == EXPECT_SCHEMA and pf.metadata.num_rows == (hi - lo):
        chk = pq.read_table(p, columns=[V2[n] for n in SCORERS], use_threads=False)
        if all(chk.column(V2[n]).null_count == 0 and
               np.array_equal(chk.column(V2[n]).to_numpy(zero_copy_only=False), sl[n])
               for n in SCORERS):
            return dict(i=i, status='skip-verified', nrows=hi - lo,
                        mn={n: float(sl[n].min()) for n in SCORERS},
                        mx={n: float(sl[n].max()) for n in SCORERS})

    told = pq.read_table(p, use_threads=False)
    if told.num_rows != (hi - lo):
        raise RuntimeError(f'shard {i}: {told.num_rows} rows')
    base = told
    drop = [V2[n] for n in SCORERS if V2[n] in base.column_names]
    if drop:
        base = base.drop_columns(drop)
    if list(base.column_names) != ORIG_COLS:
        raise RuntimeError(f'shard {i}: unexpected base cols {base.column_names}')
    new = base
    for n in SCORERS:
        new = new.append_column(V2[n], pa.array(sl[n], type=pa.float32()))
    pq.write_table(new, tmp, compression='zstd')
    os.replace(tmp, p)

    pfn = pq.ParquetFile(p)
    if list(pfn.schema_arrow.names) != EXPECT_SCHEMA:
        raise RuntimeError(f'shard {i}: post-write schema {pfn.schema_arrow.names}')
    if pfn.metadata.num_rows != (hi - lo):
        raise RuntimeError(f'shard {i}: post-write rows {pfn.metadata.num_rows}')
    for n in SCORERS:
        col = pq.read_table(p, columns=[V2[n]], use_threads=False).column(0)
        if col.null_count != 0 or not np.array_equal(col.to_numpy(zero_copy_only=False), sl[n]):
            raise RuntimeError(f'shard {i}: {V2[n]} mismatch/null')
        del col
    for c in ORIG_COLS:
        rc = pq.read_table(p, columns=[c], use_threads=False).column(0)
        if not rc.equals(told.column(c)):
            raise RuntimeError(f'shard {i}: original col {c} changed')
        del rc
    return dict(i=i, status=('overwritten' if had else 'written'), nrows=hi - lo,
                mn={n: float(sl[n].min()) for n in SCORERS},
                mx={n: float(sl[n].max()) for n in SCORERS})


def pass2(workers: int) -> list[dict]:
    log(f'PASS2: appending 3 v2 cols to 200 shards with {workers} workers...')
    res = []
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_append_shard, i): i for i in range(NSHARDS)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res.append(fut.result())
            except Exception as e:   # noqa: BLE001
                stop(f'PASS2 shard {i:05d} failed: {e!r}')
            done += 1
            if done % 25 == 0:
                log(f'  PASS2 {done}/200 ({time.time()-t0:.0f}s)')
    res.sort(key=lambda d: d['i'])
    log(f'PASS2 OK ({time.time()-t0:.0f}s): {dict(Counter(r["status"] for r in res))}')
    return res


# ----------------------------------------------------------------------------- report
def write_report(n_actual, scorer_lines, results, worker_info):
    L = ['# Clean v2 percentile columns — validation report', '',
         f'- SLURM job {os.environ.get("SLURM_JOB_ID","?")}. {worker_info}',
         f'- Dataset: `{CLEAN}` (clean, post-50k-removal). N_actual = **{n_actual:,}** '
         f'(asserted == {N_EXPECT:,}).',
         '- New cols (append-only, float32 [0,1], higher=better): '
         '`fasttext-ranking-v2`, `fineweb-edu-ranking-v2`, `modernbert-ranking-v2`.',
         f'- Flat artifacts: `{FLAT}/rank_*_v2_pct.npy`.',
         '- Recipe: `pct = rankdata(score, "average") / N_actual` (same N for all three).', '',
         '> `max_pct` is reported, not asserted == 1.0: a tie-free scorer\'s top value maps to 1.0, but a '
         'tie cluster at the max (e.g. fasttext score==1) gives `max_pct = (N-(k-1)/2)/N < 1.0` — the intended '
         'tie-aware behaviour. float32 also merges distinct ranks (~1e-8 apart), so `n_unique_pct ≤ n_unique_score`; '
         'this only merges, never splits a tie group, so "identical raw -> identical v2" holds exactly.', '']
    L += scorer_lines
    L += ['', '## Per-shard verification (all 200)']
    nn = all(r['nrows'] == shard_rows(r['i']) for r in results)
    c = Counter(r['status'] for r in results)
    L.append(f'- shards: **{len(results)}/200**, status {dict(c)}; row counts correct: '
             f'**{"PASS" if nn else "FAIL"}**; new cols non-null; schema == 9 orig + 3 v2.')
    L.append('- all 9 original columns verified byte/value-identical after rewrite (`Array.equals` on re-read).')
    L += ['', '| shard | rows | ft_v2 min/max | fw_v2 min/max | mb_v2 min/max | status |',
          '|---:|---:|---|---|---|---|']
    for r in results[:5] + results[-5:]:
        L.append('| {:05d} | {:,} | {:.5f}/{:.5f} | {:.5f}/{:.5f} | {:.5f}/{:.5f} | {} |'.format(
            r['i'], r['nrows'],
            r['mn']['fasttext'], r['mx']['fasttext'],
            r['mn']['fineweb-edu'], r['mx']['fineweb-edu'],
            r['mn']['modernbert'], r['mx']['modernbert'], r['status']))
    REPORT.write_text('\n'.join(L))
    log(f'REPORT -> {REPORT}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-rank', action='store_true')
    ap.add_argument('--mem-per-worker-gb', type=float, default=7.0)
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    workers = max(1, min(cpus, int((mem_gb - 16) // args.mem_per_worker_gb)))
    worker_info = f'cpus={cpus}, mem={mem_gb:.0f}G -> {workers} workers (~{args.mem_per_worker_gb:g} GB/worker).'
    log(f'START: {worker_info}')

    t0 = time.time()
    n_actual = metadata_check()
    assert n_actual == N_EXPECT

    pass1_assemble(n_actual, workers)
    rank_compute(n_actual, force=args.force_rank)
    scorer_lines = validate(n_actual)

    global _RAW
    _RAW = {}; gc.collect()

    results = pass2(workers)
    write_report(n_actual, scorer_lines, results, worker_info)
    log(f'ALL DONE in {time.time()-t0:.0f}s. 3 v2 cols on 200 clean shards; report at {REPORT}')


if __name__ == '__main__':
    main()
