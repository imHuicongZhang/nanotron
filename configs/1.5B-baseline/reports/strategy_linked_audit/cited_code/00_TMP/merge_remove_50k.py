#!/usr/bin/env python
"""Option B — clean MERGE + physical removal of the 50,838 contaminated rows into
ONE canonical dataset (no v2, no old int ranks). Writes a NEW dir only.

Pipeline (single SLURM job, idempotent, atomic per-shard):

  PHASE A  ALIGNMENT GATE (must pass before any write)
    - assert all 4 "full" pool dirs are 100M rows / 200 x 500k.
    - hash text[:256] at 100 fixed-seed random flat positions; ALL dirs must agree
      at every position (proves the family is row-aligned, so the removal .npy
      positions — computed on 3_scorers_full_scored — apply to the merge source).

  PHASE B  MERGE + REMOVE  (per NEW output shard, parallel)
    - data source: 3_scorers_full_scored_topic (text,url,metadata,3 scores,topic).
    - removal set: match_50k_prefix4000.npy verbatim (50,838 sorted flat positions).
    - survivors kept in ORIGINAL flat-position order, re-packed to 500k/shard
      (last shard 449,162). NEW contiguous doc_id 0..99,949,161; orig_doc_id kept.

  PHASE C  VERIFY
    - total surviving == 99,949,162; removed positions ∩ output orig_doc_id == ∅;
      doc_id == 0..N-1; schema == whitelist; no nulls in score/topic cols;
      100-row value-identical spot-check against the source.

Whitelist output schema (exact order):
    orig_doc_id int64, doc_id int64, text, url, metadata,
    fasttext, fineweb-edu, modernbert, topic
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- paths / constants
BASE     = Path('/scratch/bvandur1/zhuicon1/data_rewrite')
SRC      = BASE / '3_scorers_full_scored_topic'          # single data source
OUT_DIR  = BASE / '6_merged_clean'
MATCH_NPY = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/01_explore/match_50k_prefix4000.npy')

ALIGN_DIRS = [BASE / '3_scorers_full', BASE / '3_scorers_full_scored',
              BASE / '3_scorers_full_scored_topic', BASE / '4_scorers_full_ranked']

ROWS_PER_FILE = 500_000
N_FILES_IN    = 200
N_EXPECT      = 100_000_000
N_REMOVE      = 50_838
N_TOTAL_OUT   = N_EXPECT - N_REMOVE      # 99,949,162
ROWS_PER_OUT  = 500_000
N_OUT_SHARDS  = (N_TOTAL_OUT + ROWS_PER_OUT - 1) // ROWS_PER_OUT   # 200

DATA_COLS = ['text', 'url', 'metadata', 'fasttext', 'fineweb-edu', 'modernbert', 'topic']
WHITELIST = ['orig_doc_id', 'doc_id'] + DATA_COLS
SCORE_TOPIC_COLS = ['fasttext', 'fineweb-edu', 'modernbert', 'topic']

ALIGN_SEED = 20260531
SPOT_SEED  = 76543210
N_ALIGN = 100
N_SPOT  = 100

# module globals shared with forked workers (read-only)
_REMOVED: np.ndarray = np.empty(0, dtype=np.int64)   # sorted removed flat positions
_O: np.ndarray = np.empty(0, dtype=np.int64)          # prefix surviving offsets, len 201


def src_shard(i: int) -> Path:
    return SRC / f'dclm_refinedweb_sample_{i:05d}.parquet'


def out_shard(k: int) -> Path:
    return OUT_DIR / f'merged_clean_{k:05d}.parquet'


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def stop(msg: str) -> 'NoReturn':
    print(f'\n*** STOP: {msg}\n', flush=True)
    sys.exit(2)


def _init_worker() -> None:
    pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- PHASE A: alignment
def _read_text_rows(args):
    """Return {flat_pos: sha1(text[:256])} for the needed rows of one (dir, shard)."""
    d, f, rows = args
    p = Path(d) / f'dclm_refinedweb_sample_{f:05d}.parquet'
    col = pq.read_table(p, columns=['text'], use_threads=False).column('text')
    out = {}
    for r in rows:
        s = col[r].as_py()
        h = hashlib.sha1((s[:256] if s else '').encode('utf-8')).hexdigest()
        out[f * ROWS_PER_FILE + r] = h
    return str(d), out


def alignment_gate(workers: int) -> None:
    log('PHASE A: alignment gate (row counts + text[:256] hashes at 100 positions)...')
    # row-count / shard structure for every align dir
    for d in ALIGN_DIRS:
        shards = sorted(d.glob('*.parquet'))
        if len(shards) != N_FILES_IN:
            stop(f'{d}: {len(shards)} shards != {N_FILES_IN}')
        counts = [pq.ParquetFile(p).metadata.num_rows for p in shards]
        if set(counts) != {ROWS_PER_FILE} or sum(counts) != N_EXPECT:
            stop(f'{d}: not 200x500k=100M (distinct counts={sorted(set(counts))}, total={sum(counts)})')
    log(f'  row structure OK for all {len(ALIGN_DIRS)} dirs (200 x 500k = 100M each).')

    rng = np.random.default_rng(ALIGN_SEED)
    positions = np.sort(rng.choice(N_EXPECT, size=N_ALIGN, replace=False))
    by_shard: dict[int, list[int]] = defaultdict(list)
    for p in positions:
        by_shard[int(p) // ROWS_PER_FILE].append(int(p) % ROWS_PER_FILE)

    tasks = [(str(d), f, rows) for d in ALIGN_DIRS for f, rows in by_shard.items()]
    per_dir_hashes: dict[str, dict[int, str]] = {str(d): {} for d in ALIGN_DIRS}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        for d, out in ex.map(_read_text_rows, tasks):
            per_dir_hashes[d].update(out)

    ref = per_dir_hashes[str(ALIGN_DIRS[0])]
    mism = 0
    for pos in positions:
        pos = int(pos)
        hs = {d: per_dir_hashes[d].get(pos) for d in per_dir_hashes}
        if len(set(hs.values())) != 1:
            mism += 1
            if mism <= 5:
                print(f'    MISMATCH at pos {pos}: {hs}', flush=True)
    if mism:
        stop(f'alignment FAILED: {mism}/{N_ALIGN} positions disagree across dirs. NEVER merge by position.')
    log(f'  alignment OK: all {N_ALIGN} positions agree across all {len(ALIGN_DIRS)} dirs.')


# ----------------------------------------------------------------------------- plan
def build_plan() -> list[dict]:
    """Per NEW output shard k: contributing (old_f, lo_rank, hi_rank) segments."""
    global _REMOVED, _O
    removed = np.load(MATCH_NPY).astype(np.int64)
    if removed.size != N_REMOVE or not np.all(np.diff(removed) > 0):
        stop(f'removal npy bad: size={removed.size}, sorted/unique check failed.')
    if removed.min() < 0 or removed.max() >= N_EXPECT:
        stop('removal npy positions out of range.')
    _REMOVED = removed

    bounds = np.searchsorted(removed, np.arange(0, N_EXPECT + 1, ROWS_PER_FILE))  # len 201
    removed_per_shard = np.diff(bounds)
    surviving = ROWS_PER_FILE - removed_per_shard
    O = np.empty(N_FILES_IN + 1, dtype=np.int64)
    O[0] = 0
    np.cumsum(surviving, out=O[1:])
    if O[-1] != N_TOTAL_OUT:
        stop(f'prefix-sum total {O[-1]:,} != {N_TOTAL_OUT:,}')
    _O = O

    plan = []
    for k in range(N_OUT_SHARDS):
        a = k * ROWS_PER_OUT
        b = min((k + 1) * ROWS_PER_OUT, N_TOTAL_OUT)
        f_start = int(np.searchsorted(O, a, side='right') - 1)
        f_end   = int(np.searchsorted(O, b - 1, side='right') - 1)
        segs = []
        for f in range(f_start, f_end + 1):
            lo_new = max(a, int(O[f]))
            hi_new = min(b, int(O[f + 1]))
            if hi_new <= lo_new:
                continue
            segs.append((f, lo_new - int(O[f]), hi_new - int(O[f])))  # (old_f, lo_rank, hi_rank)
        plan.append(dict(k=k, a=a, b=b, nrows=b - a, segs=segs))
    return plan


# ----------------------------------------------------------------------------- PHASE B: merge worker
def _surviving_row_idx(f: int) -> np.ndarray:
    """Sorted local row indices (0..499999) that survive in old shard f, in order."""
    lo = np.searchsorted(_REMOVED, f * ROWS_PER_FILE)
    hi = np.searchsorted(_REMOVED, (f + 1) * ROWS_PER_FILE)
    rel = _REMOVED[lo:hi] - f * ROWS_PER_FILE
    if rel.size == 0:
        return np.arange(ROWS_PER_FILE, dtype=np.int64)
    mask = np.ones(ROWS_PER_FILE, dtype=bool)
    mask[rel] = False
    return np.flatnonzero(mask).astype(np.int64)


def _verify_table(tbl, a: int, b: int) -> None:
    if list(tbl.schema.names) != WHITELIST:
        raise RuntimeError(f'schema {tbl.schema.names} != whitelist')
    if tbl.num_rows != b - a:
        raise RuntimeError(f'rows {tbl.num_rows} != {b - a}')
    did = tbl.column('doc_id').to_numpy(zero_copy_only=False)
    if not np.array_equal(did, np.arange(a, b, dtype=np.int64)):
        raise RuntimeError('doc_id not contiguous a..b')
    oid = tbl.column('orig_doc_id').to_numpy(zero_copy_only=False)
    if not np.all(np.diff(oid) > 0):
        raise RuntimeError('orig_doc_id not strictly increasing')
    # none of the surviving orig_doc_id may be in the removal set
    idx = np.searchsorted(_REMOVED, oid)
    hit = (idx < _REMOVED.size) & (_REMOVED[np.clip(idx, 0, _REMOVED.size - 1)] == oid)
    if hit.any():
        raise RuntimeError(f'{int(hit.sum())} removed positions leaked into output')
    for c in SCORE_TOPIC_COLS:
        if tbl.column(c).null_count != 0:
            raise RuntimeError(f'nulls in {c}')


def _merge_one(item: dict):
    k, a, b, segs = item['k'], item['a'], item['b'], item['segs']
    p = out_shard(k)
    tmp = p.with_suffix(p.suffix + '.tmp')

    # idempotent skip if a correct output already exists
    if p.exists():
        try:
            pf = pq.ParquetFile(p)
            if list(pf.schema_arrow.names) == WHITELIST and pf.metadata.num_rows == (b - a):
                t = pq.read_table(p, columns=['doc_id', 'orig_doc_id'], use_threads=False)
                did = t.column('doc_id').to_numpy(zero_copy_only=False)
                oid = t.column('orig_doc_id').to_numpy(zero_copy_only=False)
                idx = np.searchsorted(_REMOVED, oid)
                hit = (idx < _REMOVED.size) & (_REMOVED[np.clip(idx, 0, _REMOVED.size - 1)] == oid)
                if (np.array_equal(did, np.arange(a, b, dtype=np.int64))
                        and np.all(np.diff(oid) > 0) and not hit.any()):
                    return dict(k=k, nrows=b - a, orig_min=int(oid[0]), orig_max=int(oid[-1]),
                                status='skip-verified')
        except Exception:
            pass  # fall through and rebuild

    subs = []
    orig_chunks = []
    for (f, lo_rank, hi_rank) in segs:
        surv = _surviving_row_idx(f)
        take = surv[lo_rank:hi_rank]
        tbl = pq.read_table(src_shard(f), columns=DATA_COLS, use_threads=False)
        subs.append(tbl.take(pa.array(take)))
        orig_chunks.append(f * ROWS_PER_FILE + take)
        del tbl
    data = pa.concat_tables(subs) if len(subs) > 1 else subs[0]
    orig_ids = np.concatenate(orig_chunks).astype(np.int64)
    doc_ids = np.arange(a, b, dtype=np.int64)

    arrays = [pa.array(orig_ids, type=pa.int64()), pa.array(doc_ids, type=pa.int64())]
    names = ['orig_doc_id', 'doc_id']
    for c in DATA_COLS:
        arrays.append(data.column(c))
        names.append(c)
    final = pa.table(arrays, names=names)

    _verify_table(final, a, b)
    pq.write_table(final, tmp, compression='zstd')
    os.replace(tmp, p)

    # re-open & re-verify the written file
    chk = pq.read_table(p, use_threads=False)
    _verify_table(chk, a, b)
    return dict(k=k, nrows=b - a, orig_min=int(orig_ids[0]), orig_max=int(orig_ids[-1]),
                status='written')


def phase_b(plan: list[dict], workers: int) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f'PHASE B: merge+remove into {OUT_DIR} ({N_OUT_SHARDS} shards) with {workers} workers...')
    results = []
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_merge_one, item): item['k'] for item in plan}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:   # noqa: BLE001
                stop(f'PHASE B shard {k:05d} failed: {e!r}')
            done += 1
            if done % 25 == 0:
                log(f'  PHASE B {done}/{N_OUT_SHARDS} ({time.time()-t0:.0f}s)')
    results.sort(key=lambda d: d['k'])
    log(f'PHASE B OK ({time.time()-t0:.0f}s): {dict(Counter(r["status"] for r in results))}')
    return results


# ----------------------------------------------------------------------------- PHASE C: global verify
def _read_orig(k: int):
    return k, pq.read_table(out_shard(k), columns=['orig_doc_id'], use_threads=False
                            ).column('orig_doc_id').to_numpy(zero_copy_only=False)


def phase_c_global(workers: int) -> None:
    log('PHASE C: global verify (orig_doc_id == complement of removal set)...')
    all_orig = np.empty(N_TOTAL_OUT, dtype=np.int64)
    pos = 0
    parts = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        for k, arr in ex.map(_read_orig, range(N_OUT_SHARDS)):
            parts[k] = arr
    for k in range(N_OUT_SHARDS):
        arr = parts[k]
        all_orig[pos:pos + arr.size] = arr
        pos += arr.size
    if pos != N_TOTAL_OUT:
        stop(f'assembled {pos:,} orig_doc_id != {N_TOTAL_OUT:,}')
    expected = np.setdiff1d(np.arange(N_EXPECT, dtype=np.int64), _REMOVED, assume_unique=True)
    if expected.size != N_TOTAL_OUT:
        stop(f'complement size {expected.size:,} != {N_TOTAL_OUT:,}')
    if not np.array_equal(all_orig, expected):
        stop('output orig_doc_id (in order) != sorted complement of removal set.')
    log(f'  GLOBAL OK: {N_TOTAL_OUT:,} survivors, exactly the complement of the 50,838 removed; '
        'order preserved; 0 removed positions leaked.')


def spot_check(workers: int) -> int:
    log(f'PHASE C: {N_SPOT}-row value-identical spot-check vs source...')
    rng = np.random.default_rng(SPOT_SEED)
    gids = np.sort(rng.choice(N_TOTAL_OUT, size=N_SPOT, replace=False))
    # group target rows by output shard
    by_out: dict[int, list[int]] = defaultdict(list)
    for g in gids:
        by_out[int(g) // ROWS_PER_OUT].append(int(g) % ROWS_PER_OUT)

    # read needed output rows -> (orig_doc_id, dict of data col values)
    out_rows = {}   # global_id -> (orig_doc_id, {col: val})
    for k, rows in by_out.items():
        t = pq.read_table(out_shard(k), use_threads=False)
        for r in rows:
            g = k * ROWS_PER_OUT + r
            rec = {c: t.column(c)[r].as_py() for c in DATA_COLS}
            out_rows[g] = (int(t.column('orig_doc_id')[r].as_py()), rec)

    # group source lookups by source shard
    by_src: dict[int, list[tuple[int, int]]] = defaultdict(list)  # f -> [(orig, g)]
    for g, (orig, _) in out_rows.items():
        by_src[orig // ROWS_PER_FILE].append((orig % ROWS_PER_FILE, g))

    mismatches = 0
    for f, lst in by_src.items():
        t = pq.read_table(src_shard(f), columns=DATA_COLS, use_threads=False)
        for orow, g in lst:
            src_rec = {c: t.column(c)[orow].as_py() for c in DATA_COLS}
            if src_rec != out_rows[g][1]:
                mismatches += 1
                if mismatches <= 5:
                    print(f'    SPOT MISMATCH g={g} orig={out_rows[g][0]}', flush=True)
    if mismatches:
        stop(f'spot-check FAILED: {mismatches}/{N_SPOT} rows differ from source.')
    log(f'  SPOT OK: all {N_SPOT} output rows value-identical to their source (matched via orig_doc_id).')
    return N_SPOT


# ----------------------------------------------------------------------------- report
def write_report(results: list[dict], worker_info: str) -> None:
    rep = OUT_DIR.parent / 'scripts' / 'merge_remove_50k_report.md'
    rep = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/00_TMP/merge_remove_50k_report.md')
    total = sum(r['nrows'] for r in results)
    L = []
    L.append('# Option B — merged clean dataset (50,838 rows physically removed)')
    L.append('')
    L.append(f'- SLURM job {os.environ.get("SLURM_JOB_ID","?")}. {worker_info}')
    L.append(f'- Output dir: `{OUT_DIR}`  ({len(results)} shards)')
    L.append(f'- Total rows: **{total:,}** (must be {N_TOTAL_OUT:,}: '
             f'**{"PASS" if total==N_TOTAL_OUT else "FAIL"}**)')
    L.append(f'- Removed: **{N_REMOVE:,}** rows (from `match_50k_prefix4000.npy`, verbatim) = 100,000,000 − {N_TOTAL_OUT:,}.')
    L.append('- Source: `3_scorers_full_scored_topic` (single, row-aligned). Alignment gate passed across '
             'all 4 full pool dirs (100 positions).')
    L.append('')
    L.append('## Final schema (whitelist, exact order)')
    L.append('```')
    L.append('orig_doc_id  int64        # original file_idx*500000 + row_idx')
    L.append('doc_id       int64        # new contiguous 0 .. {0:,}'.format(N_TOTAL_OUT - 1))
    L.append('text         large_string')
    L.append('url          large_string')
    L.append('metadata     large_string # raw WARC JSON blob, kept whole')
    L.append('fasttext     float')
    L.append('fineweb-edu  float')
    L.append('modernbert   float')
    L.append('topic        large_string')
    L.append('```')
    L.append('Dropped (per spec): old int ranks ×3, stale v2 ×3, is_50k_contam.')
    L.append('')
    L.append('## Sharding')
    L.append(f'- {N_OUT_SHARDS} shards, 500,000 rows each except the last.')
    L.append(f'- shard 00000..{N_OUT_SHARDS-2:05d}: 500,000 rows; shard {N_OUT_SHARDS-1:05d}: '
             f'{results[-1]["nrows"]:,} rows.')
    L.append('')
    L.append('## Verification')
    L.append('- total survivors == 99,949,162 ✅')
    L.append('- output `orig_doc_id` (in order) == sorted complement of the 50,838 removed positions '
             '(0 leaked, order preserved) ✅')
    L.append('- `doc_id` == 0..N−1 contiguous; schema == whitelist; no nulls in score/topic cols ✅')
    L.append(f'- {N_SPOT}-row value-identical spot-check vs source passed ✅')
    L.append('')
    L.append('## Now-stale artifacts (informational — NOT modified by this task)')
    L.append('These were computed on the OLD 100M indexing and do not match the new 99,949,162-row doc_id:')
    L.append('- `4_scorers_full_ranked/` v2 columns (`*-ranking-v2`) and old int ranks — to be recomputed separately.')
    L.append('- `3_scorers_full_flat/`: `rank_*_v2_pct.npy`, `rank_*.npy`, `mask_*.npy`, `selected_*.npy`, '
             '`char_length.npy`, `fasttext/fineweb/modernbert.npy` (all 100M-indexed).')
    L.append('- `01_explore/` selection/mask arrays keyed to 100M flat positions '
             '(`mask_50k_prefix4000.npy` etc. — still valid as a record of *which* positions were removed).')
    L.append('- `is_50k_contam` flag in `4_scorers_full_ranked/` (rows now physically gone here).')
    L.append('')
    L.append('## Per-shard (first/last 5)')
    L.append('| shard | rows | orig_doc_id min..max | status |')
    L.append('|---:|---:|---|---|')
    for r in results[:5] + results[-5:]:
        L.append(f'| {r["k"]:05d} | {r["nrows"]:,} | {r["orig_min"]:,} .. {r["orig_max"]:,} | {r["status"]} |')
    rep.write_text('\n'.join(L))
    log(f'REPORT -> {rep}')


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mem-per-worker-gb', type=float, default=7.0)
    ap.add_argument('--skip-align', action='store_true', help='(debug) skip Phase A')
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    workers = max(1, min(cpus, int((mem_gb - 16) // args.mem_per_worker_gb)))
    worker_info = f'cpus={cpus}, mem={mem_gb:.0f}G -> {workers} workers (~{args.mem_per_worker_gb:g} GB/worker).'
    log(f'START: {worker_info}')

    t0 = time.time()
    if not args.skip_align:
        alignment_gate(workers)

    plan = build_plan()
    log(f'plan: {N_OUT_SHARDS} output shards; survivors={N_TOTAL_OUT:,}; '
        f'max segs/shard={max(len(p["segs"]) for p in plan)}')

    results = phase_b(plan, workers)

    if sum(r['nrows'] for r in results) != N_TOTAL_OUT:
        stop('total rows mismatch after Phase B.')
    phase_c_global(workers)
    spot_check(workers)
    write_report(results, worker_info)
    log(f'ALL DONE in {time.time()-t0:.0f}s. Clean merged dataset at {OUT_DIR} '
        f'({N_TOTAL_OUT:,} rows, {N_OUT_SHARDS} shards).')


if __name__ == '__main__':
    main()
