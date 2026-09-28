"""Step 2c (prefix-L) — locate the 50k ModernBERT-training docs inside the
100M DCLM-RefinedWeb pool by matching on the first L characters of `text`.

L is reconstructed identically on both sides of the join. The 50k side has
`annotation_text = str(text)[:T]` (T = truncate_chars from selected_50k_report.json).
The pool side has full-length `text`. We use a long prefix L = min(4000, T) so
matching reproduces what the labeller actually saw without under-matching docs
whose full text exceeds T.

NO normalization, NO whitespace fixing, NO case fold — annotation_text was a
raw str.slice; we mirror that exactly. Encode utf-8 once, sha1-hash, compare
on bytes.

This run RECORDS positions only — no parquet/.npy is modified.

Outputs (in this directory):
  match_50k_prefix{L}.npy     int32, sorted, flat positions of matched pool rows
                              (flat_position = file_idx * 500_000 + row_idx)
  mask_50k_prefix{L}.npy      bool, length 100_000_000, True at matched positions
  match_50k_prefix{L}_report.json  per-50k-key match counts + stats summary

Parallelism: multiprocessing.Pool, workers = os.cpu_count().
"""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ── Paths / constants ─────────────────────────────────────────────────────────

POOL_DIR    = Path('/scratch/bvandur1/zhuicon1/data_rewrite/3_scorers_full_scored')
REPORT_50K  = Path('/scratch/bvandur1/zhuicon1/dataset/dclm-refinedweb-50k/selected_50k_report.json')
JSONL_50K   = Path('/scratch/bvandur1/zhuicon1/dataset/dclm-refinedweb-50k/selected_50k_for_claude.jsonl')
OUT_DIR     = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/01_explore')

ROWS_PER_FILE = 500_000
N_FILES       = 200
N_TOTAL       = ROWS_PER_FILE * N_FILES  # 100_000_000


def sha1_bytes(s: str) -> bytes:
    return hashlib.sha1(s.encode('utf-8')).digest()


# ── Worker (one parquet per call) ─────────────────────────────────────────────

# Global injected at worker init so pool tasks don't re-pickle the dict.
_WORKER_HASH_TO_50K_IDXS: dict[bytes, list[int]] = {}
_WORKER_L: int = 0


def _worker_init(hash_to_50k_idxs: dict[bytes, list[int]], L: int):
    global _WORKER_HASH_TO_50K_IDXS, _WORKER_L
    _WORKER_HASH_TO_50K_IDXS = hash_to_50k_idxs
    _WORKER_L = L


def _process_file(args) -> tuple[int, np.ndarray, Counter]:
    """Scan one parquet, return (file_idx, matched_row_idx_array, per-key match Counter).

    Counter keys are the sha1 bytes (same dict the 50k side uses).
    """
    file_idx, path = args
    keys = _WORKER_HASH_TO_50K_IDXS
    L    = _WORKER_L
    matched_rows: list[int] = []
    matched_hash_counter: Counter = Counter()
    # Stream the `text` column in small record batches instead of materializing
    # the whole 500k-string column as one Python list. This bounds per-worker
    # memory (only ~batch_size strings are alive at once) so the cgroup never
    # hits its limit. The matching is byte-identical to the previous full-column
    # version: same sha1_bytes(s[:L]) membership test, same row indexing.
    j = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=20_000, columns=['text']):
        for s in batch.column('text').to_pylist():
            if s:
                h = sha1_bytes(s[:L])
                if h in keys:
                    matched_rows.append(j)
                    matched_hash_counter[h] += 1
            j += 1
    return file_idx, np.array(matched_rows, dtype=np.int32), matched_hash_counter


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Step 0 — discover T and L.
    with open(REPORT_50K) as f:
        report = json.load(f)
    T = int(report['truncation']['truncate_chars'])
    docs_truncated = int(report['truncation']['docs_truncated'])
    L = min(4000, T)
    print(f'[step 0] truncate_chars (T): {T}')
    print(f'[step 0] docs_truncated (full-text match would miss): {docs_truncated:,}')
    print(f'[step 0] match length L = min(4000, T) = {L}', flush=True)
    if L > T:
        sys.exit(f'L={L} would exceed T={T}; refusing to proceed.')

    # ── Step A — build 50k-side hash dict from annotation_text[:L] ──────────
    t0 = time.time()
    hash_to_50k_idxs: dict[bytes, list[int]] = defaultdict(list)
    with open(JSONL_50K) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            ann = rec['annotation_text']  # already str(text)[:T]
            h = sha1_bytes(ann[:L])
            hash_to_50k_idxs[h].append(i)
    n_keys = len(hash_to_50k_idxs)
    n_50k = sum(len(v) for v in hash_to_50k_idxs.values())
    collisions_50k_side = n_50k - n_keys
    print(f'[50k side] rows: {n_50k:,}  unique prefix-L hashes: {n_keys:,}  '
          f'internal collisions: {collisions_50k_side:,} ({time.time()-t0:.1f}s)', flush=True)

    # ── Step B — scan pool in parallel ──────────────────────────────────────
    files = sorted(POOL_DIR.glob('dclm_refinedweb_sample_*.parquet'))
    assert len(files) == N_FILES, f'expected {N_FILES} files, got {len(files)}'
    tasks = [(i, str(p)) for i, p in enumerate(files)]

    # Default = os.cpu_count() (unchanged). STEP2C_WORKERS lets the SLURM wrapper
    # cap concurrency to fit the job's memory (each worker holds a full shard's
    # text column, ~3.5 GB). This is a resource/parallelism knob ONLY — it does
    # not affect which rows match (key derivation is untouched), so the recorded
    # positions are identical regardless of worker count.
    n_workers = int(os.environ.get('STEP2C_WORKERS', '0') or 0) or (os.cpu_count() or 8)
    print(f'[scan] launching multiprocessing.Pool workers={n_workers}', flush=True)
    t0 = time.time()
    matched_pos_per_file: dict[int, np.ndarray] = {}
    per_hash_match_counter: Counter = Counter()
    with mp.Pool(processes=n_workers,
                 initializer=_worker_init,
                 initargs=(dict(hash_to_50k_idxs), L)) as pool:
        for k, (file_idx, idxs, hash_counter) in enumerate(
                pool.imap_unordered(_process_file, tasks, chunksize=2), 1):
            matched_pos_per_file[file_idx] = idxs
            per_hash_match_counter.update(hash_counter)
            if k % 25 == 0:
                tot_so_far = sum(v.size for v in matched_pos_per_file.values())
                print(f'  {k:3d}/{N_FILES}  matches_so_far={tot_so_far:,}  ({time.time()-t0:.1f}s)',
                      flush=True)
    elapsed = time.time() - t0
    print(f'[scan] done in {elapsed:.1f}s', flush=True)

    # ── Step C — assemble outputs ───────────────────────────────────────────
    # Sorted int32 flat positions.
    flat_positions: list[int] = []
    for fi in sorted(matched_pos_per_file):
        idxs = matched_pos_per_file[fi]
        base = fi * ROWS_PER_FILE
        flat_positions.extend((int(j) + base) for j in idxs.tolist())
    flat_positions.sort()
    flat_positions_arr = np.array(flat_positions, dtype=np.int32)
    total_matches = flat_positions_arr.size

    # Bool mask 100M
    mask = np.zeros(N_TOTAL, dtype=bool)
    mask[flat_positions_arr] = True

    # ── Step D — per-50k-key match-count breakdown ──────────────────────────
    # Map per-hash counter back to 50k rows; if a hash has K 50k-side rows
    # they all share the same per-hash match count.
    keys_matched_0    = 0
    keys_matched_1    = 0
    keys_matched_many = 0
    pool_per_50k_row = []
    for h, idxs50k in hash_to_50k_idxs.items():
        c = per_hash_match_counter.get(h, 0)
        if   c == 0: keys_matched_0    += 1
        elif c == 1: keys_matched_1    += 1
        else:        keys_matched_many += 1
        for _ in idxs50k:
            pool_per_50k_row.append(c)
    pool_per_50k_row = np.array(pool_per_50k_row, dtype=np.int64)

    # ── Step E — write artifacts ────────────────────────────────────────────
    match_npy = OUT_DIR / f'match_50k_prefix{L}.npy'
    mask_npy  = OUT_DIR / f'mask_50k_prefix{L}.npy'
    rep_json  = OUT_DIR / f'match_50k_prefix{L}_report.json'

    np.save(match_npy, flat_positions_arr)
    np.save(mask_npy,  mask)

    summary = {
        'T_truncate_chars':           T,
        'docs_truncated_in_50k':      docs_truncated,
        'L_used':                     L,
        'normalization_policy':       'raw str.slice, utf-8 encode, sha1 — no whitespace/casefold',
        'n_50k_rows':                 n_50k,
        'n_unique_50k_prefix_hashes': n_keys,
        'collisions_within_50k_side': collisions_50k_side,
        'pool_total_rows_scanned':    N_TOTAL,
        'pool_total_matched_rows':    int(total_matches),
        'prior_prefix500_overmatch':  85_301,
        'keys_matched_0_pool_rows':   int(keys_matched_0),
        'keys_matched_exactly_1':     int(keys_matched_1),
        'keys_matched_more_than_1':   int(keys_matched_many),
        'pool_match_count_per_50k_row_summary': {
            'min':    int(pool_per_50k_row.min()),
            'max':    int(pool_per_50k_row.max()),
            'mean':   float(pool_per_50k_row.mean()),
            'median': float(np.median(pool_per_50k_row)),
        },
        'artifacts': {
            'matched_positions_int32': str(match_npy),
            'matched_mask_bool':       str(mask_npy),
        },
        'elapsed_seconds_scan': elapsed,
        'workers':              n_workers,
    }
    rep_json.write_text(json.dumps(summary, indent=2))

    print('\n=== SUMMARY ===')
    print(f'  T (truncate_chars):                {T}')
    print(f'  docs_truncated_in_50k:             {docs_truncated:,}')
    print(f'  L used:                            {L}')
    print(f'  50k rows:                          {n_50k:,}')
    print(f'  unique 50k prefix-L hashes:        {n_keys:,}')
    print(f'  50k-side internal key collisions:  {collisions_50k_side:,} '
          f'(prior at L=500 was 14)')
    print(f'  pool rows scanned:                 {N_TOTAL:,}')
    print(f'  pool rows matched (prefix-L):      {total_matches:,}')
    print(f'  prior prefix-500 over-match was:   85,301')
    print(f'  50k-keys matched 0 pool rows:      {keys_matched_0:,}')
    print(f'  50k-keys matched exactly 1 pool:   {keys_matched_1:,}')
    print(f'  50k-keys matched >1 pool rows:     {keys_matched_many:,}   <-- duplicates in the pool')
    print(f'  per-50k-row pool match count: min={pool_per_50k_row.min()} '
          f'max={pool_per_50k_row.max()} mean={pool_per_50k_row.mean():.3f}')
    print(f'\n  saved: {match_npy}')
    print(f'  saved: {mask_npy}')
    print(f'  saved: {rep_json}')


if __name__ == '__main__':
    main()
