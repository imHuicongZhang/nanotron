#!/usr/bin/env python
"""Assemble the two BASELINE (unmodified) training datasets — physical copy + verification.

No rewriting, no modification: each output is a byte-for-byte copy of shared-top-5B (original
text) + a quality-base block. Shards are copied with shutil.copy2 (preserves schema exactly),
atomically (.tmp + os.replace). Shared-top shards keep their original names; quality-base
shards are renamed `qbase_NNNNN.parquet` to avoid filename collisions.

  DATASET 1 — 10B-base : shared-top-5B  +  10B/quality-base   (target 10,000,000,000)
  DATASET 2 — 15B-base : shared-top-5B  +  15B/quality-base   (target 15,000,000,000)

Verification (read back from the COPIES):
  - total tokens = sum(tokens-llama2 + 1) across ALL copied parquet files,
  - shared-top vs quality-base docs + tokens, combined docs + tokens,
  - combined total vs target (exact number + deviation),
  - ZERO doc_id overlap between shared-top and quality-base.

Writes pretrain/base_assembly_report.md. CPU-only, read-only on the sources.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pp_io import atomic_copy

# ----------------------------------------------------------------------------- paths / constants
BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train')
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
SHARED_SRC = BASE / '10B' / 'shared-top-5B'

# (name, quality-base source dir, target tokens)
DATASETS = [
    ('10B-base', BASE / '10B' / 'quality-base', 10_000_000_000),
    ('15B-base', BASE / '15B' / 'quality-base', 15_000_000_000),
]


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)


# ----------------------------------------------------------------------------- copy + verify worker
def _copy_verify(job):
    """Copy one source shard to its output name, then read back doc_id + tokens-llama2.

    Returns (group, n_docs, token_sum, doc_id_array). token_sum is sum(tokens-llama2) only;
    the +1-per-doc BOS is added when aggregating.
    """
    group, src, out_path = job
    atomic_copy(src, out_path)

    src_rows = pq.ParquetFile(src).metadata.num_rows
    t = pq.read_table(out_path, columns=['doc_id', 'tokens-llama2'], use_threads=False)
    n = t.num_rows
    if n != src_rows:
        raise RuntimeError(f'{os.path.basename(out_path)}: copied {n} rows != source {src_rows}')
    tok = int(t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64).sum())
    did = t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64)
    return group, n, tok, did


def build_jobs(name, qbase_dir, out_dir):
    """Plan the copy: shared-top keeps part_NNNNN.parquet; quality-base -> qbase_NNNNN.parquet."""
    shared = sorted(glob.glob(str(SHARED_SRC / 'part_*.parquet')))
    qbase = sorted(glob.glob(str(qbase_dir / 'part_*.parquet')))
    if not shared:
        stop(f'no shared-top shards under {SHARED_SRC}')
    if not qbase:
        stop(f'no quality-base shards under {qbase_dir}')
    jobs = []
    for s in shared:
        jobs.append(('shared', s, str(out_dir / Path(s).name)))           # keep original name
    for q in qbase:
        idx = Path(q).name.split('_')[1]                                   # NNNNN.parquet
        jobs.append(('qbase', q, str(out_dir / f'qbase_{idx}')))          # prefixed -> no collision
    return jobs, len(shared), len(qbase)


def assemble(name, qbase_dir, target, workers):
    out_dir = PRETRAIN / name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, n_shared_files, n_qbase_files = build_jobs(name, qbase_dir, out_dir)
    log(f'=== {name}: copying {n_shared_files} shared-top + {n_qbase_files} quality-base shards '
        f'-> {out_dir} ===')

    agg = {'shared': dict(docs=0, tok=0, ids=[]), 'qbase': dict(docs=0, tok=0, ids=[])}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as ex:
        futs = {ex.submit(_copy_verify, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                group, n, tok, did = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'{name}: copy/verify {os.path.basename(j[2])} failed: {e!r}')
            a = agg[group]
            a['docs'] += n; a['tok'] += tok; a['ids'].append(did)
            done += 1
            if done % 50 == 0:
                log(f'  {name}: {done}/{len(jobs)} shards ({time.time()-t0:.0f}s)')

    sh, qb = agg['shared'], agg['qbase']
    shared_docs, qbase_docs = sh['docs'], qb['docs']
    shared_tokens = sh['tok'] + shared_docs                 # +1 BOS per doc
    qbase_tokens = qb['tok'] + qbase_docs
    total_docs = shared_docs + qbase_docs
    total_tokens = shared_tokens + qbase_tokens
    deviation = total_tokens - target

    shared_ids = np.concatenate(sh['ids']) if sh['ids'] else np.empty(0, np.int64)
    qbase_ids = np.concatenate(qb['ids']) if qb['ids'] else np.empty(0, np.int64)
    overlap = int(np.intersect1d(shared_ids, qbase_ids).size)

    log(f'  {name}: shared-top {shared_docs:,} docs / {shared_tokens:,} tok; '
        f'quality-base {qbase_docs:,} docs / {qbase_tokens:,} tok')
    log(f'  {name}: TOTAL {total_docs:,} docs / {total_tokens:,} tok; target {target:,}; '
        f'deviation {deviation:+,}')
    log(f'  {name}: doc_id overlap (shared-top ∩ quality-base) = {overlap} (expect 0)')

    return dict(name=name, target=target,
                shared_docs=shared_docs, shared_tokens=shared_tokens,
                qbase_docs=qbase_docs, qbase_tokens=qbase_tokens,
                total_docs=total_docs, total_tokens=total_tokens,
                deviation=deviation, overlap=overlap)


# ----------------------------------------------------------------------------- report
def write_report(rows):
    L = ['# Baseline (unmodified) dataset assembly report', '',
         f'_Generated {time.strftime("%Y-%m-%d %H:%M:%S")}_', '',
         'Physical byte-for-byte copies (no rewriting / no modification). Token budget uses '
         '`sum(tokens-llama2 + 1)` (one leading BOS per doc). Shared-top shards keep their '
         'original `part_NNNNN.parquet` names; quality-base shards are copied as '
         '`qbase_NNNNN.parquet`.', '',
         '| dataset | shared-top docs | shared-top tokens | quality-base docs | '
         'quality-base tokens | total docs | total tokens | target | deviation |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    targets = {10_000_000_000: '10B', 15_000_000_000: '15B'}
    for r in rows:
        L.append(f'| {r["name"]} | {r["shared_docs"]:,} | {r["shared_tokens"]:,} | '
                 f'{r["qbase_docs"]:,} | {r["qbase_tokens"]:,} | {r["total_docs"]:,} | '
                 f'{r["total_tokens"]:,} | {targets.get(r["target"], r["target"])} | '
                 f'{r["deviation"]:+,} |')
    L += ['', '## doc_id overlap (shared-top ∩ quality-base, must be 0)']
    for r in rows:
        flag = '' if r['overlap'] == 0 else '  ⚠️ NON-ZERO'
        L.append(f'- **{r["name"]}**: {r["overlap"]}{flag}')
    L += ['', '## Output layout', '```']
    for r in rows:
        L.append(f'pretrain/{r["name"]}/  (part_NNNNN.parquet = shared-top, '
                 f'qbase_NNNNN.parquet = quality-base)')
    L += ['```', '']
    out = PRETRAIN / 'base_assembly_report.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L))
    log(f'wrote {out}')
    print('\n' + '\n'.join(L[5:]))   # echo the table block to stdout


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'BASE ASSEMBLY START: datasets={[d[0] for d in DATASETS]} workers={workers}')

    rows = []
    bad = []
    for name, qbase_dir, target in DATASETS:
        r = assemble(name, qbase_dir, target, workers)
        rows.append(r)
        if r['overlap'] != 0:
            bad.append(name)

    write_report(rows)
    log('BASE ASSEMBLY DONE.')
    if bad:
        stop(f'doc_id overlap non-zero for: {bad}')


if __name__ == '__main__':
    main()
