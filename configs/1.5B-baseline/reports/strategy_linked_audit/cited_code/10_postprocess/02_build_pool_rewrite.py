#!/usr/bin/env python
"""STEP 2 (rewrite / ReWire-inspired) — build the rewritten-token POOL (pre-FastText).

The ReWire pipeline rewrites broadly, then scores the REWRITTEN output and keeps the best 5B.
This step characterizes the pool that feeds FastText scoring (Step 3): ALL status==2 wiki
rewrites + ALL status==2 distill rewrites combined (no de-dup — the same doc_id rewritten by
two prompts is two different rewritten texts, both kept).

  2a/2b. wiki + distill status==2 docs and tokens (rewritten_tokens + 1).
  2c.    combined pool tokens; report shortfall vs ~15B but PROCEED (we filter from whatever
         is available — wiki alone, ~9.26B, already exceeds the 5B we keep).
  2e.    dual-rewrite doc_ids (same doc_id status==2 in both passes).
  Cross-pass coverage + the pre-FastText funnel (original selection → status2 wiki → status2
  distill → combined pool) are also computed here for the final report.

Read-only on the inputs. CPU-only. Writes _step2_rewrite_summary.json.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pp_io import paired_wiki_status

BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B/rewrite')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')
POOL_TARGET = 15_000_000_000
SHARD_RE = re.compile(r'part_(\d+)\.parquet$')


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


def list_shards(subdir):
    out = []
    for p in sorted(glob.glob(str(BASE / subdir / 'part_*.parquet'))):
        out.append((int(SHARD_RE.search(os.path.basename(p)).group(1)), p))
    return out


def _read_numeric(job):
    subdir, k, path = job
    t = pq.read_table(path, columns=['doc_id', 'status', 'rewritten_tokens', 'tokens-llama2'],
                      use_threads=False)
    return dict(
        k=k, n=t.num_rows,
        doc_id=t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64),
        status=t.column('status').to_numpy(zero_copy_only=False).astype(np.int8),
        rtok=t.column('rewritten_tokens').to_numpy(zero_copy_only=False).astype(np.int64),
        llama=t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64))


def collect_numeric(subdir, shards, workers):
    if not shards:
        return dict(present=[], pos={}, offs=np.zeros(1, np.int64),
                    doc_id=np.empty(0, np.int64), status=np.empty(0, np.int8),
                    rtok=np.empty(0, np.int64), llama=np.empty(0, np.int64),
                    len=np.empty(0, np.int64))
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
               doc_id=cat('doc_id'), status=cat('status'), rtok=cat('rtok'), llama=cat('llama'))
    out['len'] = out['rtok'] + 1
    return out


def cross_pass(wiki, distill):
    if distill['doc_id'].size == 0:
        return dict(paired_docs=0, wiki_docs_without_distill=int(wiki['doc_id'].size),
                    status0_both=0, status0_both_tokens=0, status1_both=0,
                    status1_both_tokens=0, recovered_by_distill=0, status2_wiki_not_distill=0,
                    unique_docs_with_any_status2=int((wiki['status'] == 2).sum()))
    w_for_d = paired_wiki_status(wiki, distill)   # assert per-shard equality; else doc_id join
    d = distill['status']; ll = distill['llama']

    def ct(mask): return int(mask.sum()), int(ll[mask].sum())
    s0n, s0t = ct((w_for_d == 0) & (d == 0))
    s1bn, s1bt = ct((w_for_d == 1) & (d == 1))
    recn, _ = ct((w_for_d == 1) & (d == 2))
    s2wn, _ = ct((w_for_d == 2) & ((d == 0) | (d == 1)))
    wiki_s2_total = int((wiki['status'] == 2).sum())
    extra = int(((d == 2) & (w_for_d != 2)).sum())
    return dict(paired_docs=int(distill['doc_id'].size),
                wiki_docs_without_distill=int(wiki['doc_id'].size - distill['doc_id'].size),
                status0_both=s0n, status0_both_tokens=s0t,
                status1_both=s1bn, status1_both_tokens=s1bt,
                recovered_by_distill=recn, status2_wiki_not_distill=s2wn,
                unique_docs_with_any_status2=wiki_s2_total + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP2 (rewrite) START: pool target ~{POOL_TARGET:,} workers={workers}')

    wiki = collect_numeric('rewritten', list_shards('rewritten'), workers)
    distill = collect_numeric('distill', list_shards('distill'), workers)

    w_s2 = wiki['status'] == 2
    d_s2 = distill['status'] == 2
    wiki_docs = int(w_s2.sum()); wiki_tokens = int(wiki['len'][w_s2].sum())
    distill_docs = int(d_s2.sum()); distill_tokens = int(distill['len'][d_s2].sum())
    pool_docs = wiki_docs + distill_docs
    pool_tokens = wiki_tokens + distill_tokens
    dual = int(np.intersect1d(wiki['doc_id'][w_s2], distill['doc_id'][d_s2]).size)

    log(f'2a wiki status2: {wiki_docs:,} docs / {wiki_tokens:,} tok')
    log(f'2b distill status2: {distill_docs:,} docs / {distill_tokens:,} tok')
    log(f'2c combined pool: {pool_docs:,} docs / {pool_tokens:,} tok '
        f'(target ~{POOL_TARGET:,})')
    if pool_tokens < POOL_TARGET:
        log(f'  NOTE: pool {pool_tokens:,} < 15B by {POOL_TARGET-pool_tokens:,} '
            f'(distill {len(distill["present"])} shards). Proceeding — top-5B filter only '
            f'needs the pool to exceed 5B.')
    log(f'2e dual-rewrite doc_ids (status2 in both): {dual:,}')

    cov = cross_pass(wiki, distill)
    sel_docs = int(wiki['doc_id'].size)
    sel_tokens = int((wiki['llama'] + 1).sum())          # original selection (all wiki rows)
    funnel = [
        dict(stage='original selection (rewrite, ~20B)', docs=sel_docs, tokens=sel_tokens),
        dict(stage='status=2 wiki', docs=wiki_docs, tokens=wiki_tokens),
        dict(stage='status=2 distill', docs=distill_docs, tokens=distill_tokens),
        dict(stage='combined pool (pre-FastText)', docs=pool_docs, tokens=pool_tokens),
    ]

    summary = dict(
        setting='rewrite', pool_target=POOL_TARGET,
        wiki_shards=len(wiki['present']), distill_shards=len(distill['present']),
        pool_docs=pool_docs, pool_tokens=pool_tokens,
        pool_from_wikipedia=dict(docs=wiki_docs, tokens=wiki_tokens),
        pool_from_distill=dict(docs=distill_docs, tokens=distill_tokens),
        dual_rewrite_doc_ids=dual,
        pool_shortfall_vs_15B=max(0, POOL_TARGET - pool_tokens),
        coverage=cov, funnel_pre_fasttext=funnel)
    (HERE / '_step2_rewrite_summary.json').write_text(json.dumps(summary, indent=2))
    log('STEP2 DONE; wrote _step2_rewrite_summary.json')


if __name__ == '__main__':
    main()
