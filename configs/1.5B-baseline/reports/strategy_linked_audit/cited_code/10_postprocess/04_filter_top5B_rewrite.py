#!/usr/bin/env python
"""STEP 4 (rewrite / ReWire-inspired) — keep the top 5B by rewritten_fasttext_score.

Global sort of the ENTIRE scored pool by `rewritten_fasttext_score` DESC (raw P(hq)) — no
preference for wiki vs distill; the best-scoring rewrites win regardless of prompt. Keep docs
until cumulative (rewritten_tokens + 1) reaches 5,000,000,000 (last doc whole). Ordering by the
raw score is identical to ordering by rewritten_fasttext_ranking_v2 (monotonic).

Reports: kept docs/tokens; wiki vs distill split; kept-vs-rejected score distributions; the
score cutoff; and the dual-rewrite kept/rejected breakdown. Writes the filtered 5B to
pretrain/rewrite/rewritten/ + _assembly_manifest.json + _step4_rewrite_summary.json. CPU-only.
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

from pp_io import atomic_write_table

BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B/rewrite')
SCORED_POOL = BASE / 'scored_pool'
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

SETTING = 'rewrite'
TARGET = 5_000_000_000
NAME_RE = re.compile(r'(wiki|distill)_(\d+)\.parquet$')

KEEP_OUT = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2',
            'source_prompt', 'rewritten_fasttext_score', 'rewritten_fasttext_ranking_v2',
            'fasttext-ranking-v2', 'url', 'metadata', 'topic']


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


def list_scored():
    out = []
    for p in sorted(glob.glob(str(SCORED_POOL / '*.parquet'))):
        m = NAME_RE.search(os.path.basename(p))
        if m:
            out.append((m.group(1), int(m.group(2)), p))
    return out


def _read_numeric(job):
    tag, k, path = job
    t = pq.read_table(path, columns=['doc_id', 'rewritten_tokens',
                                     'rewritten_fasttext_score'], use_threads=False)
    return dict(tag=tag, k=k, path=path, n=t.num_rows,
                doc_id=t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64),
                rtok=t.column('rewritten_tokens').to_numpy(zero_copy_only=False).astype(np.int64),
                score=t.column('rewritten_fasttext_score').to_numpy(zero_copy_only=False)
                .astype(np.float32))


def collect(scored, workers):
    res = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_read_numeric, j): idx for idx, j in enumerate(scored)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                res[idx] = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'read scored shard {scored[idx][2]} failed: {e!r}')
            done += 1
            if done % 50 == 0:
                log(f'  read scored: {done}/{len(scored)} ({time.time()-t0:.0f}s)')
    ordered = [res[i] for i in range(len(scored))]
    rows = np.array([s['n'] for s in ordered], dtype=np.int64)
    offs = np.zeros(len(ordered) + 1, dtype=np.int64); offs[1:] = np.cumsum(rows)
    cat = lambda key: np.concatenate([s[key] for s in ordered]) if ordered else np.empty(0)  # noqa
    meta = [(s['tag'], s['k'], s['path']) for s in ordered]
    return dict(meta=meta, offs=offs,
                doc_id=cat('doc_id').astype(np.int64),
                rtok=cat('rtok').astype(np.int64),
                score=cat('score').astype(np.float32),
                is_distill=np.concatenate([np.full(s['n'], s['tag'] == 'distill', bool)
                                           for s in ordered]) if ordered
                else np.empty(0, bool))


# ----------------------------------------------------------------------------- write pass
_WRITE = {}


def _write_shard(job):
    idx, tag, k, path = job
    mask = _WRITE['keep'][_WRITE['offs'][idx]:_WRITE['offs'][idx + 1]]
    cnt = int(mask.sum())
    if cnt == 0:
        return 0
    t = pq.read_table(path, use_threads=False)
    sub = t.filter(pa.array(mask))
    out = sub.select(KEEP_OUT)
    names = out.column_names
    names[names.index('rewritten')] = 'text'
    out = out.rename_columns(names)
    outp = PRETRAIN / SETTING / 'rewritten' / f'{tag}_{k:05d}.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(out, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'{tag}_{k}: rowcount mismatch')
    return cnt


def write_kept(meta, offs, keep, workers):
    global _WRITE
    _WRITE = dict(keep=keep, offs=offs)
    jobs = [(idx, tag, k, path) for idx, (tag, k, path) in enumerate(meta)]
    total = 0
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_shard, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                total += fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'write {j[1]}_{j[2]:05d} failed: {e!r}')
            done += 1
            if done % 50 == 0:
                log(f'  write kept: {done}/{len(jobs)} ({time.time()-t0:.0f}s)')
    if total != int(keep.sum()):
        stop(f'written {total} != kept {int(keep.sum())}')
    return total


def dist(scores):
    if scores.size == 0:
        return {q: None for q in ('min', 'p10', 'median', 'p90', 'max')}
    p = np.percentile(scores, [0, 10, 50, 90, 100])
    return dict(min=float(p[0]), p10=float(p[1]), median=float(p[2]), p90=float(p[3]),
                max=float(p[4]), n=int(scores.size))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP4 (rewrite) START: target={TARGET:,} workers={workers}')

    scored = list_scored()
    if not scored:
        stop(f'no scored shards under {SCORED_POOL} (run Step 3 first)')
    pool = collect(scored, workers)
    score = pool['score']; rtok = pool['rtok']; doc_id = pool['doc_id']
    is_distill = pool['is_distill']
    length = rtok + 1
    n_pool = score.size
    pool_tokens = int(length.sum())
    log(f'  pool: {n_pool:,} docs / {pool_tokens:,} tok')

    # 4a/4b global sort by raw score DESC (stable), fill to 5B
    order = np.argsort(-score, kind='stable')
    c = np.cumsum(length[order])
    if c[-1] < TARGET:
        keep_order = order
        kept_tokens = int(c[-1]); filled = False
    else:
        cut = int(np.searchsorted(c, TARGET, side='left'))
        keep_order = order[:cut + 1]
        kept_tokens = int(c[cut]); filled = True
    keep = np.zeros(n_pool, dtype=bool); keep[keep_order] = True
    cutoff = float(score[keep].min())
    n_kept = int(keep.sum())
    overshoot = kept_tokens - TARGET
    log(f'  kept {n_kept:,} docs / {kept_tokens:,} tok (cutoff raw score={cutoff:.6f}; '
        f'filled={filled}; overshoot {overshoot:,})')

    # 4c splits
    kw = keep & ~is_distill; kd = keep & is_distill
    kept_wiki = dict(docs=int(kw.sum()), tokens=int(length[kw].sum()))
    kept_distill = dict(docs=int(kd.sum()), tokens=int(length[kd].sum()))
    kept_dist = dist(score[keep]); rej_dist = dist(score[~keep])

    # dual-rewrite kept/rejected breakdown (doc_ids present in BOTH passes within the pool)
    wiki_doc = doc_id[~is_distill]; wiki_keep = keep[~is_distill]
    distill_doc = doc_id[is_distill]; distill_keep = keep[is_distill]
    dual_ids = np.intersect1d(wiki_doc, distill_doc)
    dual = dict(both_kept=0, both_rejected=0, wiki_kept_distill_rejected=0,
                distill_kept_wiki_rejected=0)
    if dual_ids.size:
        ow = np.argsort(wiki_doc); wd, wk = wiki_doc[ow], wiki_keep[ow]
        od = np.argsort(distill_doc); dd, dk = distill_doc[od], distill_keep[od]
        wk_dual = wk[np.searchsorted(wd, dual_ids)]
        dk_dual = dk[np.searchsorted(dd, dual_ids)]
        dual['both_kept'] = int((wk_dual & dk_dual).sum())
        dual['both_rejected'] = int((~wk_dual & ~dk_dual).sum())
        dual['wiki_kept_distill_rejected'] = int((wk_dual & ~dk_dual).sum())
        dual['distill_kept_wiki_rejected'] = int((~wk_dual & dk_dual).sum())
    dual_in_pool = int(dual_ids.size)

    # 4d write
    log('=== writing filtered top-5B rewritten shards ===')
    write_kept(pool['meta'], pool['offs'], keep, workers)

    manifest = {
        'setting': SETTING,
        'pipeline': 'rewire-inspired: rewrite broadly → fasttext score rewritten → keep top 5B',
        'selection_note': ('follows ReWire rewrite-then-filter; the rewrite pool is a RANDOM '
                           'upstream subset selected in 07_rewrite (compute-limited approximation '
                           'of "rewrite everything"); kept set is the token-budget-matched top-5B '
                           '(NOT the paper top-10%); score identical to fasttext-ranking-v2 '
                           '(raw P(hq) + v2 percentile vs the original 100M distribution)'),
        'pool_docs': n_pool, 'pool_tokens': pool_tokens,
        'pool_from_wikipedia': dict(docs=int((~is_distill).sum()),
                                    tokens=int(length[~is_distill].sum())),
        'pool_from_distill': dict(docs=int(is_distill.sum()),
                                  tokens=int(length[is_distill].sum())),
        'fasttext_model': 'fasttext_oh_eli5.bin',
        'fasttext_score_cutoff': cutoff,
        'kept_docs': n_kept, 'kept_tokens': kept_tokens,
        'kept_from_wikipedia': kept_wiki, 'kept_from_distill': kept_distill,
        'dual_rewrite_doc_ids_in_pool': dual_in_pool,
        'dual_rewrite_both_kept': dual['both_kept'],
        'target': TARGET, 'overshoot': int(overshoot),
    }
    (PRETRAIN / SETTING).mkdir(parents=True, exist_ok=True)
    (PRETRAIN / SETTING / '_assembly_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary = dict(manifest)
    summary.update(filled=filled, kept_score_dist=kept_dist, rejected_score_dist=rej_dist,
                   dual_breakdown=dual,
                   funnel_final=dict(stage='after FastText top-5B', docs=n_kept,
                                     tokens=kept_tokens))
    (HERE / '_step4_rewrite_summary.json').write_text(json.dumps(summary, indent=2))
    log('STEP4 DONE; wrote _assembly_manifest.json + _step4_rewrite_summary.json')


if __name__ == '__main__':
    main()
