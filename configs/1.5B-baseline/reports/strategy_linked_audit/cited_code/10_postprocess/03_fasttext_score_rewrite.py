#!/usr/bin/env python
"""STEP 3 (rewrite / ReWire-inspired) — FastText-score the REWRITTEN text (the core step).

Scores the `rewritten` column (NOT the original `text`) for EVERY status==2 doc in the pool,
using the IDENTICAL preprocessing + score definition that produced the original `fasttext`
column and its `fasttext-ranking-v2` percentile. Verified against:

  * /scratch/bvandur1/zhuicon1/projects/rewrite/01_explore/score_fasttext.py
      clean_for_fasttext: text.replace("\\n"," ").replace("\\r"," ")[:100_000]   (no lower-
      casing, no whitespace collapse, no HTML/URL stripping); model.predict(cleaned, k=1);
      raw = p if label=="__label__hq" else 1-p; empty text -> 0.0.
  * /scratch/bvandur1/zhuicon1/projects/rewrite/00_TMP/clean_v2_ranks.py
      fasttext-ranking-v2 = scipy.stats.rankdata(raw, "average") / 99,949,162, ranked GLOBALLY
      over the 6_merged_clean corpus (N=99,949,162).

Two score columns are written (the top-5B ordering is identical under both — monotonic):
  rewritten_fasttext_score        : raw P(__label__hq)  (Step 3b)
  rewritten_fasttext_ranking_v2   : that raw score's tie-aware percentile against the ORIGINAL
                                    100M-doc raw `fasttext` distribution (same recipe/scale as
                                    fasttext-ranking-v2), via searchsorted (Step 3a-pre).

Output: .../10B/rewrite/scored_pool/{wiki,distill}_NNNNN.parquet (status==2 rows + scores).
Reports the score distribution overall and split by source_prompt. CPU-only.
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
CLEAN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/6_merged_clean')
FT_MODEL = '/scratch/bvandur1/zhuicon1/models/5m/external/fasttext_oh_eli5.bin'
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

N_ORIG_EXPECT = 99_949_162                 # ranking population used by clean_v2_ranks.py
FASTTEXT_MAX_CHARS = 100_000
HQ_LABEL = '__label__hq'
SHARD_RE = re.compile(r'part_(\d+)\.parquet$')

# columns carried into scored_pool (original `text` dropped; `rewritten` kept for Step 4)
KEEP = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2', 'status',
        'fasttext-ranking-v2', 'url', 'metadata', 'topic']

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_FT = None
_ORIG_SORTED = None          # sorted ascending original raw fasttext scores (set before fork)
_ORIG_N = 0


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)


def list_shards(subdir):
    out = []
    for p in sorted(glob.glob(str(BASE / subdir / 'part_*.parquet'))):
        out.append((int(SHARD_RE.search(os.path.basename(p)).group(1)), p))
    return out


def clean_for_fasttext(text):
    """EXACT reproduction of score_fasttext.clean_for_fasttext."""
    if text is None:
        return ''
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    return text[:FASTTEXT_MAX_CHARS]


# ----------------------------------------------------------------------------- original reference dist
def _read_orig_fasttext(i):
    p = CLEAN / f'merged_clean_{i:05d}.parquet'
    return pq.read_table(p, columns=['fasttext'], use_threads=False)\
        .column('fasttext').to_numpy(zero_copy_only=False).astype(np.float64)


def build_orig_sorted(workers):
    """Load the original raw `fasttext` scores (6_merged_clean) and sort ascending — the
    reference population for the v2 percentile (same one clean_v2_ranks.py ranked over)."""
    shards = sorted(glob.glob(str(CLEAN / 'merged_clean_*.parquet')))
    if not shards:
        stop(f'cannot build reference distribution: no shards under {CLEAN}')
    log(f'building original raw-fasttext reference distribution from {len(shards)} shards...')
    parts = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as ex:
        futs = {ex.submit(_read_orig_fasttext, i): i for i in range(len(shards))}
        for fut in as_completed(futs):
            i = futs[fut]
            parts[i] = fut.result()
    arr = np.concatenate(parts)
    n = arr.size
    if n != N_ORIG_EXPECT:
        log(f'  ⚠️ reference N={n:,} != expected {N_ORIG_EXPECT:,}; using actual N for percentile.')
    arr.sort(kind='quicksort')
    log(f'reference distribution ready: N={n:,}, min={arr[0]:.4f}, max={arr[-1]:.4f}')
    return arr, n


def raw_to_v2(raw):
    """Tie-aware percentile of `raw` against _ORIG_SORTED — reproduces
    rankdata(.,'average')/N for query values: avg_rank = (#<raw + #<=raw + 1)/2."""
    left = np.searchsorted(_ORIG_SORTED, raw, side='left').astype(np.float64)
    right = np.searchsorted(_ORIG_SORTED, raw, side='right').astype(np.float64)
    return ((left + right + 1.0) * 0.5 / _ORIG_N).astype(np.float32)


# ----------------------------------------------------------------------------- scoring worker
def _init_score_worker():
    pa.set_cpu_count(1)
    global _FT
    import fasttext
    _FT = fasttext.load_model(FT_MODEL)
    if HQ_LABEL not in _FT.get_labels():
        raise RuntimeError(f'{HQ_LABEL} not in model labels {_FT.get_labels()}')


def _score_shard(job):
    subdir, k, path, source_prompt = job
    t = pq.read_table(path, use_threads=False)
    status = t.column('status').to_numpy(zero_copy_only=False)
    sel = status == 2
    cnt = int(sel.sum())
    if cnt == 0:
        return dict(subdir=subdir, cnt=0, scores=np.empty(0, np.float32))
    sub = t.filter(pa.array(sel))
    texts = sub.column('rewritten').to_pylist()

    # ONE string at a time — byte-for-byte identical to score_fasttext.py:95-105 (NOT batched)
    raw = np.zeros(cnt, dtype=np.float32)
    for i, text in enumerate(texts):
        cleaned = clean_for_fasttext(text)
        if not cleaned:
            continue                                     # empty -> 0.0 (no predict call)
        labels, probs = _FT.predict(cleaned, k=1)
        p = float(probs[0])
        raw[i] = p if labels[0] == HQ_LABEL else 1.0 - p
    v2 = raw_to_v2(raw)

    out = sub.select(KEEP)
    out = out.append_column('source_prompt',
                            pa.array([source_prompt] * cnt, type=pa.large_string()))
    out = out.append_column('rewritten_fasttext_score', pa.array(raw, type=pa.float32()))
    out = out.append_column('rewritten_fasttext_ranking_v2', pa.array(v2, type=pa.float32()))

    outp = SCORED_POOL / f'{subdir}_{k:05d}.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(out, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'{subdir}_{k}: rowcount mismatch')
    return dict(subdir=subdir, cnt=cnt, scores=raw)


def score_all(jobs, workers):
    by_source = {'wiki': [], 'distill': []}
    total = 0
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_score_worker) as ex:
        futs = {ex.submit(_score_shard, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'score {j[0]}_{j[1]:05d} failed: {e!r}')
            if r['cnt']:
                by_source[r['subdir']].append(r['scores'])
                total += r['cnt']
            done += 1
            if done % 25 == 0:
                log(f'  scored {done}/{len(jobs)} shards, {total:,} docs ({time.time()-t0:.0f}s)')
    return by_source, total


def dist(scores):
    if scores.size == 0:
        return {q: None for q in ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')}
    p = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
    return dict(min=float(p[0]), p10=float(p[1]), p25=float(p[2]), median=float(p[3]),
                p75=float(p[4]), p90=float(p[5]), max=float(p[6]), mean=float(scores.mean()),
                n=int(scores.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)),
                    help='workers for reading the reference distribution')
    ap.add_argument('--score-workers', type=int, default=0,
                    help='workers for FastText scoring (each loads a ~2.4GB model copy). '
                         '0 = auto from SLURM mem (cap at cpus).')
    args = ap.parse_args()
    workers = max(1, args.workers)
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    score_workers = args.score_workers or max(1, min(workers, int((mem_gb - 32) // 3.0)))
    log(f'STEP3 (rewrite) START: read workers={workers}, score workers={score_workers} '
        f'(mem={mem_gb:.0f}G), model={FT_MODEL}')

    # reference distribution for v2 (set as globals BEFORE forking scoring workers -> COW shared)
    global _ORIG_SORTED, _ORIG_N
    _ORIG_SORTED, _ORIG_N = build_orig_sorted(workers)

    jobs = ([('wiki', k, p, 'wikipedia') for k, p in list_shards('rewritten')]
            + [('distill', k, p, 'distill') for k, p in list_shards('distill')])
    log(f'scoring {len(jobs)} shards (rewritten text only)...')
    by_source, total = score_all(jobs, score_workers)

    wiki_scores = np.concatenate(by_source['wiki']) if by_source['wiki'] else np.empty(0, np.float32)
    distill_scores = (np.concatenate(by_source['distill']) if by_source['distill']
                      else np.empty(0, np.float32))
    all_scores = np.concatenate([wiki_scores, distill_scores]) if total else np.empty(0, np.float32)

    overall = dist(all_scores); wiki_d = dist(wiki_scores); distill_d = dist(distill_scores)
    log(f'3c overall raw score: min={overall["min"]:.4f} p10={overall["p10"]:.4f} '
        f'median={overall["median"]:.4f} p90={overall["p90"]:.4f} max={overall["max"]:.4f} '
        f'(n={overall["n"]:,})')
    log(f'   wikipedia: median={wiki_d["median"] if wiki_d["n"] else "n/a"} ; '
        f'distill: median={distill_d["median"] if distill_d["n"] else "n/a"}')

    summary = dict(setting='rewrite', fasttext_model=os.path.basename(FT_MODEL),
                   reference_N=_ORIG_N, scored_docs=int(total),
                   score_definition='raw = p if __label__hq else 1-p; '
                                    'v2 = rankdata-percentile vs original 100M raw fasttext',
                   distribution_overall=overall,
                   distribution_wikipedia=wiki_d, distribution_distill=distill_d)
    (HERE / '_step3_rewrite_summary.json').write_text(json.dumps(summary, indent=2))
    log(f'STEP3 DONE; scored {total:,} docs -> {SCORED_POOL}; wrote _step3_rewrite_summary.json')


if __name__ == '__main__':
    main()
