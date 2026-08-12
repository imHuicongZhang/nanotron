#!/usr/bin/env python
"""Assemble the 5B shared-only pretrain baseline (the "0B-specific" / 7th arm).

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ HISTORICAL RECORD — DO NOT RUN, DO NOT "FIX" THE PATHS.                              │
    │                                                                                      │
    │ Kept as the only in-repo evidence of how pp_io.bucketed_shuffle was invoked for the  │
    │ six arms: it is what substantiates the claim that all six corpora were shuffled      │
    │ identically at document level with seed 42, which the paper may need to defend.      │
    │                                                                                      │
    │ The absolute /scratch/bvandur1/zhuicon1/... paths below are PART OF THAT RECORD.     │
    │ They name the JHU tree the corpora were actually built on. Parameterising them, or   │
    │ repointing them at a download, would destroy the evidence and gain nothing — this    │
    │ script plays no part in training, and nothing in the repo calls it.                  │
    │                                                                                      │
    │ To obtain the corpora, see HANDOVER.md §9. They are published; do not rebuild them.  │
    └─────────────────────────────────────────────────────────────────────────────────────┘

Train-data baseline = ONLY the shared-top-5B anchor (no strategy-specific 5B), so that
    marginal(setting) = setting_E1_Mean7 - shared_only_Final_Mean7
isolates the contribution of each setting's strategy-specific data.

This mirrors the arms' Step-3 assembly (rewrite/10_postprocess/03_mix_shared_top.py) EXACTLY,
minus the rewritten half:

  A1. Physically COPY all shared-top-5B shards (+ _manifest.json) into
      pretrain/5B-shared-only/  (copy, NOT symlink; columns untouched -> "no modification").
  A2. VERIFY token budget by streaming the copied shards: total_docs, sum(tokens-llama2),
      and the task's convention sum(tokens-llama2 + 1) (= sum + n_docs, the per-doc +1 BOS),
      with deviation from the 5,000,000,000 target. The AUTHORITATIVE corpus count for
      train_steps is the tokenized .ds.metadata total (computed later by tokenize_arm.sh),
      NOT this parquet estimate.
  A3. Document-level RANDOM SHUFFLE (seed=42) of the 5B into pretrain/5B-shared-only/shuffled/
      via pp_io.bucketed_shuffle (the SAME memory-bounded two-pass shuffle every arm used).
      Required because the TokenizedBytes loader does NOT shuffle (shuffle_files=False,
      non-shuffling sampler) and the source parquet is fasttext-DESC ranked -> training in
      native order would be a quality curriculum the arms don't have.
  A4. Write _pretrain_manifest.json (same shape as the arms', shuffle_seed=42).

CPU-only. Reuses rewrite/10_postprocess/pp_io.py so the shuffle method is byte-identical to
the arms. After this, tokenize with:
    sbatch --job-name=tok_5B-shared-only tokenize_arm.sh 5B-shared-only
(its default input is pretrain/5B-shared-only/shuffled, output
 nanotron_tokenized/5B-shared-only/tokenized).

Usage (run on the cpu partition, nanotron-train env):
    python tools/assemble_5B_shared_only.py [--workers N] [--skip-copy] [--skip-shuffle]
    # --skip-shuffle runs copy+verify only (STEP A as the user described it).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the arms' exact I/O + shuffle helpers (byte-identical shuffle method, seed 42).
PP_DIR = '/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess'
sys.path.insert(0, PP_DIR)
from pp_io import atomic_copy, bucketed_shuffle, stream_shuffle_stats  # noqa: E402

# ----------------------------------------------------------------------------- paths / constants
SHARED_SRC = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B/shared-top-5B')
OUT = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain/5B-shared-only')
SHUFFLED = OUT / 'shuffled'
TMP = OUT / '_shuffle_tmp'

SETTING = '5B-shared-only'
SHUFFLE_SEED = 42
ROWS_PER_SHARD = 500_000
TARGET_5B = 5_000_000_000

# Same unified shuffle schema as the arms (03_mix_shared_top.py). `text` is the heavy column;
# source_prompt='original' lets stream_shuffle_stats report the +1 budget under "shared_top".
UNIFIED = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()),
                     ('text', pa.large_string()), ('source_prompt', pa.large_string()),
                     ('train_tokens', pa.int32())])


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- A1 copy (no modification)
def _copy_one(src):
    """Byte-for-byte atomic copy of one shard; assert rowcount preserved."""
    name = Path(src).name
    dest = OUT / name
    n_src = pq.ParquetFile(src).metadata.num_rows
    atomic_copy(src, dest)
    n_dst = pq.ParquetFile(dest).metadata.num_rows
    if n_dst != n_src:
        raise RuntimeError(f'copy {name}: rowcount {n_dst} != {n_src}')
    return n_dst


def copy_shared(workers):
    srcs = sorted(glob.glob(str(SHARED_SRC / 'part_*.parquet')))
    if not srcs:
        stop(f'no shared-top shards under {SHARED_SRC}')
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_copy_one, s): s for s in srcs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                total += fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'copy {Path(s).name} failed: {e!r}')
    # carry the source manifest alongside the copy (informational; not modified)
    src_manifest = SHARED_SRC / '_manifest.json'
    if src_manifest.exists():
        atomic_copy(src_manifest, OUT / '_source_manifest.json')
    log(f'  A1 copied {len(srcs)} shards -> {total:,} rows into {OUT}')
    return len(srcs), total


# ----------------------------------------------------------------------------- A2 verify token budget
def verify_tokens():
    """Stream the COPIED shards' light columns and report the token budget.
    Reports both raw sum(tokens-llama2) and the task convention sum(tokens-llama2 + 1)."""
    files = sorted(glob.glob(str(OUT / 'part_*.parquet')))
    if not files:
        stop(f'no copied shards under {OUT} (run copy first)')
    n_docs = 0
    sum_tok = 0
    for p in files:
        col = pq.read_table(p, columns=['tokens-llama2'], use_threads=False).column('tokens-llama2')
        arr = col.to_numpy(zero_copy_only=False).astype(np.int64)
        n_docs += arr.size
        sum_tok += int(arr.sum())
    sum_tok_plus1 = sum_tok + n_docs  # sum(tokens-llama2 + 1): one +1 (BOS) per document
    dev_raw = sum_tok - TARGET_5B
    dev_plus1 = sum_tok_plus1 - TARGET_5B
    log('  A2 token verification:')
    log(f'      shards            = {len(files)}')
    log(f'      total_docs        = {n_docs:,}')
    log(f'      sum(tokens-llama2)            = {sum_tok:,}   (deviation {dev_raw:+,})')
    log(f'      sum(tokens-llama2 + 1)        = {sum_tok_plus1:,}   (deviation {dev_plus1:+,})')
    log(f'      target                       = {TARGET_5B:,}')
    log(f'      ~5B check         = {"OK" if abs(dev_plus1) < 0.01 * TARGET_5B else "OFF >1%"}')
    return dict(shards=len(files), total_docs=n_docs, sum_tokens_llama2=sum_tok,
                sum_tokens_plus1=sum_tok_plus1, deviation_plus1=dev_plus1)


# ----------------------------------------------------------------------------- A3 shuffle (seed 42)
def load_unified(path):
    """Read one copied shard into the UNIFIED schema (text = original text, source_prompt='original',
    train_tokens = tokens-llama2)."""
    t = pq.read_table(path, columns=['doc_id', 'orig_doc_id', 'text', 'tokens-llama2'],
                      use_threads=False)
    n = t.num_rows
    return pa.table({
        'doc_id': t.column('doc_id').cast(pa.int64()),
        'orig_doc_id': t.column('orig_doc_id').cast(pa.int64()),
        'text': t.column('text').cast(pa.large_string()),
        'source_prompt': pa.array(['original'] * n, type=pa.large_string()),
        'train_tokens': t.column('tokens-llama2').cast(pa.int32()),
    }, schema=UNIFIED)


def shuffle_and_write():
    # specs are (path, aux) 2-tuples — pp_io.choose_buckets unpacks `for p, _ in specs`.
    specs = [(p, None) for p in sorted(glob.glob(str(OUT / 'part_*.parquet')))]
    if not specs:
        stop(f'no copied shards under {OUT} to shuffle')
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_bytes = mem_mb * 1024 * 1024 if mem_mb else None
    total_rows, n_shards, B = bucketed_shuffle(
        specs, lambda s: load_unified(s[0]),
        SHUFFLED, TMP,
        seed=SHUFFLE_SEED, rows_per_shard=ROWS_PER_SHARD, mem_bytes=mem_bytes, log=log)
    log(f'  A3 shuffled {total_rows:,} rows into {n_shards} shards (B={B}, seed={SHUFFLE_SEED})')
    return total_rows, n_shards


# ----------------------------------------------------------------------------- A4 manifest
def write_manifest(total_rows, n_shards):
    st = stream_shuffle_stats(SHUFFLED)  # 'original' bucket -> shared_top_* (+1 budget)
    manifest = {
        'setting': SETTING,
        'shared_top_docs': st['shared_top_docs'],
        'shared_top_tokens': st['shared_top_tokens'],   # = sum(tokens-llama2 + 1)
        'rewritten_docs': 0, 'rewritten_tokens': 0,     # none, by construction
        'total_docs_in_shuffled': int(total_rows),
        'total_tokens': st['total_tokens'],
        'target': TARGET_5B, 'overshoot': st['total_tokens'] - TARGET_5B,
        'shuffle_seed': SHUFFLE_SEED,
        'shuffled_shards': n_shards,
        'note': 'shared-top-5B anchor only; no strategy-specific data; 1-epoch baseline.',
    }
    (OUT / '_pretrain_manifest.json').write_text(json.dumps(manifest, indent=2))
    log(f'  A4 wrote {OUT / "_pretrain_manifest.json"}')
    log(f'      total_tokens (sum+1) = {st["total_tokens"]:,}  overshoot vs 5B = {manifest["overshoot"]:+,}')
    # train_steps preview (final value comes from tokenized .ds.metadata, not this estimate)
    tok_per_step = 4 * 64 * 4 * 2048  # mbs * accum * dp * seq = 2,097,152
    est_steps = round(st['total_tokens'] / tok_per_step)
    log(f'      train_steps preview (1 epoch) = round({st["total_tokens"]:,} / {tok_per_step:,}) '
        f'= {est_steps}  [finalize from .ds.metadata after tokenizing]')
    return manifest


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    ap.add_argument('--skip-copy', action='store_true', help='reuse an existing copy under OUT')
    ap.add_argument('--skip-shuffle', action='store_true',
                    help='copy + verify only (STEP A as originally scoped; no /shuffled, no manifest)')
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'ASSEMBLE 5B-shared-only START: workers={workers} seed={SHUFFLE_SEED}')
    log(f'  src={SHARED_SRC}')
    log(f'  out={OUT}')

    if not args.skip_copy:
        copy_shared(workers)
    else:
        log('  A1 skipped (--skip-copy)')

    verify_tokens()

    if args.skip_shuffle:
        log('DONE (copy + verify only; --skip-shuffle).')
        return

    total_rows, n_shards = shuffle_and_write()
    write_manifest(total_rows, n_shards)
    log('ASSEMBLE 5B-shared-only DONE.')


if __name__ == '__main__':
    main()
