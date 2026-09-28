#!/usr/bin/env python
"""Pull the selection columns of the 99,949,162-row scored pool into flat arrays.

The scored pool (the table every KYS selection ran over, `6_merged_clean`) is published unchanged as
the HF dataset blab-jhu/KYS-DCLM-Refinedweb-100M-Scored. This reads only the light columns, at a
pinned revision, and writes one array per column indexed by `doc_id` (0 .. 99,949,161):

    doc_id -> orig_doc_id.npy (int64)       position in the raw 100M reservoir pool
              tokens_llama2.npy (int32)     len(tokenize(text, add_special_tokens=False))
              ft_v2.npy fw_v2.npy mb_v2.npy (float32)  fasttext / fineweb-edu / modernbert -ranking-v2
              topic_code.npy (int8)         index into topic_vocab.json (sorted, 24 labels)

Checks, identical to 04_select/select_10b.py pass1: 200 shards, 500,000 rows each except the last
(449,162); doc_id contiguous; zero nulls in every column read; exactly 24 topic labels.

    python tools/kys_raw/extract_scored_columns.py --out <dir> [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = 'blab-jhu/KYS-DCLM-Refinedweb-100M-Scored'
REVISION = 'dcbbc360838f3f7108ad8f2b810cc4814f9d0431'
NSHARDS, ROWS_FULL, LAST_ROWS, N = 200, 500_000, 449_162, 99_949_162
V2 = {'ft_v2': 'fasttext-ranking-v2', 'fw_v2': 'fineweb-edu-ranking-v2', 'mb_v2': 'modernbert-ranking-v2'}
COLS = ['doc_id', 'orig_doc_id', 'tokens-llama2', 'topic', *V2.values()]


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def read_shard(i, local_dir):
    if local_dir:
        return i, pq.read_table(Path(local_dir) / f'merged_clean_{i:05d}.parquet', columns=COLS)
    from huggingface_hub import HfFileSystem
    path = f'datasets/{REPO}@{REVISION}/merged_clean_{i:05d}.parquet'
    for attempt in range(5):
        try:
            with HfFileSystem().open(path, 'rb', block_size=16 << 20) as fh:
                return i, pq.ParquetFile(fh).read(columns=COLS)
        except Exception as e:  # transient HTTP errors
            log(f'shard {i}: {type(e).__name__} {e}; retry {attempt + 1}')
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f'shard {i}: failed after 5 attempts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--local-dir', default=None, help='read local copies of merged_clean_*.parquet instead of the Hub')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arr = {'orig_doc_id': np.empty(N, np.int64), 'tokens_llama2': np.empty(N, np.int32),
           **{k: np.empty(N, np.float32) for k in V2}}
    topic_local, nulls = {}, {c: 0 for c in COLS}
    with ThreadPoolExecutor(args.workers) as ex:
        for done, (i, t) in enumerate(ex.map(lambda i: read_shard(i, args.local_dir), range(NSHARDS)), 1):
            n = ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS
            if t.num_rows != n:
                raise SystemExit(f'shard {i}: {t.num_rows} rows != {n}')
            lo = i * ROWS_FULL
            if not np.array_equal(t['doc_id'].to_numpy(), np.arange(lo, lo + n)):
                raise SystemExit(f'shard {i}: doc_id not contiguous')
            for c in COLS:
                nulls[c] += t[c].null_count
            arr['orig_doc_id'][lo:lo + n] = t['orig_doc_id'].to_numpy()
            arr['tokens_llama2'][lo:lo + n] = t['tokens-llama2'].to_numpy()
            for k, c in V2.items():
                arr[k][lo:lo + n] = t[c].to_numpy()
            d = t['topic'].combine_chunks().dictionary_encode()
            topic_local[i] = (d.dictionary.to_pylist(), d.indices.to_numpy().astype(np.int32))
            if done % 20 == 0:
                log(f'{done}/{NSHARDS} shards')
    if any(nulls.values()):
        raise SystemExit(f'nulls: {nulls}')
    vocab = sorted({s for v, _ in topic_local.values() for s in v})
    if len(vocab) != 24:
        raise SystemExit(f'{len(vocab)} topic labels, expected 24: {vocab}')
    code = {s: k for k, s in enumerate(vocab)}
    topic = np.empty(N, np.int8)
    for i, (v, c) in topic_local.items():
        topic[i * ROWS_FULL:i * ROWS_FULL + c.size] = np.array([code[s] for s in v], np.int8)[c]
    arr['topic_code'] = topic

    meta = {'repo': REPO, 'revision': REVISION, 'rows': N, 'columns': COLS, 'nulls': nulls,
            'topic_vocab': vocab, 'sha256': {}}
    for k, a in arr.items():
        np.save(args.out / f'{k}.npy', a)
        meta['sha256'][f'{k}.npy'] = hashlib.sha256(a.tobytes()).hexdigest()
    (args.out / 'topic_vocab.json').write_text(json.dumps(vocab, indent=1))
    (args.out / 'extract_manifest.json').write_text(json.dumps(meta, indent=1))
    log(f'wrote {len(arr)} arrays to {args.out}; orig_doc_id range {arr["orig_doc_id"].min()}..{arr["orig_doc_id"].max()}')


if __name__ == '__main__':
    main()
