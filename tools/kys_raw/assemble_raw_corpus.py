#!/usr/bin/env python
"""Assemble one raw-selected baseline corpus as shuffled parquet, ready for datatrove tokenization.

TEXT PROVENANCE — the property this script exists to guarantee:
    Every `text` value written here is read from the 100M raw DCLM-RefinedWeb pool
    (datasets/ppl-dsai/dclm-refinedweb-100m-sample) at position orig_doc_id, in
    `pool_texts()` below and nowhere else. The published wytro/Know-Your-Sources parquet is
    only ever read for its `orig_doc_id` / `source_prompt` columns (by build_raw_sources.py),
    never for `text`, and no rewrite output directory is opened at all.

Inputs (from tools/kys_raw/build_raw_sources.py):
    <sources>/<raw_setting>/selected_doc_ids.npy   strategy half, after the 5B budget rule
    <sources>/anchor_doc_ids.npy                   shared anchor ids (only with --with-anchor)

Output:
    <out>/<raw_setting>/shuffled/part_NNNNN.parquet  columns: doc_id, orig_doc_id, text,
        source ('anchor' | 'raw_selected'), train_tokens (= tokens-llama2, without the +1),
    shuffled at document level with seed 42 by the same pp_io.bucketed_shuffle every grid arm
    used, plus _raw_manifest.json.

Usage:
    python tools/kys_raw/assemble_raw_corpus.py --setting raw_diversity_oriented \
        --sources <build_raw_sources out> --pool <100m pool> --tokenizer <dir> --out <dir> \
        [--with-anchor] [--strategy-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PP_DIR = '/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/10_postprocess'
ROWS_PER_POOL_SHARD = 500_000
POOL_FILE = 'dclm_refinedweb_sample_{:05d}.parquet'
SCHEMA = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()), ('text', pa.large_string()),
                    ('source', pa.large_string()), ('train_tokens', pa.int32())])


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def pool_texts(pool: Path, shard: int, rows: np.ndarray) -> list[str]:
    """THE ONLY PLACE TEXT IS READ: the raw 100M pool, by position."""
    col = pq.read_table(pool / POOL_FILE.format(shard), columns=['text']).column('text')
    return col.take(pa.array(rows)).to_pylist()


_TOK = None


def _write_shard(job):
    shard, ids, sources, pool, tokenizer, stage_dir = job
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(Path(tokenizer) / 'tokenizer.json'))
    texts = pool_texts(Path(pool), shard, ids % ROWS_PER_POOL_SHARD)
    ntok = []
    for i in range(0, len(texts), 2_000):   # chunked: a full-shard encode_batch can exceed worker memory
        ntok += [len(e.ids) for e in _TOK.encode_batch(texts[i:i + 2_000], add_special_tokens=False)]
    t = pa.table({'doc_id': pa.array(ids, pa.int64()), 'orig_doc_id': pa.array(ids, pa.int64()),
                  'text': pa.array(texts, pa.large_string()), 'source': pa.array(sources, pa.large_string()),
                  'train_tokens': pa.array(ntok, pa.int32())}, schema=SCHEMA)
    out = Path(stage_dir) / f'pool_{shard:05d}.parquet'
    pq.write_table(t, out, compression='zstd')
    return shard, t.num_rows, int(np.sum(ntok)) + t.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=['raw_diversity_oriented', 'raw_disagreement_aware', 'raw_random', 'raw_rewire_inspired'])
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--with-anchor', action='store_true', help='include the shared anchor (text still read from the pool)')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK') or 4))
    ap.add_argument('--no-shuffle', action='store_true', help='stop after the per-pool-shard stage (verification only)')
    args = ap.parse_args()

    sel = np.load(args.sources / args.setting / 'selected_doc_ids.npy')
    parts = [(sel, 'raw_selected')]
    if args.with_anchor:
        anc = np.load(args.sources / 'anchor_doc_ids.npy')
        if np.intersect1d(sel, anc).size:
            sys.exit('selected documents overlap the anchor')
        parts.append((anc, 'anchor'))
    ids = np.concatenate([p for p, _ in parts])
    src = np.concatenate([np.full(p.size, s, dtype=object) for p, s in parts])
    order = np.argsort(ids, kind='stable')
    ids, src = ids[order], src[order]
    if np.unique(ids).size != ids.size:
        sys.exit('duplicate orig_doc_id in corpus')

    root = args.out / args.setting
    stage = root / '_by_pool_shard'
    stage.mkdir(parents=True, exist_ok=True)
    shards = ids // ROWS_PER_POOL_SHARD
    jobs = []
    for s in np.unique(shards):
        m = shards == s
        jobs.append((int(s), ids[m], src[m].tolist(), str(args.pool), str(args.tokenizer), str(stage)))
    n_docs = n_tok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (s, n, t) in enumerate(ex.map(_write_shard, jobs, chunksize=1), 1):
            n_docs += n
            n_tok += t
            if i % 20 == 0:
                log(f'  {i}/{len(jobs)} pool shards, {n_docs:,} docs')
    log(f'{args.setting}: staged {n_docs:,} docs / {n_tok:,} tokens (+1 convention) from the raw pool')
    manifest = {'setting': args.setting, 'with_anchor': args.with_anchor, 'docs': n_docs,
                'tokens_plus1': n_tok, 'text_source': str(args.pool), 'text_read_in': 'assemble_raw_corpus.pool_texts',
                'shuffle_seed': 42 if not args.no_shuffle else None}

    if not args.no_shuffle:
        sys.path.insert(0, PP_DIR)
        from pp_io import bucketed_shuffle  # same two-pass document-level shuffle as every grid arm
        specs = [(str(p), None) for p in sorted(stage.glob('pool_*.parquet'))]

        def read_staged(spec):
            # re-reads this script's own staged shards (text already taken from the pool above)
            return pq.read_table(spec[0])  # staged

        total_rows, n_shards, B = bucketed_shuffle(
            specs, read_staged, root / 'shuffled', root / '_shuffle_tmp',
            seed=42, rows_per_shard=500_000, mem_bytes=None, log=log)
        manifest.update(shuffled_rows=int(total_rows), shuffled_shards=int(n_shards), buckets=int(B))
    (root / '_raw_manifest.json').write_text(json.dumps(manifest, indent=2))
    log(f'wrote {root / "_raw_manifest.json"}')


if __name__ == '__main__':
    main()
