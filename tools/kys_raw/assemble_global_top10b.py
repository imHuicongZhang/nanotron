#!/usr/bin/env python
"""Materialize the three global Top-10B corpora as raw text, shuffled, ready for export and tokenization.

Input: select_global_top10b.py output (<selection>/<setting>/selected_orig_doc_ids.npy, selected_doc_ids_order.npy)
and the flat arrays of extract_scored_columns.py (for the per-document token check).

  1. STAGE (one pass over the raw pool for all settings). For every raw-pool shard, read `text` and take the
     rows of the union of the selections: position orig_doc_id -> shard = id // 500000, row = id % 500000,
     as assemble_raw_corpus.pool_texts does for the four existing raw settings. No rewritten text, no
     transformation of any kind. Every document is re-tokenized (llama-2 tokenizer.json,
     add_special_tokens=False) and its length must equal the scored pool's tokens-llama2 EXACTLY; any
     mismatch stops the run. Per setting and shard: <out>/<setting>/_by_pool_shard/pool_NNNNN.parquet with
     columns doc_id (scored-pool id), orig_doc_id, text, source ('selected'), train_tokens (= tokens-llama2,
     without the +1), rows sorted by orig_doc_id.
  2. SHUFFLE. pp_io.bucketed_shuffle(seed=42, rows_per_shard=500_000) -- tools/kys_raw/pp_io.py is a
     byte-identical copy of projects/rewrite/10_postprocess/pp_io.py (sha256 5134dcc1...), the shuffle every
     published arm and the four raw settings used -- into <out>/<setting>/shuffled/part_NNNNN.parquet.
  3. <out>/<setting>/_raw_manifest.json with docs, TRAIN tokens (sum of train_tokens + 1), the selection
     digests, the shuffle parameters, and the sha256 of every shuffled part.

These corpora contain NO anchor: every row is a selected document.

    python tools/kys_raw/assemble_global_top10b.py --selection <dir> --flat <dir> --pool <100M raw pool> \
        --tokenizer <dir with tokenizer.json> --out <dir> [--workers 8] [--settings a,b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pp_io import bucketed_shuffle  # noqa: E402  (vendored copy, see docstring)

SETTINGS = ['raw_top10b_fineweb_edu', 'raw_top10b_modernbert', 'raw_top10b_consensus']
ROWS_PER_POOL_SHARD = 500_000
POOL_FILE = 'dclm_refinedweb_sample_{:05d}.parquet'
PP_IO_SHA256 = '5134dcc16bde1757345212ae7afd7a737133c8d7934e3570e9138e1b4e5f4d52'
SCHEMA = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()), ('text', pa.large_string()),
                    ('source', pa.large_string()), ('train_tokens', pa.int32())])


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def sha256_file(p, buf=16 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


_TOK = None


def _stage(job):
    shard, orig_ids, doc_ids, expect_tok, member, pool, tokenizer, out, settings = job
    staged = {s: Path(out) / s / '_by_pool_shard' / f'pool_{shard:05d}.parquet' for k, s in enumerate(settings) if member[:, k].any()}
    if all(p.is_file() for p in staged.values()):  # resume: this shard was staged by an earlier run (atomic writes)
        res = {}
        for s, p in staged.items():
            t = pq.read_table(p, columns=['doc_id', 'train_tokens'])
            k = settings.index(s)
            if not np.array_equal(t['doc_id'].to_numpy(), doc_ids[member[:, k]]):
                return shard, {'mismatch': [('staged file does not hold the expected ids', str(p), 0)], 'n_mismatch': 1}
            res[s] = (t.num_rows, int(t['train_tokens'].to_numpy().astype(np.int64).sum()) + t.num_rows)
        return shard, res
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(Path(tokenizer) / 'tokenizer.json'))
    col = pq.read_table(Path(pool) / POOL_FILE.format(shard), columns=['text'], use_threads=False).column('text')
    texts = col.take(pa.array(orig_ids % ROWS_PER_POOL_SHARD)).to_pylist()
    del col
    ntok = np.empty(len(texts), np.int64)
    for i in range(0, len(texts), 2_000):
        enc = _TOK.encode_batch(texts[i:i + 2_000], add_special_tokens=False)
        ntok[i:i + len(enc)] = [len(e.ids) for e in enc]
    bad = np.flatnonzero(ntok != expect_tok)
    if bad.size:
        return shard, {'mismatch': [(int(doc_ids[j]), int(ntok[j]), int(expect_tok[j])) for j in bad[:20]], 'n_mismatch': int(bad.size)}
    res = {}
    for k, s in enumerate(settings):
        m = member[:, k]
        if not m.any():
            continue
        t = pa.table({'doc_id': pa.array(doc_ids[m], pa.int64()), 'orig_doc_id': pa.array(orig_ids[m], pa.int64()),
                      'text': pa.array([texts[j] for j in np.flatnonzero(m)], pa.large_string()),
                      'source': pa.array(['selected'] * int(m.sum()), pa.large_string()),
                      'train_tokens': pa.array(ntok[m], pa.int32())}, schema=SCHEMA)
        p = Path(out) / s / '_by_pool_shard' / f'pool_{shard:05d}.parquet'
        tmp = p.with_name(p.name + '.tmp')
        pq.write_table(t, tmp, compression='zstd')
        os.replace(tmp, p)
        res[s] = (t.num_rows, int(ntok[m].sum()) + t.num_rows)
    return shard, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--settings', default=','.join(SETTINGS))
    args = ap.parse_args()
    settings = args.settings.split(',')
    if set(settings) - set(SETTINGS):
        sys.exit(f'unknown settings {sorted(set(settings) - set(SETTINGS))}')
    if sha256_file(Path(__file__).with_name('pp_io.py')) != PP_IO_SHA256:
        sys.exit('tools/kys_raw/pp_io.py differs from the pp_io.py every published arm was shuffled with')
    # workers are forked before any tokenizer exists, so each can use the Rust thread pool for encode_batch
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ.setdefault('RAYON_NUM_THREADS', str(max(1, len(os.sched_getaffinity(0)) // max(1, args.workers))))

    orig_all = np.load(args.flat / 'orig_doc_id.npy')
    tok_all = np.load(args.flat / 'tokens_llama2.npy')
    sel_man = json.loads((args.selection / 'selection_manifest.json').read_text())
    sets = {}
    for s in settings:
        o = np.load(args.selection / s / 'selected_doc_ids_order.npy')
        so = np.load(args.selection / s / 'selected_orig_doc_ids.npy')
        if not np.array_equal(np.sort(orig_all[o]), so) or np.unique(o).size != o.size:
            sys.exit(f'{s}: selection files inconsistent')
        sets[s] = np.sort(o)
    union = np.unique(np.concatenate(list(sets.values())))
    member = np.stack([np.isin(union, sets[s], assume_unique=True) for s in settings], axis=1)
    uorig = orig_all[union]  # strictly increasing with doc_id, so sorted
    log(f'union of {len(settings)} selections: {union.size:,} docs')

    for s in settings:
        d = args.out / s
        if (d / '_raw_manifest.json').exists():
            sys.exit(f'{d} already assembled; remove it to rebuild')
        shutil.rmtree(d / 'shuffled', ignore_errors=True)       # a shuffle is always redone from the staged shards
        shutil.rmtree(d / '_shuffle_tmp', ignore_errors=True)
        (d / '_by_pool_shard').mkdir(parents=True, exist_ok=True)  # completed staged shards are reused (resume)
    shards = uorig // ROWS_PER_POOL_SHARD
    jobs = []
    for sh in np.unique(shards):
        m = shards == sh
        jobs.append((int(sh), uorig[m], union[m], tok_all[union[m]].astype(np.int64), member[m],
                     str(args.pool), str(args.tokenizer), str(args.out), settings))
    tot = {s: [0, 0] for s in settings}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (sh, res) in enumerate(ex.map(_stage, jobs, chunksize=1), 1):
            if 'mismatch' in res:
                sys.exit(f'pool shard {sh}: {res["n_mismatch"]} documents whose re-tokenized length differs from '
                         f'tokens-llama2, e.g. (doc_id, got, expected) {res["mismatch"]}')
            for s, (n, t) in res.items():
                tot[s][0] += n
                tot[s][1] += t
            if i % 20 == 0:
                log(f'  staged {i}/{len(jobs)} pool shards')
    for s in settings:
        want = sel_man['settings'][s]
        if tot[s] != [want['docs'], want['train_tokens']]:
            sys.exit(f'{s}: staged {tot[s]} != selection {want["docs"]}/{want["train_tokens"]}')
        log(f'{s}: staged {tot[s][0]:,} docs / {tot[s][1]:,} TRAIN tokens; every document length equals tokens-llama2')

    for s in settings:
        root = args.out / s
        specs = [(str(p), None) for p in sorted((root / '_by_pool_shard').glob('pool_*.parquet'))]
        rows, n_sh, B = bucketed_shuffle(specs, lambda spec: pq.read_table(spec[0]), root / 'shuffled', root / '_shuffle_tmp',
                                         seed=42, rows_per_shard=500_000, mem_bytes=None, log=log)
        parts = sorted((root / 'shuffled').glob('part_*.parquet'))
        man = {'setting': s, 'contains_anchor': False, 'docs': tot[s][0], 'train_tokens': tot[s][1],
               'token_check': 'every document re-tokenized; length == scored-pool tokens-llama2 for all documents',
               'text_source': str(args.pool), 'text_read_in': 'assemble_global_top10b._stage (raw pool by orig_doc_id)',
               'selection': {k: sel_man['settings'][s][k] for k in ('score', 'docs', 'train_tokens', 'overshoot',
                                                                     'selection_order_sha256', 'docset_sha256', 'orig_docset_sha256')},
               'shuffle': {'function': 'tools/kys_raw/pp_io.py bucketed_shuffle', 'pp_io_sha256': PP_IO_SHA256, 'seed': 42,
                           'rows_per_shard': 500_000, 'buckets': int(B), 'input_order': 'staged pool_NNNNN.parquet sorted by path'},
               'shuffled_rows': int(rows), 'shuffled_shards': int(n_sh),
               'shuffled_sha256': {p.name: sha256_file(p) for p in parts}}
        (root / '_raw_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
        log(f'{s}: shuffled {rows:,} rows into {n_sh} parts (B={B}); wrote {root / "_raw_manifest.json"}')


if __name__ == '__main__':
    main()
