#!/usr/bin/env python
"""Build the source-document sets for the raw-selected baselines (configs/1.5B-baseline-seed42).

For each rewritten arm (diversity_oriented, disagreement_aware, wrap_inspired) the raw
counterpart trains on the shared 5B anchor plus the ORIGINAL text of the documents that arm
rewrote ("option B"):

  1. SOURCE SET. Read `orig_doc_id` and `source_prompt` from the published
     wytro/Know-Your-Sources/<arm>/*.parquet. Rows with source_prompt == 'original' are the
     anchor; every other row is a rewrite (wikipedia/distill/wrap styles), and one source
     document may appear under several prompts, so the source set is the UNIQUE orig_doc_id of
     the non-anchor rows.
  2. TEXT. `orig_doc_id` is a positional index into the 100M DCLM-RefinedWeb pool
     (500,000 rows per shard): shard = id // 500000, row = id % 500000. Verified by exact text
     match of anchor rows against the pool.
  3. TOKENS. The grid convention: len(llama2 tokenizer(text, add_special_tokens=False)) + 1 per
     document. (The published `train_tokens` column is the count WITHOUT the +1 — verified on
     anchor rows — and the +1 is added at budget time, as in the recorded anchor total
     5,000,002,332 = sum(train_tokens + 1).)
  4. BUDGET. If the source set exceeds 5,000,000,000 tokens, take documents in the order of a
     uniform random permutation (np.random.default_rng(42) over the sorted unique ids) and keep
     the shortest prefix whose cumulative tokens reach 5B — whole documents, overshoot at most
     one document, the same fill-to convention the selections used. At or below 5B the set is
     used as is.

The anchor ids are extracted and checked too (count, token total, identical across the three
arms), so the anchor can be sourced from the published parquet if that route is approved.

Outputs under --out:
    <raw_setting>/source_doc_ids.npy      sorted unique source ids (before budget)
    <raw_setting>/source_tokens.npy       tokens+1 aligned to source_doc_ids
    <raw_setting>/selected_doc_ids.npy    ids kept after the budget rule (permutation order)
    anchor_doc_ids.npy / anchor_train_tokens_plus1.npy
    manifest.json                         every count reported in the README

Usage (CPU node):
    python tools/kys_raw/build_raw_sources.py --parquet-root <hf_parquet> --pool <100m pool dir> \
        --tokenizer <tokenizer dir> --out <dir> --workers 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ARMS = {  # raw setting -> published rewritten arm
    'raw_diversity_oriented': 'diversity_oriented',
    'raw_disagreement_aware': 'disagreement_aware',
    'raw_random': 'wrap_inspired',
    'raw_rewire_inspired': 'rewire_inspired',
}
ROWS_PER_POOL_SHARD = 500_000
BUDGET = 5_000_000_000
SEED = 42
ANCHOR_DOCS, ANCHOR_TOKENS = 4_120_164, 5_000_002_332
ENCODE_CHUNK = 2_000   # documents per encode_batch call; bounds worker memory


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def read_arm(parquet_root: Path, arm: str):
    files = sorted(glob.glob(str(parquet_root / arm / '*.parquet')))
    if not files:
        raise SystemExit(f'no parquet under {parquet_root / arm}')
    src, anc, anc_tok, n_rows, n_rewrites = [], [], [], 0, 0
    for f in files:
        t = pq.read_table(f, columns=['orig_doc_id', 'source_prompt', 'train_tokens'])
        oid = t.column('orig_doc_id').to_numpy().astype(np.int64)
        is_anchor = np.asarray(t.column('source_prompt').to_pylist(), dtype=object) == 'original'
        src.append(oid[~is_anchor])
        anc.append(oid[is_anchor])
        anc_tok.append(t.column('train_tokens').to_numpy().astype(np.int64)[is_anchor] + 1)
        n_rows += t.num_rows
        n_rewrites += int((~is_anchor).sum())
    anchor = np.concatenate(anc)
    order = np.argsort(anchor)
    return dict(files=len(files), rows=n_rows, rewrite_rows=n_rewrites,
                source_ids=np.unique(np.concatenate(src)),
                anchor_ids=anchor[order], anchor_tok=np.concatenate(anc_tok)[order])


_TOK = None


def _count_shard(job):
    shard, rows, pool, tokenizer = job
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(Path(tokenizer) / 'tokenizer.json'))
    col = pq.read_table(Path(pool) / f'dclm_refinedweb_sample_{shard:05d}.parquet', columns=['text']).column('text')
    texts = col.take(rows).to_pylist()
    del col
    # Encode in chunks and keep only lengths: a full-shard encode_batch materializes an Encoding
    # (ids, tokens, offsets, masks) per document, which OOM-killed a worker at ~16 GB RSS.
    out = np.empty(len(texts), dtype=np.int64)
    for i in range(0, len(texts), ENCODE_CHUNK):
        enc = _TOK.encode_batch(texts[i:i + ENCODE_CHUNK], add_special_tokens=False)
        out[i:i + len(enc)] = [len(e.ids) + 1 for e in enc]
    return shard, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-root', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK') or 8))
    ap.add_argument('--settings', default=','.join(ARMS),
                    help='comma-separated subset; results merge into an existing manifest.json')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    todo = args.settings.split(',')
    unknown = sorted(set(todo) - set(ARMS))
    if unknown:
        raise SystemExit(f'unknown settings {unknown}')

    mpath = args.out / 'manifest.json'
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {'settings': {}}
    manifest.update({'budget_tokens': BUDGET, 'seed': SEED, 'token_convention': 'len(tokenize(text, add_special_tokens=False)) + 1',
                     'subsample_rule': 'np.random.default_rng(42).permutation over sorted unique ids; shortest prefix with cumulative tokens >= budget'})
    arms, anchor_ref = {}, None
    if (args.out / 'anchor_doc_ids.npy').exists():
        # anchor from an earlier run: every arm processed now must carry the identical anchor
        anchor_ref = {'anchor_ids': np.load(args.out / 'anchor_doc_ids.npy'),
                      'anchor_tok': np.load(args.out / 'anchor_train_tokens_plus1.npy')}
    for raw, arm in ((r, ARMS[r]) for r in todo):
        a = read_arm(args.parquet_root, arm)
        arms[raw] = a
        log(f'{arm}: {a["files"]} files, {a["rows"]:,} rows, {a["rewrite_rows"]:,} rewrite rows, '
            f'{a["source_ids"].size:,} unique source docs, anchor {a["anchor_ids"].size:,} docs / {int(a["anchor_tok"].sum()):,} tok')
        if anchor_ref is None:
            anchor_ref = a
        elif not (np.array_equal(a['anchor_ids'], anchor_ref['anchor_ids'])
                  and np.array_equal(a['anchor_tok'], anchor_ref['anchor_tok'])):
            raise SystemExit(f'anchor doc ids or token counts in {arm} differ from the arms processed before it')
    n_anchor, anchor_total = int(anchor_ref['anchor_ids'].size), int(anchor_ref['anchor_tok'].sum())
    checked = sorted(set(manifest.get('anchor', {}).get('identical_across', [])) | {ARMS[r] for r in todo})
    manifest['anchor'] = {'docs': n_anchor, 'tokens': anchor_total,
                          'matches_recorded': n_anchor == ANCHOR_DOCS and anchor_total == ANCHOR_TOKENS,
                          'identical_across': checked}
    # Written once. Later runs only compare against it, and never rewrite a file that jobs on other
    # nodes may be reading at the same moment.
    for name, arr in (('anchor_doc_ids.npy', anchor_ref['anchor_ids']),
                      ('anchor_train_tokens_plus1.npy', anchor_ref['anchor_tok'])):
        if not (args.out / name).exists():
            tmp = args.out / f'.{name}.tmp.npy'
            np.save(tmp, arr)
            os.replace(tmp, args.out / name)

    union = np.unique(np.concatenate([a['source_ids'] for a in arms.values()]))
    if np.intersect1d(union, anchor_ref['anchor_ids']).size:
        raise SystemExit('a source document is also in the anchor')
    log(f'union of source docs: {union.size:,}; counting tokens with {args.workers} workers')
    shards = union // ROWS_PER_POOL_SHARD
    jobs = []
    for s in np.unique(shards):
        rows = union[shards == s] % ROWS_PER_POOL_SHARD
        jobs.append((int(s), rows, str(args.pool), str(args.tokenizer)))
    lengths = np.empty(union.size, dtype=np.int64)
    starts = {int(s): int(np.searchsorted(shards, s)) for s in np.unique(shards)}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for s, lens in ex.map(_count_shard, jobs, chunksize=1):
            lengths[starts[s]:starts[s] + lens.size] = lens
            done += 1
            if done % 20 == 0:
                log(f'  {done}/{len(jobs)} pool shards')

    for raw, a in arms.items():
        ids = a['source_ids']
        tok = lengths[np.searchsorted(union, ids)]
        total = int(tok.sum())
        d = args.out / raw
        d.mkdir(exist_ok=True)
        np.save(d / 'source_doc_ids.npy', ids)
        np.save(d / 'source_tokens.npy', tok)
        if total > BUDGET:
            perm = np.random.default_rng(SEED).permutation(ids.size)
            cum = np.cumsum(tok[perm])
            k = int(np.searchsorted(cum, BUDGET)) + 1
            keep = perm[:k]
            subsampled = True
        else:
            keep = np.arange(ids.size)
            subsampled = False
        np.save(d / 'selected_doc_ids.npy', ids[keep])
        after = int(tok[keep].sum())
        manifest['settings'][raw] = {
            'counterpart': ARMS[raw], 'published_rows': a['rows'], 'rewrite_rows': a['rewrite_rows'],
            'source_docs_dedup': int(ids.size), 'raw_tokens_before': total,
            'subsampled': subsampled, 'docs_after': int(keep.size), 'tokens_after': after,
            'overshoot_after': after - BUDGET if subsampled else None,
        }
        log(f'{raw}: {ids.size:,} source docs, {total:,} raw tokens -> {keep.size:,} docs, {after:,} tokens')
    # atomic: jobs on other nodes read this file while a count may be finishing
    tmp = mpath.with_name('.manifest.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, mpath)
    log(f'wrote {mpath}')


if __name__ == '__main__':
    main()
