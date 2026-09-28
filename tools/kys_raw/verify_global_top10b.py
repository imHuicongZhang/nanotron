#!/usr/bin/env python
"""Independent check of the exported global Top-10B corpora (publish_global_top10b.py --stage export output).

Per setting, reading only the 16 exported files, the selection arrays, the flat scored-pool arrays and the raw pool:

  membership  exported doc_id set == selected doc_id set (docset_sha256 recomputed), no duplicate doc_id, row count
              == selection docs, orig_doc_id == flat orig_doc_id[doc_id] for every row, source == 'selected'
  order       exported row order == the assembled shuffle order (assemble_global_top10b.py shuffled/part_*.parquet
              concatenated); file_order_sha256 = sha256 of doc_id int64 LE in file order
  tokens      sum(tokens-llama2[doc_id] + 1) over exported rows == selection train_tokens
  files       sha256 and row count of every exported file == its _export.json record
  text        byte-exact equality with the raw pool text at orig_doc_id for a seeded random sample of rows
              (default 2,000 per file = 32,000 per setting; every file sampled). The assembly step already
              re-tokenized every document and matched tokens-llama2 exactly.

    python tools/kys_raw/verify_global_top10b.py --stage-dir <out>/stage/raw_text --selection <dir> --flat <dir> \
        --corpus <assemble out> --pool <raw 100M pool> --out <verify.json> [--per-file 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import RAW_SETTINGS  # noqa: E402

SETTINGS = [s for s, v in RAW_SETTINGS.items() if v['family'] == 'global_top10b']
POOL_FILE = 'dclm_refinedweb_sample_{:05d}.parquet'
ROWS_PER_POOL_SHARD = 500_000


def sha_ids(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def sha_file(p, buf=16 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def check(s, a, orig_all, tok_all, sel_man):
    want = sel_man['settings'][s]
    d = a.stage_dir / s
    exp = json.loads((d / '_export.json').read_text())
    files = sorted(d.glob('part-*.parquet'))
    r = {'files': len(files), 'problems': []}
    if [p.name for p in files] != sorted(exp['files']) or len(files) != 16:
        r['problems'].append(f'file list {[p.name for p in files]} != _export.json')
    for p in files:
        rec = exp['files'][p.name]
        if sha_file(p) != rec['sha256'] or pq.ParquetFile(p).metadata.num_rows != rec['rows']:
            r['problems'].append(f'{p.name}: sha256/rows differ from _export.json')
    ids, origs, rows_per_file = [], [], []
    for p in files:
        t = pq.read_table(p, columns=['doc_id', 'orig_doc_id', 'source'])
        if set(t['source'].unique().to_pylist()) != {'selected'}:
            r['problems'].append(f'{p.name}: source values {t["source"].unique().to_pylist()}')
        ids.append(t['doc_id'].to_numpy())
        origs.append(t['orig_doc_id'].to_numpy())
        rows_per_file.append(t.num_rows)
    ids, origs = np.concatenate(ids), np.concatenate(origs)
    sel = np.load(a.selection / s / 'selected_doc_ids_order.npy')
    r['rows'] = int(ids.size)
    r['rows_per_file'] = rows_per_file
    r['duplicate_doc_ids'] = int(ids.size - np.unique(ids).size)
    r['docset_sha256'] = sha_ids(np.sort(ids))
    r['orig_docset_sha256'] = sha_ids(np.sort(origs))
    r['file_order_sha256'] = sha_ids(ids)
    r['orig_doc_id_consistent'] = bool(np.array_equal(origs, orig_all[ids]))
    r['train_tokens'] = int((tok_all[ids].astype(np.int64) + 1).sum())
    shuf = np.concatenate([pq.read_table(p, columns=['doc_id'])['doc_id'].to_numpy()
                           for p in sorted((a.corpus / s / 'shuffled').glob('part_*.parquet'))])
    r['order_equals_assembled_shuffle'] = bool(np.array_equal(ids, shuf))
    r['order_differs_from_selection_order'] = bool(not np.array_equal(ids, sel))
    for k, got, exp_v in [('docs', r['rows'], want['docs']), ('docset_sha256', r['docset_sha256'], want['docset_sha256']),
                          ('orig_docset_sha256', r['orig_docset_sha256'], want['orig_docset_sha256']),
                          ('train_tokens', r['train_tokens'], want['train_tokens'])]:
        if got != exp_v:
            r['problems'].append(f'{k}: {got} != selection {exp_v}')
    if r['duplicate_doc_ids'] or not r['orig_doc_id_consistent'] or not r['order_equals_assembled_shuffle']:
        r['problems'].append('duplicates / orig_doc_id / order check failed')

    # text fidelity: seeded sample of rows in every file, compared byte-exact with the raw pool
    rng = np.random.default_rng(12345)
    picks = []  # (file index, row in file)
    for fi, n in enumerate(rows_per_file):
        for row in np.sort(rng.choice(n, size=min(a.per_file, n), replace=False)):
            picks.append((fi, int(row)))
    exported = {}
    for fi, p in enumerate(files):
        rows = [row for f, row in picks if f == fi]
        t = pq.read_table(p, columns=['orig_doc_id', 'text']).take(pa.array(rows))
        for o, txt in zip(t['orig_doc_id'].to_pylist(), t['text'].to_pylist()):
            exported[o] = txt
    return r, exported


def text_check(results, samples, pool):
    """One pass over the raw pool for the sampled rows of all settings."""
    by_shard = {}
    for s, ex in samples.items():
        for o in ex:
            by_shard.setdefault(o // ROWS_PER_POOL_SHARD, set()).add(o)
    mism = {s: [] for s in samples}
    for sh, os_ in sorted(by_shard.items()):
        os_ = sorted(os_)
        col = pq.read_table(pool / POOL_FILE.format(sh), columns=['text']).column('text')
        for o, txt in zip(os_, col.take(pa.array([o % ROWS_PER_POOL_SHARD for o in os_])).to_pylist()):
            b = txt.encode('utf-8')
            for s, ex in samples.items():
                if o in ex and ex[o].encode('utf-8') != b:
                    mism[s].append(o)
    for s, r in results.items():
        r['text_sample'] = {'rows_checked': len(samples[s]), 'pool_shards_read': len({o // ROWS_PER_POOL_SHARD for o in samples[s]}),
                            'mismatches': len(mism[s]), 'mismatch_examples': mism[s][:10],
                            'rng': 'default_rng(12345), per file without replacement'}
        if mism[s]:
            r['problems'].append(f'{len(mism[s])} sampled texts differ from the raw pool')
        r['ok'] = not r['problems']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage-dir', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-file', type=int, default=2000)
    ap.add_argument('--settings', default=','.join(SETTINGS))
    a = ap.parse_args()
    orig_all = np.load(a.flat / 'orig_doc_id.npy')
    tok_all = np.load(a.flat / 'tokens_llama2.npy')
    sel_man = json.loads((a.selection / 'selection_manifest.json').read_text())
    res, samples = {}, {}
    for s in a.settings.split(','):
        res[s], samples[s] = check(s, a, orig_all, tok_all, sel_man)
        print(s, 'structural checks done', res[s]['problems'] or 'no problems', flush=True)
    text_check(res, samples, a.pool)
    for s, r in res.items():
        print(s, 'OK' if r['ok'] else r['problems'], {k: r[k] for k in ('rows', 'train_tokens', 'file_order_sha256')},
              r['text_sample'], flush=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    sys.exit(0 if all(r['ok'] for r in res.values()) else 1)


if __name__ == '__main__':
    main()
