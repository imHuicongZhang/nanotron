#!/usr/bin/env python
"""Independent reconstruction of the four strategy-linked raw strategy halves (read-only).

Unlike audit_raw_provenance.py, nothing here is read from the raw build's own outputs (kys/raw_sources/):

  R      unique orig_doc_id of the non-anchor rows (source_prompt != 'original') of the rewritten arm, read
         directly from the local copy of wytro/Know-Your-Sources/<arm>/*.parquet (kys/hf_parquet; its Hub
         revision is pinned separately by sha256, pin_rewritten_revision.py)
  tokens TRAIN tokens = tokens-llama2 + 1 of the SOURCE document, from the scored pool (flat arrays of
         tools/kys_raw/extract_scored_columns.py), joined by orig_doc_id
  S'     the documented rule applied to R: ids sorted ascending, perm = default_rng(42).permutation(|R|),
         keep the shortest prefix of ids[perm] whose cumulative TRAIN tokens reach 5e9
  pub    the published raw_text/<setting>/part-*.parquet (local staging copy; tokenize_raw_text.sh checks
         every file's sha256 against manifest.json @ ed09db2a): rows with source == 'strategy' / 'anchor'

and compares S' with pub (set + digests), the anchor with the arm's anchor, and reports the one-to-many
source/rewrite structure and length differences.

    python 15_kys_raw_audit/independent_rebuild.py --flat <dir> --out independent_rebuild.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

K = Path('/projects/bvandur1/zhuicon1/kys')
PAIRS = {'raw_diversity_oriented': 'diversity_oriented', 'raw_disagreement_aware': 'disagreement_aware',
         'raw_random': 'wrap_inspired', 'raw_rewire_inspired': 'rewire_inspired'}
BUDGET = 5_000_000_000


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    orig = np.load(a.flat / 'orig_doc_id.npy')                      # strictly increasing in doc_id
    tok = np.load(a.flat / 'tokens_llama2.npy').astype(np.int64) + 1
    assert np.all(np.diff(orig) > 0)

    def train_tokens(ids):
        pos = np.searchsorted(orig, ids)
        ok = (pos < orig.size) & (orig[np.minimum(pos, orig.size - 1)] == ids)
        if not ok.all():
            raise SystemExit(f'{int((~ok).sum())} ids not in the scored pool')
        return tok[pos]

    res, anchors = {}, {}
    for raw, arm in PAIRS.items():
        src, anc, rw_tok, prompts = [], [], [], []
        for f in sorted(glob.glob(str(K / 'hf_parquet' / arm / '*.parquet'))):
            t = pq.read_table(f, columns=['orig_doc_id', 'source_prompt', 'train_tokens'])
            sp = np.asarray(t['source_prompt'].to_pylist(), dtype=object)
            o = t['orig_doc_id'].to_numpy().astype(np.int64)
            m = sp == 'original'
            anc.append(o[m]); src.append(o[~m]); rw_tok.append(t['train_tokens'].to_numpy()[~m].astype(np.int64) + 1)
            prompts.append(sp[~m])
        anc, src, rw_tok, prompts = np.concatenate(anc), np.concatenate(src), np.concatenate(rw_tok), np.concatenate(prompts)
        anchors[arm] = np.sort(anc)
        R, per_src = np.unique(src, return_counts=True)
        Rtok = train_tokens(R)
        perm = np.random.default_rng(42).permutation(R.size)
        cum = np.cumsum(Rtok[perm])
        k = int(np.searchsorted(cum, BUDGET)) + 1
        S = R[perm[:k]]

        pub_s, pub_a, order = [], [], []
        for f in sorted(glob.glob(str(K / 'hf_stage' / 'raw_text' / raw / 'part-*.parquet'))):
            t = pq.read_table(f, columns=['orig_doc_id', 'source'])
            o = t['orig_doc_id'].to_numpy().astype(np.int64)
            s = np.asarray(t['source'].to_pylist(), dtype=object)
            pub_s.append(o[s == 'strategy']); pub_a.append(o[s == 'anchor']); order.append(o)
        pub_s, pub_a, order = np.concatenate(pub_s), np.concatenate(pub_a), np.concatenate(order)
        inS = np.isin(src, S)
        res[raw] = {
            'counterpart': arm,
            'rewritten_rows': int(src.size), 'rewrite_prompts': {p: int((prompts == p).sum()) for p in np.unique(prompts)},
            'R_docs': int(R.size), 'R_train_tokens': int(Rtok.sum()),
            'R_sources_with_n_outputs': {int(n): int((per_src == n).sum()) for n in np.unique(per_src)},
            'rule': {'k_docs': k, 'train_tokens': int(cum[k - 1]), 'overshoot': int(cum[k - 1] - BUDGET),
                     'prefix_k_minus_1_tokens': int(cum[k - 2])},
            'S_rebuilt_sorted_sha256': sha(np.sort(S)),
            'published_strategy_rows': int(pub_s.size), 'published_anchor_rows': int(pub_a.size),
            'published_strategy_sorted_sha256': sha(np.sort(pub_s)),
            'S_rebuilt_equals_published': bool(np.array_equal(np.sort(S), np.sort(pub_s))),
            'published_anchor_equals_arm_anchor': bool(np.array_equal(np.sort(pub_a), anchors[arm])),
            'anchor_strategy_overlap': int(np.intersect1d(pub_a, pub_s).size),
            'duplicates_in_published': int(order.size - np.unique(order).size),
            'published_file_order_sha256': sha(order),
            'published_train_tokens': int(train_tokens(np.sort(order)).sum()),
            'coverage': {
                'S_in_R_docs': float(np.isin(S, R).mean()),
                'R_in_S_docs': float(S.size / R.size),
                'R_in_S_source_tokens': float(train_tokens(np.sort(S)).sum() / Rtok.sum()),
                'rewritten_rows_whose_source_in_S': float(inS.mean()),
                'rewritten_output_tokens_whose_source_in_S': float(rw_tok[inS].sum() / rw_tok.sum()),
            },
            'lengths': {
                'rewritten_half_output_train_tokens': int(rw_tok.sum()),
                'mean_output_tokens_per_rewrite_row': float(rw_tok.mean()),
                'mean_source_tokens_R': float(Rtok.mean()),
                'mean_source_tokens_S': float(train_tokens(np.sort(S)).mean()),
                'rewritten_rows_per_source_mean': float(src.size / R.size),
                'output_to_source_token_ratio_R': float(rw_tok.sum() / Rtok.sum()),
            },
        }
        print(raw, json.dumps({k: v for k, v in res[raw].items() if k in (
            'R_docs', 'rule', 'S_rebuilt_equals_published', 'published_anchor_equals_arm_anchor', 'anchor_strategy_overlap',
            'duplicates_in_published', 'published_train_tokens')}), flush=True)
    base = next(iter(anchors.values()))
    out = {'arms_anchor_identical': all(np.array_equal(v, base) for v in anchors.values()),
           'anchor_docs': int(base.size), 'anchor_train_tokens': int(train_tokens(base).sum()),
           'anchor_sorted_orig_sha256': sha(base), 'settings': res}
    a.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
