#!/usr/bin/env python
"""Does the existing Quality-Base supply the raw comparison for Quality-First? (read-only)

Re-derives from the select_10b.py rules (val holdout, SeedSequence(42) tie-break, fastText-v2 DESC, fill_to):
    anchor   = fastText prefix to 5e9 over (all - val)
    QB block = next 5e9 along the same order          (Quality-Base = anchor + QB block)
    QF input = next 10e9 along the same order         (Quality-First rewriting input)
and compares them with R_QF = unique source orig_doc_id of the non-anchor rows of the published Quality-First
rewritten mixture (wytro/Know-Your-Sources@9e5ff241/quality_first; qf_sources.npz from qf_sources.py).

    python 15_kys_raw_audit/quality_first_vs_base.py --flat <dir> --qf <qf_sources.npz> --out quality_first_vs_base.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N, VAL = 99_949_162, 50_000


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def fill_to(order, tok, target):
    c = np.cumsum(tok[order])
    i = int(np.searchsorted(c, target, side='left'))
    return order[:i + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--qf', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    orig = np.load(a.flat / 'orig_doc_id.npy')
    tok = np.load(a.flat / 'tokens_llama2.npy').astype(np.int64) + 1
    ft = np.load(a.flat / 'ft_v2.npy')
    ch = np.random.SeedSequence(42).spawn(8)
    val = np.random.default_rng(ch[0]).choice(N, size=VAL, replace=False)
    tie = np.random.default_rng(ch[1]).permutation(N).astype(np.int64)
    alive = np.ones(N, bool); alive[val] = False
    idx = np.flatnonzero(alive)
    order = idx[np.lexsort((tie[idx], -ft[idx].astype(np.float64)))]
    anchor = fill_to(order, tok, 5e9)
    rest = order[anchor.size:]
    qb = fill_to(rest, tok, 5e9)
    qf = fill_to(rest, tok, 10e9)

    z = np.load(a.qf)
    R = np.unique(z['src'])                                  # orig_doc_id space
    pos = np.searchsorted(orig, R)
    assert np.array_equal(orig[pos], R), 'R_QF source not in scored pool'
    Rd = np.sort(pos)                                        # doc_id space
    qb_s, qf_s, anc_s = np.sort(qb), np.sort(qf), np.sort(anchor)
    rng = np.random.default_rng(42)                          # the strategy-linked rule, applied hypothetically to R_QF
    perm = rng.permutation(Rd.size)
    rand_half = np.sort(fill_to(Rd[perm], tok, 5e9))

    def cov(A, B):  # |A∩B|/|A| docs and TRAIN tokens
        m = np.isin(A, B, assume_unique=True)
        return {'docs': float(m.mean()), 'tokens': float(tok[A[m]].sum() / tok[A].sum())}

    def prof(A):
        return {'docs': int(A.size), 'train_tokens': int(tok[A].sum()), 'mean_ft_pct': float(ft[A].mean()),
                'min_ft_pct': float(ft[A].min()), 'mean_train_tokens_per_doc': float(tok[A].mean())}

    out = {
        'rewritten_revision': 'wytro/Know-Your-Sources@9e5ff24149c2957c30f0c8fdd051a8eb3b75baad (quality_first)',
        'qf_arm_anchor_equals_rule_anchor': bool(np.array_equal(np.sort(np.searchsorted(orig, np.sort(z['anchor']))), anc_s)),
        'quality_base_docset_sha256': sha(np.sort(np.concatenate([anchor, qb]))),
        'profiles': {'QB_block': prof(qb_s), 'QF_input': prof(qf_s), 'R_QF': prof(Rd),
                     'hypothetical_random_half_of_R_QF': prof(rand_half)},
        'QB_block_subset_of_QF_input': bool(np.isin(qb_s, qf_s, assume_unique=True).all()),
        'QB_block_covered_by_R_QF': cov(qb_s, Rd), 'R_QF_covered_by_QB_block': cov(Rd, qb_s),
        'QF_input_covered_by_R_QF': cov(qf_s, Rd), 'R_QF_subset_of_QF_input': bool(np.isin(Rd, qf_s, assume_unique=True).all()),
        'QF_rewrite_rows': int(z['src'].size), 'QF_rewrite_prompts': {p: int((z['prompts'] == p).sum()) for p in np.unique(z['prompts'])},
        'QF_rewritten_train_tokens': int(z['rtok'].sum()),
        'QF_output_tokens_whose_source_in_QB_block': float(z['rtok'][np.isin(np.searchsorted(orig, z['src']), qb_s)].sum() / z['rtok'].sum()),
    }
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
