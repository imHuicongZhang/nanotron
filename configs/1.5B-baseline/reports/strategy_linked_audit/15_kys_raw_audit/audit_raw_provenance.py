#!/usr/bin/env python
"""Provenance audit of the four strategy-linked raw baselines (blab-jhu/KYS-Pre-Rewritten).

For each rewritten strategy it reconstructs three document sets, all in scored-pool `doc_id` space:
    I  the strategy's original REWRITING INPUT (source selection), re-derived here from the original
       selection code (04_select/select_10b.py; 05_select_s5_variants/select_s5.py for lambda=0.5)
    R  the unique source documents of the FINAL rewritten half actually trained on
       (published wytro/Know-Your-Sources/<arm>, rows with source_prompt != 'original')
    S  the raw strategy half actually published (kys/raw_sources/<raw>/selected_doc_ids.npy)
and reports sizes, TRAIN tokens (tokens-llama2 + 1 of the SOURCE document), subset relations, directional
coverage (docs and source-token weighted), Jaccard, and how far R and S drift from I (topic TV/JS, mean
percentiles, document length).

It also re-executes the raw subsampling rule of tools/kys_raw/build_raw_sources.py from R and the pool's
own token counts, and checks the published raw_text/<setting>/ files (local staging copy, sha256 listed in
manifest.json) hold exactly anchor + S.

    python 15_kys_raw_audit/audit_raw_provenance.py --flat <extract_scored_columns out> --out <json>

Inputs are read-only. The original selection outputs (data_rewrite/experiments/...) no longer exist on disk,
so I is re-derived; its correctness is checked by (a) the shared-top reproduction equalling the published
anchor, (b) R being a subset of I, (c) the lambda=0.5 set equalling bit 2 of 06_lambda_grid/lambda_grid.npz.
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
HERE = Path(__file__).resolve().parent.parent
N = 99_949_162
ARMS = {'raw_diversity_oriented': ('diversity_oriented', 'diversity'),
        'raw_disagreement_aware': ('disagreement_aware', 'lambda05'),
        'raw_random': ('wrap_inspired', 'wrap'),
        'raw_rewire_inspired': ('rewire_inspired', 'rewrite')}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def order_desc(idx, score, tie):
    return idx[np.lexsort((tie[idx], -score[idx].astype(np.float64)))]


def fill_to(order, tok, target):
    c = np.cumsum(tok[order])
    i = int(np.searchsorted(c, target, side='left'))
    return order[:i + 1]


def mask(idx):
    m = np.zeros(N, bool)
    m[idx] = True
    return m


def reproduce_inputs(tok, ft, fw, mb, topic):
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3.0).astype(np.float32)
    ch = np.random.SeedSequence(42).spawn(8)
    val = np.sort(np.random.default_rng(ch[0]).choice(N, size=50_000, replace=False))
    tie = np.random.default_rng(ch[1]).permutation(N).astype(np.int64)
    alive = ~mask(val)
    shared = fill_to(order_desc(np.flatnonzero(alive), ft, tie), tok, 5e9)
    rem = np.flatnonzero(alive & ~mask(shared))
    rem_tok = int(tok[rem].sum())
    out = {'val': val, 'shared': shared, 'remaining': rem}
    qf_order = order_desc(np.flatnonzero(alive), ft, tie)[shared.size:]
    out['quality_base_block'] = fill_to(qf_order, tok, 5e9)
    out['quality_first'] = fill_to(qf_order, tok, 10e9)
    out['wrap'] = fill_to(rem[np.random.default_rng(ch[2]).permutation(rem.size)], tok, 10e9)
    out['rewrite'] = fill_to(rem[np.random.default_rng(ch[3]).permutation(rem.size)], tok, 20e9)
    div, rt = [], topic[rem]
    for c in range(24):
        ci = rem[rt == c]
        div.append(fill_to(order_desc(ci, q, tie), tok, 10e9 * (int(tok[ci].sum()) / rem_tok)))
    out['diversity'] = np.concatenate(div)
    assert tok[out['diversity']].sum() >= 10e9  # no top-up branch taken (select_10b.py l.269)

    def top10(score):
        return fill_to(order_desc(rem, score, tie), tok, 0.10 * rem_tok)
    U = np.flatnonzero(mask(top10(ft)) | mask(top10(fw)) | mask(top10(mb)))
    Q30, V90 = float(np.percentile(q[U], 30.0)), float(np.percentile(v[U], 90.0))
    surv = U[(q[U] >= Q30) & (v[U] <= V90)]
    u = q[surv] + 0.5 * np.sqrt(v)[surv]
    out['lambda05'] = fill_to(surv[np.lexsort((tie[surv], -u.astype(np.float64)))], tok, 10e9)
    out['_lambda05_meta'] = dict(U_docs=int(U.size), Q30=Q30, V90=V90, survivors=int(surv.size))
    return out, q


def dist(ids, tok, topic):
    w = np.bincount(topic[ids], weights=tok[ids], minlength=24)
    return w / w.sum()


def tv_js(p, r):
    m = 0.5 * (p + r)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * float(np.abs(p - r).sum()), 0.5 * kl(p, m) + 0.5 * kl(r, m)


def cover(a, b, tok):
    """directional coverage of set a by set b (docs, source-token weighted)."""
    inter = np.intersect1d(a, b, assume_unique=True)
    return dict(docs=float(inter.size / a.size), tokens=float(tok[inter].sum() / tok[a].sum()), n=int(inter.size))


def quality_first(inp, tok, ft, to_doc, anchor):
    """Why Quality-First has no raw arm: Quality-Base = anchor + the fastText-best 5B of Quality-First's own input."""
    from huggingface_hub import HfApi, HfFileSystem
    repo, rev = 'wytro/Know-Your-Sources', HfApi().repo_info('wytro/Know-Your-Sources', repo_type='dataset').sha
    fs = HfFileSystem()
    oid = []
    for p in sorted(x for x in fs.ls(f'datasets/{repo}@{rev}/quality_first', detail=False) if x.endswith('.parquet')):
        with fs.open(p, 'rb', block_size=16 << 20) as fh:
            t = pq.ParquetFile(fh).read(columns=['orig_doc_id', 'source_prompt'])
        rw = np.asarray(t['source_prompt'].to_pylist(), dtype=object) != 'original'
        oid.append(t['orig_doc_id'].to_numpy()[rw])
    R = to_doc(np.unique(np.concatenate(oid)))
    I, QB = np.sort(inp['quality_first']), np.sort(inp['quality_base_block'])
    prof = lambda x: dict(docs=int(x.size), train_tokens=int(tok[x].sum()), mean_ft=float(ft[x].mean()), min_ft=float(ft[x].min()))  # noqa: E731
    return dict(hf_revision=rev, I_quality_first_input=prof(I), R_quality_first_rewritten_sources=prof(R),
                quality_base_block=prof(QB), qb_block_subset_of_I=bool(np.isin(QB, I).all()), R_subset_I=bool(np.isin(R, I).all()),
                qb_block_in_R=cover(QB, R, tok), R_in_qb_block=cover(R, QB, tok), qb_overlap_anchor=int(np.intersect1d(QB, anchor).size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    f = args.flat
    tok = np.load(f / 'tokens_llama2.npy').astype(np.int64) + 1
    ft, fw, mb = (np.load(f / f'{k}_v2.npy') for k in ('ft', 'fw', 'mb'))
    topic = np.load(f / 'topic_code.npy')
    orig = np.load(f / 'orig_doc_id.npy')
    assert np.all(np.diff(orig) > 0), 'orig_doc_id must be strictly increasing in doc_id'
    to_doc = lambda o: np.searchsorted(orig, o)  # noqa: E731  (orig -> doc_id; exact because strictly increasing)

    inp, q = reproduce_inputs(tok, ft, fw, mb, topic)
    anchor_orig = np.sort(np.load(K / 'raw_sources' / 'anchor_doc_ids.npy'))
    anchor = to_doc(anchor_orig)
    assert np.array_equal(orig[anchor], anchor_orig)
    rep = {'flat': str(f), 'checks': {}, 'inputs': {}, 'settings': {}}
    rep['checks']['shared_top_equals_published_anchor'] = bool(np.array_equal(np.sort(inp['shared']), anchor))
    rep['checks']['anchor'] = dict(docs=int(anchor.size), train_tokens=int(tok[anchor].sum()),
                                   orig_docset_sha256=sha(anchor_orig), docset_sha256=sha(anchor))
    grid = np.load(HERE / '06_lambda_grid' / 'lambda_grid.npz')
    bit2 = np.sort(grid['doc_id'][(grid['pattern'] >> 2) & 1 == 1])
    rep['checks']['lambda05_equals_lambda_grid_bit2'] = bool(np.array_equal(np.sort(inp['lambda05']), bit2))
    rep['checks']['lambda05_meta'] = inp.pop('_lambda05_meta')
    pool_d = dist(inp['remaining'], tok, topic)
    for k in ('wrap', 'rewrite', 'diversity', 'lambda05'):
        a = np.sort(inp[k])
        rep['inputs'][k] = dict(docs=int(a.size), train_tokens=int(tok[a].sum()), docset_sha256=sha(a),
                                overlap_anchor=int(np.intersect1d(a, anchor).size),
                                overlap_val=int(np.intersect1d(a, inp['val']).size))

    for raw, (arm, key) in ARMS.items():
        I = np.sort(inp[key])
        R = to_doc(np.load(K / 'raw_sources' / raw / 'source_doc_ids.npy'))
        S = np.sort(to_doc(np.load(K / 'raw_sources' / raw / 'selected_doc_ids.npy')))
        src_tok = np.load(K / 'raw_sources' / raw / 'source_tokens.npy')
        d = dict(counterpart=arm, input_key=key)
        d['sizes'] = {n: dict(docs=int(x.size), train_tokens=int(tok[x].sum())) for n, x in (('I', I), ('R', R), ('S', S))}
        d['R_subset_I'] = bool(np.isin(R, I).all())
        d['S_subset_R'] = bool(np.isin(S, R).all())
        d['S_overlap_anchor'] = int(np.intersect1d(S, anchor).size)
        d['I_minus_R'] = dict(docs=int(I.size - R.size), train_tokens=int(tok[I].sum() - tok[R].sum()))
        d['source_tokens_match_pool'] = bool(np.array_equal(src_tok, tok[R]))
        # re-execute build_raw_sources.py's budget rule on (R, pool tokens)
        perm = np.random.default_rng(42).permutation(R.size)
        k_ = int(np.searchsorted(np.cumsum(tok[R][perm]), 5_000_000_000)) + 1
        d['subsample_rule_reproduces_S'] = bool(np.array_equal(np.sort(R[perm[:k_]]), S))
        d['coverage'] = {'S_in_R': cover(S, R, tok), 'R_in_S': cover(R, S, tok), 'S_in_I': cover(S, I, tok),
                         'I_in_S': cover(I, S, tok), 'R_in_I': cover(R, I, tok), 'I_in_R': cover(I, R, tok)}
        d['jaccard'] = {'S_R': float(S.size / R.size), 'S_I': float(S.size / I.size), 'R_I': float(R.size / I.size)}
        pI = dist(I, tok, topic)
        d['topic_shift'] = {}
        for n, x in (('R_vs_I', R), ('S_vs_I', S), ('S_vs_R', S)):
            ref = pI if n.endswith('_I') else dist(R, tok, topic)
            tv, js = tv_js(dist(x, tok, topic), ref)
            d['topic_shift'][n] = dict(tv=tv, js_bits=js)
        tv, js = tv_js(pI, pool_d)
        d['topic_shift']['I_vs_remaining_pool'] = dict(tv=tv, js_bits=js)
        d['profile'] = {n: dict(mean_ft=float(ft[x].mean()), mean_fw=float(fw[x].mean()), mean_mb=float(mb[x].mean()),
                                mean_q=float(q[x].mean()), mean_train_tokens=float(tok[x].mean()),
                                median_train_tokens=float(np.median(tok[x])))
                        for n, x in (('I', I), ('R', R), ('S', S), ('I_minus_R', np.setdiff1d(I, R, assume_unique=True)))}
        # published raw_text (local staging copy of the Hub files)
        files = sorted(glob.glob(str(K / 'hf_stage' / 'raw_text' / raw / 'part-*.parquet')))
        oid, srcs, rows = [], [], []
        for p in files:
            t = pq.read_table(p, columns=['orig_doc_id', 'source'])
            oid.append(t['orig_doc_id'].to_numpy())
            srcs.append(np.asarray(t['source'].to_pylist(), dtype=object))
            rows.append(t.num_rows)
        oid, srcs = np.concatenate(oid), np.concatenate(srcs)
        pub = to_doc(oid)
        d['published'] = dict(files=len(files), rows_per_file=rows, rows=int(oid.size),
                              train_tokens=int(tok[pub].sum()),
                              anchor_rows=int((srcs == 'anchor').sum()), strategy_rows=int((srcs == 'strategy').sum()),
                              docset_equals_anchor_plus_S=bool(np.array_equal(np.sort(pub), np.union1d(anchor, S))),
                              anchor_rows_equal_anchor=bool(np.array_equal(np.sort(pub[srcs == 'anchor']), anchor)),
                              duplicates=int(oid.size - np.unique(oid).size),
                              file_order_orig_sha256=sha(oid))
        rep['settings'][raw] = d
        print(raw, json.dumps({k: d[k] for k in ('sizes', 'R_subset_I', 'S_subset_R', 'subsample_rule_reproduces_S')}), flush=True)
    rep['quality_first_vs_quality_base'] = quality_first(inp, tok, ft, to_doc, anchor)
    args.out.write_text(json.dumps(rep, indent=1) + '\n')
    print('checks', json.dumps(rep['checks']))


if __name__ == '__main__':
    main()
