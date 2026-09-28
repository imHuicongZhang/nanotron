#!/usr/bin/env python
"""Global Top-10B selections for the three anchor-free 1.5B quality baselines.

Each setting selects its ENTIRE ~10B TRAIN-token corpus from the same eligible universe as the original
1.5B Quality-Base, with the original selection primitives of projects/rewrite/04_select/select_10b.py
reproduced verbatim (order_desc, fill_to, the SeedSequence(42) children):

    universe   all 99,949,162 rows of the scored pool minus the 50,000-doc validation holdout
               val = sort(default_rng(SeedSequence(42).spawn(8)[0]).choice(N, 50_000, replace=False))
               (no other exclusion: the 5M analysis sample is NOT excluded, as in the original)
    length     TRAIN tokens = tokens-llama2 + 1 (one </s> appended by datatrove at tokenization)
    order      score DESC, ties by tie = default_rng(SeedSequence(42).spawn(8)[1]).permutation(N) ASC
    budget     walk the order, keep whole documents, stop at the first document whose cumulative TRAIN
               tokens reach 10,000,000,000 (np.searchsorted(cumsum, target, 'left'))

    raw_top10b_fineweb_edu   score = fineweb-edu-ranking-v2
    raw_top10b_modernbert    score = modernbert-ranking-v2
    raw_top10b_consensus     score = q = ((fasttext-ranking-v2 + fineweb-edu-ranking-v2
                                           + modernbert-ranking-v2) / 3.0).astype(float32)
                             exactly the q of select_10b.py / select_s5.py: percentiles, not raw scores;
                             no variance term, floor, quota or domain restriction

A fastText order (the Quality-Base rule) is computed as a reference, never written as a setting:
    fasttext_global_top10b   fill_to(fastText order, 1e10)
    fasttext_quality_base    the original construction: shared-top prefix to 5e9, then 5e9 more along the
                             same order (effective cumulative target 5e9 + shared overshoot + 5e9)
and both are compared with the published anchor ids (reproduction check of the conventions).

Doc-set digest conventions (recorded in selection_manifest.json):
    selection_order_sha256  sha256 of the selected scored-pool doc_ids, int64 little-endian, in selection
                            order (score DESC, tie ASC)
    docset_sha256           sha256 of the same ids sorted ascending, int64 little-endian (order-free)
    orig_docset_sha256      sha256 of the sorted orig_doc_id (raw-pool positions), int64 little-endian

Inputs: the flat arrays written by tools/kys_raw/extract_scored_columns.py.

    python tools/kys_raw/select_global_top10b.py --flat <dir> --out <dir> [--anchor-orig-ids anchor_doc_ids.npy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

N = 99_949_162
VAL_SIZE = 50_000
SEED = 42
TARGET = 10_000_000_000
SHARED_TARGET = 5_000_000_000
SETTINGS = {
    'raw_top10b_fineweb_edu': 'fineweb-edu-ranking-v2',
    'raw_top10b_modernbert': 'modernbert-ranking-v2',
    'raw_top10b_consensus': 'q = (fasttext-ranking-v2 + fineweb-edu-ranking-v2 + modernbert-ranking-v2) / 3, float32',
}


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def order_desc(idxs, score, tie):
    """select_10b.order_desc: idxs sorted by score DESC, ties by random priority `tie` ASC."""
    return idxs[np.lexsort((tie[idxs], -score[idxs].astype(np.float64)))]


def fill_to(order, tok, target):
    """select_10b.fill_to: accumulate along `order` until cumsum first >= target; keep the last doc whole."""
    c = np.cumsum(tok[order])
    if c[-1] < target:
        raise SystemExit(f'pool holds only {int(c[-1]):,} tokens < {target:,}')
    i = int(np.searchsorted(c, target, side='left'))
    return order[:i + 1], int(c[i]), int(c[i] - target)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def select(flat: Path):
    tok = np.load(flat / 'tokens_llama2.npy').astype(np.int64) + 1
    ft, fw, mb = (np.load(flat / f'{k}_v2.npy') for k in ('ft', 'fw', 'mb'))
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    ch = np.random.SeedSequence(SEED).spawn(8)
    val = np.sort(np.random.default_rng(ch[0]).choice(N, size=VAL_SIZE, replace=False)).astype(np.int64)
    tie = np.random.default_rng(ch[1]).permutation(N).astype(np.int64)
    alive = np.ones(N, bool)
    alive[val] = False
    alive_idx = np.flatnonzero(alive)
    scores = {'fasttext': ft, 'raw_top10b_fineweb_edu': fw, 'raw_top10b_modernbert': mb, 'raw_top10b_consensus': q}
    res = {}
    for name, sc in scores.items():
        order = order_desc(alive_idx, sc, tie)
        sel, total, over = fill_to(order, tok, TARGET)
        last = sel[-1]
        rec = dict(order=sel, docs=int(sel.size), train_tokens=total, overshoot=over,
                   boundary_score=float(sc[last]), boundary_doc_id=int(last), boundary_doc_train_tokens=int(tok[last]),
                   boundary_tie_cluster_total=int(np.count_nonzero(sc[alive_idx] == sc[last])),
                   boundary_tie_cluster_selected=int(np.count_nonzero(sc[sel] == sc[last])),
                   next_doc_id=int(order[sel.size]), next_doc_score=float(sc[order[sel.size]]))
        if name == 'fasttext':
            sh, sh_tok, sh_over = fill_to(order, tok, SHARED_TARGET)
            rest, qb_tok, qb_over = fill_to(order[sh.size:], tok, SHARED_TARGET)
            rec['quality_base'] = dict(order=order[:sh.size + rest.size], shared_docs=int(sh.size), shared_tokens=sh_tok,
                                       shared_overshoot=sh_over, qb_docs=int(rest.size), qb_tokens=qb_tok, qb_overshoot=qb_over)
        res[name] = rec
        log(f'{name}: {sel.size:,} docs, {total:,} TRAIN tokens (overshoot {over:,}), boundary score {sc[last]!r}')
    return res, dict(val=val, alive_idx=alive_idx, tok=tok, q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--anchor-orig-ids', type=Path, default=None,
                    help='published anchor orig_doc_id list (raw_sources/anchor_doc_ids.npy) for the reproduction check')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    orig = np.load(args.flat / 'orig_doc_id.npy')
    ext = json.loads((args.flat / 'extract_manifest.json').read_text())

    res, aux = select(args.flat)
    res2, _ = select(args.flat)  # deterministic rerun, from scratch
    rerun = {k: sha(res[k]['order']) == sha(res2[k]['order']) for k in res}
    if not all(rerun.values()):
        raise SystemExit(f'non-deterministic selection: {rerun}')
    del res2

    man = {'code': 'tools/kys_raw/select_global_top10b.py', 'scored_pool': {k: ext[k] for k in ('repo', 'revision', 'rows')},
           'flat_sha256': ext['sha256'], 'universe': {'rows': N, 'val_holdout': VAL_SIZE, 'eligible': int(aux['alive_idx'].size),
                                                      'eligible_train_tokens': int(aux['tok'][aux['alive_idx']].sum()),
                                                      'val_doc_ids_sha256': sha(aux['val'])},
           'rules': {'train_tokens': 'tokens-llama2 + 1', 'target': TARGET,
                     'order': 'score DESC (float64 compare), ties by SeedSequence(42).spawn(8)[1] permutation ASC',
                     'cutoff': 'first cumulative >= target, last document whole'},
           'digest_conventions': {'selection_order_sha256': 'sha256 of selected doc_id (scored pool, int64 LE) in selection order',
                                  'docset_sha256': 'sha256 of selected doc_id sorted ascending (int64 LE)',
                                  'orig_docset_sha256': 'sha256 of selected orig_doc_id sorted ascending (int64 LE)'},
           'deterministic_rerun_identical': rerun, 'settings': {}, 'reference': {}}
    for name, r in res.items():
        o = r.pop('order')
        qb = r.pop('quality_base', None)
        r.update(selection_order_sha256=sha(o), docset_sha256=sha(np.sort(o)), orig_docset_sha256=sha(np.sort(orig[o])))
        if name in SETTINGS:
            d = args.out / name
            d.mkdir(exist_ok=True)
            np.save(d / 'selected_doc_ids_order.npy', o)
            np.save(d / 'selected_orig_doc_ids.npy', np.sort(orig[o]))  # consumed by assemble_raw_corpus.py
            man['settings'][name] = {'score': SETTINGS[name], **r}
        else:
            np.save(args.out / 'fasttext_global_top10b_doc_ids_order.npy', o)
            man['reference']['fasttext_global_top10b'] = r
            qo = qb.pop('order')
            np.save(args.out / 'fasttext_quality_base_doc_ids_order.npy', qo)
            qb.update(docs=int(qo.size), train_tokens=int(aux['tok'][qo].sum()), docset_sha256=sha(np.sort(qo)),
                      orig_docset_sha256=sha(np.sort(orig[qo])),
                      global_top10b_subset_of_quality_base=bool(np.isin(o, qo).all()),
                      docs_in_quality_base_not_in_global_top10b=int(qo.size - np.isin(qo, o).sum()))
            if args.anchor_orig_ids:
                anc = np.sort(np.load(args.anchor_orig_ids))
                shared_orig = np.sort(orig[qo[:qb['shared_docs']]])
                qb['shared_top_equals_published_anchor'] = bool(np.array_equal(shared_orig, anc))
                qb['published_anchor_docs'] = int(anc.size)
            man['reference']['fasttext_quality_base_reconstruction'] = qb
    (args.out / 'selection_manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    log(f'wrote {args.out / "selection_manifest.json"}')


if __name__ == '__main__':
    main()
