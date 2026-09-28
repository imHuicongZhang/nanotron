#!/usr/bin/env python
"""Selection report for the three global Top-10B baselines, against the existing fastText Quality-Base.

Reads the outputs of extract_scored_columns.py (--flat) and select_global_top10b.py (--selection) and writes
selection_report.json with:
  * percentile verification: each *-ranking-v2 column recomputed from the raw score column as
    scipy.stats.rankdata(raw.astype(float64), 'average') / 99,949,162 -> float32 (00_TMP/clean_v2_ranks.py)
    and compared element-wise with the stored column (reference population, direction, tie rule);
    the consensus order is re-derived from the recomputed percentiles and compared with the selection
  * four-way overlap (quality_base, fineweb_edu, modernbert, consensus): pairwise intersections, both
    directional coverage rates (documents and TRAIN-token weighted), Jaccard, the 4-way intersection,
    and overlap with the shared anchor
  * topic distributions over the 24 WebOrganizer labels for the eligible universe and each set, TRAIN-token
    weighted and document weighted, with TV = 0.5 * sum|p - r| and JS divergence in bits (log base 2,
    bounded by 1) of each set against the eligible universe
  * score and length profiles

    python tools/kys_raw/report_global_top10b.py --flat <dir> --selection <dir> --scored-local <dir with
        merged_clean_*.parquet> --anchor-orig-ids anchor_doc_ids.npy
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

N = 99_949_162
SETS = ['quality_base', 'raw_top10b_fineweb_edu', 'raw_top10b_modernbert', 'raw_top10b_consensus']
RAW = {'ft': 'fasttext', 'fw': 'fineweb-edu', 'mb': 'modernbert'}


def read_raw(local: Path):
    out = {k: np.empty(N, np.float64) for k in RAW}

    def one(i):
        t = pq.read_table(local / f'merged_clean_{i:05d}.parquet', columns=['doc_id', *RAW.values()])
        return i, t
    with ThreadPoolExecutor(12) as ex:
        for i, t in ex.map(one, range(200)):
            d = t['doc_id'].to_numpy()
            assert d[0] == i * 500_000 and np.all(np.diff(d) == 1)
            for k, c in RAW.items():
                assert t[c].null_count == 0
                out[k][d[0]:d[-1] + 1] = t[c].to_numpy().astype(np.float64)
    return out


def tv_js(p, r):
    m = 0.5 * (p + r)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * float(np.abs(p - r).sum()), 0.5 * kl(p, m) + 0.5 * kl(r, m)


def main():
    from scipy.stats import rankdata
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--scored-local', type=Path, required=True)
    ap.add_argument('--anchor-orig-ids', type=Path, required=True)
    args = ap.parse_args()
    f, s = args.flat, args.selection
    tok = np.load(f / 'tokens_llama2.npy').astype(np.int64) + 1
    v2 = {k: np.load(f / f'{k}_v2.npy') for k in RAW}
    topic = np.load(f / 'topic_code.npy')
    vocab = json.loads((f / 'topic_vocab.json').read_text())
    orig = np.load(f / 'orig_doc_id.npy')
    man = json.loads((s / 'selection_manifest.json').read_text())
    rep = {'definitions': {
        'train_tokens': 'tokens-llama2 + 1 per document',
        'coverage_A_by_B': '|A ∩ B| / |A| (docs) and TRAIN tokens of A ∩ B / TRAIN tokens of A (tokens)',
        'jaccard': '|A ∩ B| / |A ∪ B| (documents)',
        'tv': '0.5 * sum_t |p_t - r_t| over the 24 topic labels',
        'js_bits': 'Jensen-Shannon divergence, log base 2 (0 = identical, 1 = disjoint support)',
        'eligible_universe': 'all 99,949,162 scored rows minus the 50,000 validation holdout',
        'quality_base': 'the original 1.5B Quality-Base doc set (anchor + next 5B by fastText), reconstructed; '
                        'docset digest equals wytro/Know-Your-Sources/quality_base metadata doc_id_digest_sha256'}}

    # ---- percentile verification ----------------------------------------------------------------
    raw = read_raw(args.scored_local)
    pv, rec_pct = {}, {}
    for k in RAW:
        rec = (rankdata(raw[k], method='average') / N).astype(np.float32)
        rec_pct[k] = rec
        diff = rec != v2[k]
        a, b = np.argmax(raw[k]), np.argmin(raw[k])
        pv[k] = dict(column=f'{RAW[k]}-ranking-v2', exact_equal=int((~diff).sum()), mismatches=int(diff.sum()),
                     max_abs_diff=float(np.max(np.abs(rec.astype(np.float64) - v2[k]))),
                     direction_higher_raw_higher_pct=bool(v2[k][a] > v2[k][b]),
                     distinct_raw=int(np.unique(raw[k]).size), stored_min=float(v2[k].min()), stored_max=float(v2[k].max()))
    del raw
    q_rec = ((rec_pct['ft'] + rec_pct['fw'] + rec_pct['mb']) / 3.0).astype(np.float32)
    q = ((v2['ft'] + v2['fw'] + v2['mb']) / 3.0).astype(np.float32)
    del rec_pct
    cons = np.load(s / 'raw_top10b_consensus' / 'selected_doc_ids_order.npy')
    ch = np.random.SeedSequence(42).spawn(8)
    val = np.sort(np.random.default_rng(ch[0]).choice(N, size=50_000, replace=False))
    tie = np.random.default_rng(ch[1]).permutation(N).astype(np.int64)
    alive = np.ones(N, bool)
    alive[val] = False
    alive_idx = np.flatnonzero(alive)
    o = alive_idx[np.lexsort((tie[alive_idx], -q_rec[alive_idx].astype(np.float64)))]
    k_ = int(np.searchsorted(np.cumsum(tok[o]), 10_000_000_000)) + 1
    rep['percentile_verification'] = dict(
        per_scorer=pv, reference_population=f'all {N:,} scored rows (validation docs included), denominator {N:,}',
        consensus_q_from_recomputed_equals_stored=bool(np.array_equal(q_rec, q)),
        consensus_selection_from_recomputed_percentiles_identical=bool(np.array_equal(o[:k_], cons)),
        consensus_is_mean_of_percentiles_not_raw=True)
    del q_rec, o

    # ---- sets --------------------------------------------------------------------------------------
    ids = {'quality_base': np.load(s / 'fasttext_quality_base_doc_ids_order.npy'),
           **{n: np.load(s / n / 'selected_doc_ids_order.npy') for n in SETS[1:]}}
    srt = {n: np.sort(a) for n, a in ids.items()}
    anchor = np.searchsorted(orig, np.sort(np.load(args.anchor_orig_ids)))
    sizes = {n: dict(docs=int(a.size), train_tokens=int(tok[a].sum()),
                     anchor_docs_inside=int(np.intersect1d(a, anchor, assume_unique=True).size),
                     anchor_share_of_tokens=float(tok[np.intersect1d(a, anchor, assume_unique=True)].sum() / tok[a].sum()))
             for n, a in srt.items()}
    rep['sets'] = sizes
    pair = {}
    for i, a in enumerate(SETS):
        for b in SETS[i + 1:]:
            inter = np.intersect1d(srt[a], srt[b], assume_unique=True)
            ti = int(tok[inter].sum())
            pair[f'{a}|{b}'] = dict(intersection_docs=int(inter.size), intersection_train_tokens=ti,
                                    cov_a_by_b_docs=inter.size / srt[a].size, cov_b_by_a_docs=inter.size / srt[b].size,
                                    cov_a_by_b_tokens=ti / sizes[a]['train_tokens'], cov_b_by_a_tokens=ti / sizes[b]['train_tokens'],
                                    jaccard=inter.size / (srt[a].size + srt[b].size - inter.size))
    rep['pairwise_overlap'] = pair
    all4 = srt[SETS[0]]
    for n in SETS[1:]:
        all4 = np.intersect1d(all4, srt[n], assume_unique=True)
    union = np.unique(np.concatenate(list(srt.values())))
    rep['four_way'] = dict(intersection_docs=int(all4.size), intersection_train_tokens=int(tok[all4].sum()),
                           union_docs=int(union.size), union_train_tokens=int(tok[union].sum()))

    # ---- topics --------------------------------------------------------------------------------------
    def dists(x):
        tw = np.bincount(topic[x], weights=tok[x], minlength=24)
        dw = np.bincount(topic[x], minlength=24).astype(float)
        return tw / tw.sum(), dw / dw.sum()
    pool_t, pool_d = dists(alive_idx)
    top = {'labels': vocab, 'eligible_universe': {'token_weighted': pool_t.tolist(), 'doc_weighted': pool_d.tolist()}}
    for n, a in srt.items():
        t_, d_ = dists(a)
        tvt, jst = tv_js(t_, pool_t)
        tvd, jsd = tv_js(d_, pool_d)
        top[n] = dict(token_weighted=t_.tolist(), doc_weighted=d_.tolist(), vs_eligible_token_weighted=dict(tv=tvt, js_bits=jst),
                      vs_eligible_doc_weighted=dict(tv=tvd, js_bits=jsd))
    rep['topics'] = top

    # ---- profiles --------------------------------------------------------------------------------------
    rep['profiles'] = {n: dict(mean_ft_v2=float(v2['ft'][a].mean()), mean_fw_v2=float(v2['fw'][a].mean()),
                               mean_mb_v2=float(v2['mb'][a].mean()), mean_q=float(q[a].mean()),
                               min_ft_v2=float(v2['ft'][a].min()), min_fw_v2=float(v2['fw'][a].min()),
                               min_mb_v2=float(v2['mb'][a].min()), min_q=float(q[a].min()),
                               mean_train_tokens=float(tok[a].mean()), median_train_tokens=float(np.median(tok[a])))
                       for n, a in [('eligible_universe', alive_idx), *srt.items()]}
    rep['selection_manifest'] = man
    (s / 'selection_report.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({'percentiles': rep['percentile_verification'], 'sets': sizes, 'four_way': rep['four_way']}, indent=1))


if __name__ == '__main__':
    main()
