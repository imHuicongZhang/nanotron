#!/usr/bin/env python3
"""READ-ONLY audit: reconstruct scorer-training / analysis-sample membership IDs and
compute overlaps with the published scored pool and the KYS selections.

ID spaces:
  orig  = position in raw 100M DCLM-RefinedWeb pool = file_idx*500000 + row  (== orig_doc_id)
  docid = contiguous doc_id of the 99,949,162-row scored pool
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
PPL = Path('/weka/scratch/jhu/bvandur1/zhuicon1/datasets/ppl-dsai')
FLAT = Path('/projects/bvandur1/zhuicon1/kys/top10b/flat')
SEL = Path('/projects/bvandur1/zhuicon1/kys/top10b/selection')
RAWSRC = Path('/projects/bvandur1/zhuicon1/kys/raw_sources')
RW = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite')
N_RAW, N_POOL = 100_000_000, 99_949_162
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

def parse_fr(s):
    m = re.fullmatch(r'f(\d+):r(\d+)', s); return int(m.group(1)) * 500_000 + int(m.group(2))

def ids_from_jsonl(path, filt=None):
    out = []
    with open(path) as f:
        for line in f:
            # doc_id is the first key; avoid json-parsing the text for speed/memory
            r = json.loads(line)
            if filt and not filt(r): continue
            out.append(parse_fr(r['doc_id']))
    return np.array(out, dtype=np.int64)

R = {}
# ---------------------------------------------------------------- 50k Claude-labelled set
sel50k = ids_from_jsonl(PPL / 'dclm-refinedweb-50k/selected_50k_for_claude.jsonl')
scored50k = ids_from_jsonl(PPL / 'dclm-refinedweb-50k/claude-haiku/scored_50k_final.jsonl')
comb_dclm = ids_from_jsonl(PPL / 'mix-quality-scorer-train/combined.jsonl', lambda r: r.get('mix') == 'dclm-rw')
train_dclm = ids_from_jsonl(PPL / 'mix-quality-scorer-train/train/train.jsonl', lambda r: r.get('mix') == 'dclm-rw')
val_dclm = ids_from_jsonl(PPL / 'mix-quality-scorer-train/val/val.jsonl', lambda r: r.get('mix') == 'dclm-rw')
removed = np.load(RW / '01_explore/match_50k_prefix4000.npy').astype(np.int64)
log('50k parsed', sel50k.size, scored50k.size, comb_dclm.size, train_dclm.size, val_dclm.size, 'removed', removed.size)
R['set_50k'] = {
    'selected_50k_for_claude_rows': int(sel50k.size), 'unique': int(np.unique(sel50k).size),
    'scored_50k_final_rows': int(scored50k.size),
    'dropped_between_selection_and_scoring': sorted(map(int, np.setdiff1d(sel50k, scored50k))),
    'ridge_combined_dclm_rows': int(comb_dclm.size), 'ridge_combined_dclm_unique': int(np.unique(comb_dclm).size),
    'ridge_combined_dclm_equals_scored50k': bool(np.array_equal(np.sort(comb_dclm), np.sort(scored50k))),
    'train_split_dclm': int(train_dclm.size), 'val_split_dclm': int(val_dclm.size),
    'train_val_disjoint': int(np.intersect1d(train_dclm, val_dclm).size) == 0,
    'train_union_val_equals_combined': bool(np.array_equal(np.union1d(train_dclm, val_dclm), np.unique(comb_dclm))),
    'min_orig': int(sel50k.min()), 'max_orig': int(sel50k.max()),
    'removed_50838_count': int(removed.size),
    'sel50k_in_removed': int(np.isin(sel50k, removed).sum()),
    'ridge_dclm_in_removed': int(np.isin(comb_dclm, removed).sum()),
    'removed_not_in_sel50k_(pool_duplicates_by_prefix)': int((~np.isin(removed, sel50k)).sum()),
}
log(R['set_50k'])

# ---------------------------------------------------------------- 5M analysis sample (re-derived)
rng = np.random.default_rng(42)
s5m = np.sort(rng.choice(N_RAW, 5_000_000, replace=False)).astype(np.int64)
np.save(OUT / 'sample5m_orig_positions_rederived.npy', s5m)
log('5M re-derived', s5m.size, s5m[:5])
ex = pq.read_table(PPL.parent / 'exclusions/exclude_all_global_indices.parquet',
                   columns=['global_index', 'exclusion_source']).to_pandas()
ex5 = np.sort(ex.loc[ex.exclusion_source == '5m_basic', 'global_index'].to_numpy(np.int64))
ex50 = np.sort(ex.loc[ex.exclusion_source == '50k_claude', 'global_index'].to_numpy(np.int64))
del ex
R['set_5m'] = {
    'rederived_n': int(s5m.size), 'rederived_first5': s5m[:5].tolist(),
    'exclusion_table_5m_basic_rows': int(ex5.size), 'exclusion_table_50k_claude_rows': int(ex50.size),
    'rederived_in_exclusion_5m_basic': int(np.isin(s5m, ex5).sum()),
    'exclusion_5m_basic_not_in_rederived': int((~np.isin(ex5, s5m)).sum()),
    'exclusion_50k_claude_subset_of_sel50k': bool(np.isin(ex50, sel50k).all()),
    'exclusion_50k_claude_not_in_sel50k': int((~np.isin(ex50, sel50k)).sum()),
    'exclusion_50k_claude_not_in_sel50k_in_removed': int(np.isin(ex50[~np.isin(ex50, sel50k)], removed).sum()),
    'sel50k_not_in_exclusion_50k_claude': int((~np.isin(sel50k, ex50)).sum()),
    'sel50k_not_in_ex50_but_in_ex5': int(np.isin(sel50k[~np.isin(sel50k, ex50)], ex5).sum()),
    'exclusion_5m_basic_extra_in_removed': int(np.isin(ex5[~np.isin(ex5, s5m)], removed).sum()),
    'sel50k_in_rederived_5m': int(np.isin(sel50k, s5m).sum()),
    'ridge_dclm_in_rederived_5m': int(np.isin(comb_dclm, s5m).sum()),
    'rederived_5m_in_removed_50838': int(np.isin(s5m, removed).sum()),
    'rederived_5m_surviving_in_pool': int(s5m.size - np.isin(s5m, removed).sum()),
    'exclusion_5m_basic_in_removed_50838': int(np.isin(ex5, removed).sum()),
    'exclusion_5m_basic_surviving': int(ex5.size - np.isin(ex5, removed).sum()),
    'expected_random_5m_in_removed': 5_000_000 * removed.size / N_RAW,
}
log(R['set_5m'])

# ---------------------------------------------------------------- map orig -> scored-pool doc_id
orig = np.load(FLAT / 'orig_doc_id.npy', mmap_mode='r')
assert orig.shape[0] == N_POOL
orig = np.asarray(orig, dtype=np.int64)
assert np.all(np.diff(orig) > 0)
tok = np.asarray(np.load(FLAT / 'tokens_llama2.npy', mmap_mode='r'), dtype=np.int64) + 1
TOT = int(tok.sum())
def to_docid(o):
    o = np.unique(o); i = np.searchsorted(orig, o); i = np.minimum(i, N_POOL - 1)
    hit = orig[i] == o; return np.sort(i[hit]), int((~hit).sum())
S = {}
S['ridge_dclm_49998'], miss_r = to_docid(comb_dclm)
S['sel50k_50000'], miss_s = to_docid(sel50k)
S['sample5m_rederived'], miss_5 = to_docid(s5m)
S['sample5m_exclusion_table'], miss_e = to_docid(ex5)
R['pool_presence'] = {
    'ridge_dclm_present_in_pool': int(S['ridge_dclm_49998'].size), 'ridge_dclm_absent': miss_r,
    'sel50k_present_in_pool': int(S['sel50k_50000'].size), 'sel50k_absent': miss_s,
    'sample5m_rederived_present_in_pool': int(S['sample5m_rederived'].size), 'sample5m_rederived_absent': miss_5,
    'sample5m_excl_table_present_in_pool': int(S['sample5m_exclusion_table'].size), 'sample5m_excl_table_absent': miss_e,
}
log(R['pool_presence'])
for k, v in S.items(): np.save(OUT / f'{k}_scored_docids.npy', v)

val = np.sort(np.random.default_rng(np.random.SeedSequence(42).spawn(8)[0]).choice(N_POOL, 50_000, replace=False))

sels = {
    'raw_top10b_fineweb_edu': SEL / 'raw_top10b_fineweb_edu/selected_doc_ids_order.npy',
    'raw_top10b_modernbert': SEL / 'raw_top10b_modernbert/selected_doc_ids_order.npy',
    'raw_top10b_consensus': SEL / 'raw_top10b_consensus/selected_doc_ids_order.npy',
    'fasttext_quality_base': SEL / 'fasttext_quality_base_doc_ids_order.npy',
    'fasttext_global_top10b': SEL / 'fasttext_global_top10b_doc_ids_order.npy',
}
T = {'validation_holdout_50k': val}
for k, p in sels.items():
    a = np.load(p).astype(np.int64); assert a.size == np.unique(a).size, k; T[k] = np.sort(a)
    if k == 'fasttext_quality_base': T['shared_5b_anchor'] = np.sort(a[:4_120_164])
# anchor cross-check vs raw_sources/anchor_doc_ids.npy (orig space)
anc = np.load(RAWSRC / 'anchor_doc_ids.npy').astype(np.int64)
R['anchor_check'] = {'raw_sources_anchor_n': int(anc.size),
                     'raw_sources_anchor_equals_QB_first_4120164_as_orig_set':
                         bool(np.array_equal(np.sort(anc), np.sort(orig[T['shared_5b_anchor']])))}
log(R['anchor_check'])
for arm in ['raw_disagreement_aware', 'raw_diversity_oriented', 'raw_random', 'raw_rewire_inspired']:
    o = np.load(RAWSRC / arm / 'selected_doc_ids.npy').astype(np.int64)
    d, miss = to_docid(o); assert miss == 0, (arm, miss); T[arm + '(4 raw strategy-linked corpora)'] = d
T['full_pool'] = np.arange(N_POOL, dtype=np.int64)

res = {}
for sk, s in S.items():
    frac = s.size / N_POOL
    res[sk] = {'n_in_pool': int(s.size), 'pool_fraction': frac}
    for tk, t in T.items():
        inter = np.intersect1d(s, t, assume_unique=True)
        t_tok = int(tok[t].sum()); i_tok = int(tok[inter].sum())
        res[sk][tk] = {
            'target_docs': int(t.size), 'target_train_tokens': t_tok,
            'overlap_docs': int(inter.size), 'overlap_doc_share': inter.size / t.size,
            'overlap_train_tokens': i_tok, 'overlap_token_share': i_tok / t_tok,
            'expected_docs_if_random': frac * t.size,
            'enrichment_docs': (inter.size / (frac * t.size)) if (t.size and frac) else None,
        }
    log(sk, 'done')
R['overlaps'] = res
R['pool_total_train_tokens'] = TOT
(OUT / 'overlaps.json').write_text(json.dumps(R, indent=1))
log('wrote overlaps.json')
