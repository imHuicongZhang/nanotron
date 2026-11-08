"""READ-ONLY replay of build_raw_sources.py's seed-42 budget rule for the two raw controls."""
import numpy as np, pyarrow.parquet as pq, glob, json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from huggingface_hub import HfFileSystem
REV = '9e5ff24149c2957c30f0c8fdd051a8eb3b75baad'; BUDGET = 5_000_000_000
POOL = sorted(glob.glob('/projects/bvandur1/zhuicon1/data/rewrite-1p5b-100m-scored/merged_clean_*.parquet'))
def pool_read(f):
    t = pq.read_table(f, columns=['orig_doc_id', 'tokens-llama2'])
    return t.column(0).to_numpy().astype(np.int64), t.column(1).to_numpy().astype(np.int64)
with ProcessPoolExecutor(8) as ex: parts = list(ex.map(pool_read, POOL))
P_o = np.concatenate([p[0] for p in parts]); P_t = np.concatenate([p[1] for p in parts]) + 1; del parts
anchor = np.load('/projects/bvandur1/zhuicon1/kys/raw_sources/anchor_doc_ids.npy').astype(np.int64)
fs = HfFileSystem()
def arm_file(p):
    with fs.open(p, 'rb', block_size=4 << 20) as fh:
        t = pq.ParquetFile(fh).read(columns=['orig_doc_id', 'source_prompt'])
    o = t.column(0).to_numpy().astype(np.int64); sp = np.asarray(t.column(1).to_pylist(), dtype=object)
    return o[sp != 'original']
RAW = '/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/raw_text'
res = {}
for raw, arm in [('raw_diversity_oriented', 'diversity_oriented'), ('raw_disagreement_aware', 'disagreement_aware')]:
    files = sorted(p for p in fs.ls(f'datasets/wytro/Know-Your-Sources@{REV}/{arm}', detail=False) if p.endswith('.parquet'))
    with ThreadPoolExecutor(8) as ex: R = np.unique(np.concatenate(list(ex.map(arm_file, files))))
    tok = P_t[np.searchsorted(P_o, R)]; assert np.array_equal(P_o[np.searchsorted(P_o, R)], R)
    perm = np.random.default_rng(42).permutation(R.size); cum = np.cumsum(tok[perm])
    k = int(np.searchsorted(cum, BUDGET)) + 1; S = np.sort(R[perm[:k]])
    pub_s, pub_a = [], []
    for f in sorted(glob.glob(f'{RAW}/{raw}/part-*.parquet')):
        t = pq.read_table(f, columns=['orig_doc_id', 'source'])
        o = t.column(0).to_numpy().astype(np.int64); src = np.asarray(t.column(1).to_pylist(), dtype=object)
        pub_s.append(o[src == 'strategy']); pub_a.append(o[src == 'anchor'])
    pub_s = np.sort(np.concatenate(pub_s)); pub_a = np.sort(np.concatenate(pub_a))
    a_tok = int(P_t[np.searchsorted(P_o, pub_a)].sum()); s_tok = int(P_t[np.searchsorted(P_o, pub_s)].sum())
    res[raw] = {'R_unique_sources': int(R.size), 'R_tokens': int(tok.sum()), 'R_share_kept': round(k / R.size, 4),
                'replayed_S_docs': k, 'replayed_S_tokens': int(cum[k - 1]), 'published_strategy_docs': int(pub_s.size),
                'published_strategy_equals_replay': bool(np.array_equal(S, pub_s)),
                'published_strategy_subset_of_R': bool(np.isin(pub_s, R).all()),
                'published_anchor_equals_shared_anchor': bool(np.array_equal(pub_a, np.sort(anchor))),
                'anchor_tokens_pool_plus1': a_tok, 'strategy_tokens_pool_plus1': s_tok, 'total_tokens_pool_plus1': a_tok + s_tok,
                'R_intersect_anchor': int(np.intersect1d(R, anchor).size)}
    print(raw, res[raw], flush=True)
json.dump(res, open('subsample_replay.json', 'w'), indent=1)
