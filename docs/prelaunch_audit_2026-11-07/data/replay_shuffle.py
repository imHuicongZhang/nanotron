"""READ-ONLY replay of pp_io.bucketed_shuffle (seed 42) on document IDs only."""
import numpy as np, pyarrow.parquet as pq, glob, json, sys

def replay(spec_ids, B, seed=42):
    rng = np.random.default_rng(seed)
    bks = [rng.integers(0, B, size=a.size) for a in spec_ids]
    allid = np.concatenate(spec_ids); bk = np.concatenate(bks)
    order = np.argsort(bk, kind='stable')            # bucket files = rows in spec order, grouped by bucket
    ids_b, bk_s = allid[order], bk[order]
    out = []; starts = np.searchsorted(bk_s, np.arange(B + 1))
    for b in range(B):
        seg = ids_b[starts[b]:starts[b + 1]]
        if seg.size:
            out.append(seg[np.random.default_rng([seed, b]).permutation(seg.size)])
    return np.concatenate(out)

def find_B(spec_ids, published, Bs):
    for B in Bs:
        r = replay(spec_ids, B)
        if r.size == published.size and r[0] == published[0] and np.array_equal(r[:1000], published[:1000]):
            return B, bool(np.array_equal(r, published))
    return None, False

res = {}
R = '/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/raw_text'
for s in ['raw_diversity_oriented', 'raw_disagreement_aware']:
    pub = np.concatenate([pq.read_table(f, columns=['orig_doc_id']).column(0).to_numpy() for f in sorted(glob.glob(f'{R}/{s}/part-*.parquet'))]).astype(np.int64)
    ids = np.sort(pub)
    sh = ids // 500_000
    spec_ids = [ids[sh == k] for k in np.unique(sh)]
    B, full = find_B(spec_ids, pub, range(1, 4097))
    res[s] = {'published_rows': int(pub.size), 'pool_shard_inputs': len(spec_ids), 'B_found': B, 'full_order_identical': full,
              'is_sorted_published': bool(np.all(np.diff(pub) > 0))}
    print(s, res[s], flush=True)

L = '/weka/scratch/jhu/bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0'
pub = np.concatenate([pq.read_table(f, columns=['doc_id']).column(0).to_numpy() for f in sorted(glob.glob(f'{L}/shuffled/*.parquet'))]).astype(np.int64)
shared = [pq.read_table(f, columns=['doc_id']).column(0).to_numpy().astype(np.int64) for f in sorted(glob.glob(f'{L}/shared-top-5B/*.parquet'))]
rew = [pq.read_table(f, columns=['doc_id']).column(0).to_numpy().astype(np.int64) for f in sorted(glob.glob(f'{L}/rewritten/*.parquet'))]
print('lambda0 doc_id unique in shuffled:', np.unique(pub).size == pub.size, flush=True)
for name, specs in [('shared_then_rewritten', shared + rew), ('rewritten_then_shared', rew + shared)]:
    B, full = find_B(specs, pub, range(1, 257))
    res[f'lambda0:{name}'] = {'published_rows': int(pub.size), 'inputs': len(specs), 'B_found': B, 'full_order_identical': full}
    print(name, res[f'lambda0:{name}'], flush=True)
    if full: break
json.dump(res, open('shuffle_replay.json', 'w'), indent=1)
