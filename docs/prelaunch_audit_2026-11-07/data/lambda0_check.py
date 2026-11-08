"""READ-ONLY: lambda0 vs lambda0.5 selection domain, consensus definition, and the 2-token shortfall."""
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, glob, json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from huggingface_hub import HfFileSystem
z = np.load('/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/06_lambda_grid/lambda_grid.npz')
lam, did, q, v, ranks, cut = z['lambdas'], z['doc_id'], z['q'], z['v'], z['ranks'], z['cutoffs']
POOL = sorted(glob.glob('/projects/bvandur1/zhuicon1/data/rewrite-1p5b-100m-scored/merged_clean_*.parquet'))
def rd(f):
    t = pq.read_table(f, columns=['orig_doc_id', 'fasttext-ranking-v2', 'fineweb-edu-ranking-v2', 'modernbert-ranking-v2'])
    return [t.column(i).to_numpy() for i in range(4)]
with ProcessPoolExecutor(8) as ex: parts = list(ex.map(rd, POOL))
P_o = np.concatenate([p[0] for p in parts]).astype(np.int64)
ft, fw, mb = (np.concatenate([p[i] for p in parts]) for i in (1, 2, 3)); del parts
qq = ((ft[did] + fw[did] + mb[did]) / 3.0).astype(np.float32)
vv = (((ft[did] - qq) ** 2 + (fw[did] - qq) ** 2 + (mb[did] - qq) ** 2) / 3.0).astype(np.float32)
R = {'survivors': int(did.size), 'lambdas': lam.tolist(), 'cutoffs': cut.tolist(),
     'q_recomputed_equal': bool(np.array_equal(qq, q)), 'v_recomputed_max_abs_diff': float(np.abs(vv - v).max()),
     'tau_q_Q30_U': 0.6956847310066223, 'tau_v_Q90_U': 0.1036492049694062,
     'all_survivors_q_ge_tau_q': bool((q >= np.float32(0.6956847310066223)).all() or (q >= 0.6956847310066223).all()),
     'all_survivors_v_le_tau_v': bool((v <= 0.1036492049694062).all())}
sel = {}
for i, l in enumerate(lam):
    m = ranks[i] < cut[i]; sel[float(l)] = np.sort(P_o[did[m]])
    R[f'lambda{l}_selected_docs'] = int(m.sum())
# lambda0 ordering = q descending (ties broken by a fixed rng): selected min q >= unselected max q
m0 = ranks[0] < cut[0]
R['lambda0_selected_min_q'] = float(q[m0].min()); R['lambda0_unselected_max_q'] = float(q[~m0].max())
# lambda0 corpus rewritten-row sources vs lambda0 selection
L = '/weka/scratch/jhu/bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0'
src0, zero_rows = [], []
for f in sorted(glob.glob(f'{L}/shuffled/*.parquet')):
    t = pq.read_table(f, columns=['doc_id', 'orig_doc_id', 'source_prompt', 'train_tokens'])
    sp = np.asarray(t.column('source_prompt').to_pylist(), dtype=object); o = t.column('orig_doc_id').to_numpy()
    src0.append(o[sp != 'original'])
    tt = t.column('train_tokens').to_numpy(); zi = np.flatnonzero(tt == 0)
    if zi.size:
        txt = pq.read_table(f, columns=['text']).column('text')
        for j in zi:
            zero_rows.append({'file': f.rsplit('/', 1)[1], 'row': int(j), 'doc_id': int(t.column('doc_id')[j].as_py()),
                              'source_prompt': sp[j], 'text_len': len(txt[j].as_py() or '')})
src0 = np.unique(np.concatenate(src0))
R['lambda0_corpus_rewrite_sources'] = int(src0.size)
R['lambda0_sources_subset_of_lambda0_selection'] = bool(np.isin(src0, sel[0.0]).all())
R['lambda0_sources_in_lambda05_selection'] = int(np.isin(src0, sel[0.5]).sum())
R['train_tokens_zero_rows'] = zero_rows
# published lambda0.5 arm (wytro/Know-Your-Sources disagreement_aware): strategy sources vs lambda0.5 selection
fs = HfFileSystem(); REV = '9e5ff24149c2957c30f0c8fdd051a8eb3b75baad'
def arm_file(p):
    with fs.open(p, 'rb', block_size=4 << 20) as fh:
        t = pq.ParquetFile(fh).read(columns=['orig_doc_id', 'source_prompt'])
    sp = np.asarray(t.column(1).to_pylist(), dtype=object)
    return t.column(0).to_numpy().astype(np.int64)[sp != 'original'], np.unique(sp.astype(str), return_counts=True)
files = sorted(p for p in fs.ls(f'datasets/wytro/Know-Your-Sources@{REV}/disagreement_aware', detail=False) if p.endswith('.parquet'))
with ThreadPoolExecutor(8) as ex: res = list(ex.map(arm_file, files))
s05 = np.unique(np.concatenate([r[0] for r in res]))
pc_ = {}
for r in res:
    for k, c in zip(*r[1]): pc_[k] = pc_.get(k, 0) + int(c)
R['lambda05_published_sources'] = int(s05.size); R['lambda05_published_source_prompt_counts'] = pc_
R['lambda05_sources_subset_of_lambda05_selection'] = bool(np.isin(s05, sel[0.5]).all())
R['lambda0_vs_lambda05_selection_overlap'] = int(np.intersect1d(sel[0.0], sel[0.5]).size)
print(json.dumps(R, indent=1))
json.dump(R, open('lambda0_check.json', 'w'), indent=1)
