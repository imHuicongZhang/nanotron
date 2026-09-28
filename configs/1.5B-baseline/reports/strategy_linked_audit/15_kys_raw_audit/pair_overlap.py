#!/usr/bin/env python
"""Raw strategy half (S) vs the published rewritten half: overlap and one-to-many counts.

    python 15_kys_raw_audit/pair_overlap.py 15_kys_raw_audit/pair_overlap.json
"""
import glob, json, sys, hashlib
import numpy as np, pyarrow.parquet as pq
K='/projects/bvandur1/zhuicon1/kys'
ARMS={'raw_diversity_oriented':'diversity_oriented','raw_disagreement_aware':'disagreement_aware','raw_random':'wrap_inspired','raw_rewire_inspired':'rewire_inspired'}
out={}
anchor=np.load(f'{K}/raw_sources/anchor_doc_ids.npy')
for raw,arm in ARMS.items():
    oid=[];pr=[];tt=[]
    for f in sorted(glob.glob(f'{K}/hf_parquet/{arm}/*.parquet')):
        t=pq.read_table(f,columns=['orig_doc_id','source_prompt','train_tokens'])
        oid.append(t['orig_doc_id'].to_numpy()); pr.append(np.asarray(t['source_prompt'].to_pylist(),dtype=object)); tt.append(t['train_tokens'].to_numpy().astype(np.int64))
    oid=np.concatenate(oid);pr=np.concatenate(pr);tt=np.concatenate(tt)
    rw=pr!='original'
    roid,rtt,rpr=oid[rw],tt[rw]+1,pr[rw]
    R,cnt=np.unique(roid,return_counts=True)
    src_ids=np.load(f'{K}/raw_sources/{raw}/source_doc_ids.npy'); src_tok=np.load(f'{K}/raw_sources/{raw}/source_tokens.npy')
    assert np.array_equal(R,src_ids)
    Sel=np.sort(np.load(f'{K}/raw_sources/{raw}/selected_doc_ids.npy'))
    inS=np.isin(roid,Sel)
    selmask=np.isin(R,Sel)
    d=dict(counterpart=arm, rewrite_rows=int(rw.sum()), rewritten_train_tokens=int(rtt.sum()),
      prompts={p:int((rpr==p).sum()) for p in np.unique(rpr)},
      unique_sources=int(R.size), source_tokens_R=int(src_tok.sum()),
      multiplicity={int(k):int((cnt==k).sum()) for k in np.unique(cnt)},
      raw_docs=int(Sel.size), raw_tokens=int(src_tok[selmask].sum()),
      overlap_docs=int(selmask.sum()),
      cov_raw_in_rewrite_sources=float(selmask.sum()/Sel.size),
      cov_rewrite_sources_in_raw=float(selmask.sum()/R.size),
      jaccard=float(selmask.sum()/(R.size+Sel.size-selmask.sum())),
      tokw_source_cov=float(src_tok[selmask].sum()/src_tok.sum()),
      rewritten_rows_with_source_in_raw=int(inS.sum()),
      rewritten_tokens_with_source_in_raw=int(rtt[inS].sum()),
      rewritten_tokcov=float(rtt[inS].sum()/rtt.sum()),
      raw_overlap_anchor=int(np.intersect1d(Sel,anchor).size),
      sel_digest_sorted_int64le=hashlib.sha256(Sel.astype('<i8').tobytes()).hexdigest())
    out[raw]=d; print(raw,json.dumps(d),flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
