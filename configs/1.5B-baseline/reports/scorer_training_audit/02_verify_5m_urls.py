#!/usr/bin/env python3
"""READ-ONLY: verify the re-derived 5M positions against the materialised 5M sample parquets
by comparing the `url` column (only url is read; never text). Checks 5M shards 0 and 9 in full."""
import json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
OUT = Path(__file__).resolve().parent
RAW = Path('/weka/scratch/jhu/bvandur1/zhuicon1/datasets/ppl-dsai/dclm-refinedweb-100m-sample')
S5 = Path('/weka/scratch/jhu/bvandur1/zhuicon1/datasets/ppl-dsai/dclm-refinedweb-5m-samples')
pos = np.load(OUT / 'sample5m_orig_positions_rederived.npy')
res = {}
for k in (0, 9):
    u5 = pq.read_table(S5 / f'dclm_refinedweb_5m_sample_{k:05d}.parquet', columns=['url']).column('url').to_pylist()
    p = pos[k * 500_000:(k + 1) * 500_000]
    assert len(u5) == p.size
    fidx = p // 500_000; eq = 0; n = 0
    for f in np.unique(fidx):
        rows = p[fidx == f] - f * 500_000
        ur = pq.read_table(RAW / f'dclm_refinedweb_sample_{f:05d}.parquet', columns=['url']).column('url').take(rows).to_pylist()
        seg = u5[n:n + len(ur)]
        eq += sum(a == b for a, b in zip(seg, ur)); n += len(ur)
    res[f'shard_{k}'] = {'rows': len(u5), 'url_equal': eq, 'raw_files_touched': int(np.unique(fidx).size)}
    print(k, res[f'shard_{k}'], flush=True)
(OUT / 'verify_5m_urls.json').write_text(json.dumps(res, indent=1))
