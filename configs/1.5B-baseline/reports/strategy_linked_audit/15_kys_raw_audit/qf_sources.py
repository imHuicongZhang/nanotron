"""Unique source orig_doc_id of the Quality-First rewritten half (wytro/Know-Your-Sources@9e5ff241/quality_first),
reading only orig_doc_id/source_prompt/train_tokens columns remotely."""
import sys, numpy as np, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, HfApi
REV = '9e5ff24149c2957c30f0c8fdd051a8eb3b75baad'
fs = HfFileSystem()
files = sorted(f for f in HfApi().list_repo_files('wytro/Know-Your-Sources', repo_type='dataset', revision=REV)
               if f.startswith('quality_first/') and f.endswith('.parquet'))
src, anc, rtok, prm = [], [], [], []
for f in files:
    with fs.open(f'datasets/wytro/Know-Your-Sources@{REV}/{f}', 'rb') as fh:
        t = pq.read_table(fh, columns=['orig_doc_id', 'source_prompt', 'train_tokens'])
    sp = np.asarray(t['source_prompt'].to_pylist(), dtype=object); o = t['orig_doc_id'].to_numpy().astype(np.int64)
    m = sp == 'original'; anc.append(o[m]); src.append(o[~m]); rtok.append(t['train_tokens'].to_numpy()[~m].astype(np.int64) + 1); prm.append(sp[~m])
    print(f, t.num_rows, flush=True)
np.savez(sys.argv[1], src=np.concatenate(src), anchor=np.concatenate(anc), rtok=np.concatenate(rtok), prompts=np.concatenate(prm).astype(str))
