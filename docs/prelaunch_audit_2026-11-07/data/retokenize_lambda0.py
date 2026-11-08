"""READ-ONLY: re-tokenize sampled lambda0 rows and compare with the shipped .ds (text, no special tokens, + </s>=2)."""
import numpy as np, pyarrow.parquet as pq, json
from tokenizers import Tokenizer
tk = Tokenizer.from_file('/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/tokenizer/tokenizer.json')
L = '/weka/scratch/jhu/bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0'
tot = ok = 0; bad = None; empty_skipped = 0
for sh in (0, 13, 25):
    tbl = pq.read_table(f'{L}/shuffled/part_{sh:05d}.parquet', columns=['text'])
    texts = tbl.column('text').to_pylist()
    idx = np.fromfile(f'{L}/tokenized/{sh:05d}_unshuffled.ds.index', dtype=np.uint64).astype(np.int64)
    tok = np.memmap(f'{L}/tokenized/{sh:05d}_unshuffled.ds', dtype=np.uint16, mode='r')
    nonempty = [j for j, t in enumerate(texts) if t]          # datatrove writes no document for empty text
    empty_skipped += len(texts) - len(nonempty)
    assert idx.size == len(nonempty), (sh, idx.size, len(nonempty))
    rows = sorted(set([0, 1, len(nonempty) - 1] + list(np.random.default_rng(sh).choice(len(nonempty), 1997, replace=False))))
    for k in rows:
        j = nonempty[k]; a = 0 if k == 0 else idx[k - 1]; b = idx[k]
        want = tk.encode(texts[j], add_special_tokens=False).ids + [2]; tot += 1
        if list(tok[a:b]) == want: ok += 1
        elif bad is None: bad = (sh, j)
r = {'rows_checked': tot, 'identical': ok, 'first_mismatch': bad, 'empty_texts_in_checked_shards': empty_skipped}
print(r); json.dump(r, open(__file__.replace('.py', '.json'), 'w'), indent=1)
