import sys, numpy as np, glob
from pathlib import Path
for d in sys.argv[1:]:
    ds = sorted(glob.glob(f'{d}/*.ds'))[0]
    tok = np.memmap(ds, dtype=np.uint16, mode='r')
    idx = np.fromfile(ds + '.index', dtype=np.uint64)
    meta = Path(ds + '.metadata').read_text().splitlines()
    n = min(len(idx), 200000)                       # first 200k documents
    ends = idx[:n].astype(np.int64)                 # cumulative end offsets (tokens)
    starts = np.concatenate([[0], ends[:-1]])
    last = tok[ends - 1]; first = tok[starts]
    seg = np.asarray(tok[:ends[-1]])
    print(Path(d).parent.name, Path(ds).name, 'meta', meta[:2], 'docs_in_index', len(idx), 'tokens', tok.size, 'index_last', int(idx[-1]))
    print('   doc last token == 2 (</s>):', np.mean(last == 2), ' doc first token == 1 (<s>):', np.mean(first == 1),
          ' count(1) in first docs:', int((seg == 1).sum()), ' count(2):', int((seg == 2).sum()), 'docs:', n)
