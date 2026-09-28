#!/usr/bin/env python3
"""READ-ONLY: check which rows the production Ridge (modernbert_norm/model.pkl) was fit on, by
refitting Ridge(alpha=1) on L2-normalised embeddings of (a) all 50,427 combined rows and
(b) the first 40,340 rows (combined.jsonl = train.jsonl then val.jsonl, per prepare_data.py),
and comparing coefficients."""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
D = Path('/weka/scratch/jhu/bvandur1/zhuicon1/datasets/ppl-dsai/mix-quality-scorer-train')
OUT = Path(__file__).resolve().parent
m = pickle.load(open(D / 'models/modernbert_norm/model.pkl', 'rb'))
X = np.load(D / 'embeddings/modernbert/embeddings.npy').astype(np.float64)
y = np.load(D / 'embeddings/modernbert/scores.npy').astype(np.float64)
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
res = {'pickle_n_features_in': int(getattr(m, 'n_features_in_', -1)), 'X_shape': list(X.shape)}
for name, sl in [('all_50427', slice(None)), ('first_40340', slice(0, 40340))]:
    r = Ridge(alpha=1.0).fit(X[sl], y[sl])
    res[name] = {'max_abs_coef_diff': float(np.abs(r.coef_ - m.coef_).max()),
                 'intercept_diff': float(abs(r.intercept_ - m.intercept_))}
# also check that combined.jsonl order == train then val
ids = [hash(json.loads(l)['text']) for l in open(D / 'combined.jsonl')]
tr = [hash(json.loads(l)['text']) for l in open(D / 'train/train.jsonl')]
va = [hash(json.loads(l)['text']) for l in open(D / 'val/val.jsonl')]
res['combined_equals_train_then_val'] = ids == tr + va
print(res)
(OUT / 'verify_ridge_fit.json').write_text(json.dumps(res, indent=1))
