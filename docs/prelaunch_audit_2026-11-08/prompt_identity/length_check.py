"""Issue C follow-up: does output length explain the shard-to-shard spread of the λ0 vs λ0.5 exact-match rate, and
the small gap between it and the same-prompt control (DO vs λ0.5)? Exact-match rate by λ0.5 output-length bin for both;
per-shard rate vs per-shard mean length. Inputs: ../evidence/prompt_identity_pairs.parquet and the two published parts."""
import json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
EV = Path(__file__).resolve().parent.parent / "evidence"; K = EV.parent / "work" / "kys_da"
P = pq.read_table(EV / "prompt_identity_pairs.parquet").to_pandas()
def load(n):
    t = pq.read_table(K / n / "part_00000.parquet", columns=["doc_id", "source_prompt", "text"])
    t = t.filter(pc.is_in(t["source_prompt"], value_set=pa.array(["wikipedia", "distill"])))
    return {(int(d), s): x for d, s, x in zip(t["doc_id"].to_numpy(), t["source_prompt"].to_pylist(), t["text"].to_pylist())}
da, do = load("disagreement_aware"), load("diversity_oriented")
S = [(s, da[k] == do[k], len(da[k])) for k in set(da) & set(do) for s in [k[1]]]
bins = [0, 1000, 2000, 3000, 4000, 6000, 10**9]
res = {}
for sp in ("wikipedia", "distill"):
    p = P[P.source_prompt == sp]
    s = np.array([(e, L) for t, e, L in S if t == sp], dtype=float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        pm = (p.len_lambda05 >= lo) & (p.len_lambda05 < hi)
        sm = (s[:, 1] >= lo) & (s[:, 1] < hi)
        rows.append({"len_chars": f"[{lo},{hi})", "P_pairs": int(pm.sum()), "P_exact": float(p.exact[pm].mean()) if pm.any() else None,
                     "S_pairs": int(sm.sum()), "S_exact": float(s[sm, 0].mean()) if sm.any() else None})
    g = p[p.lambda0_shard >= 0].groupby("lambda0_shard").agg(rate=("exact", "mean"), mlen=("len_lambda05", "mean"))
    res[sp] = {"by_length": rows, "shard_rate_vs_mean_length_pearson": float(np.corrcoef(g.rate, g.mlen)[0, 1]),
               "P_mean_len": float(p.len_lambda05.mean()), "S_mean_len": float(s[:, 1].mean())}
print(json.dumps(res, indent=1)); (EV / "prompt_identity_length.json").write_text(json.dumps(res, indent=1))
