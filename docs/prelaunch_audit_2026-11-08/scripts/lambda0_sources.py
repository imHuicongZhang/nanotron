#!/usr/bin/env python
"""Goal 2 provenance: are the λ0 corpus's rewritten documents drawn from the λ = 0 selection over the same feasible
domain U_τ (quality floor Q30(q), variance cap Q90(v)) that the published λ = 0.5 arm used?

Inputs (local, read only):
  06_lambda_grid/lambda_grid.npz  doc_id, q, v, ranks[λ], cutoffs[λ], tau_q, tau_v for U_τ (10,435,667 docs);
                                  produced by select_lambda_grid.py, which gates λ ∈ {0, 0.5, 1} byte-exact against the
                                  saved doc_ids.npy of the original selection runs
  <λ0 dir>/rewritten/{wiki,distill}_*.parquet   the rewritten half of the λ0 corpus (columns doc_id, u_score, …)
  <λ0 dir>/shuffled/part_*.parquet              training order (source_prompt, doc_id)
Id spaces: `doc_id` = index into the 99,949,162-document scored pool (the npz's doc_id); `orig_doc_id` = position in
the 100M DCLM reservoir (shard*500000+row). The selection is keyed by doc_id (first run of this script keyed by
orig_doc_id by mistake; see logs/lambda0_sources_origid.log).
Checks: rewritten doc_ids ⊆ λ0 selection; coverage; u_score = q; overlap with the λ = 0.5 selection;
every source has q ≥ τ_q and v ≤ τ_v; the shuffled rows are exactly anchor ∪ rewritten.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"
NPZ = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/06_lambda_grid/lambda_grid.npz")
L0 = Path("/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0")


def selection(z, i):
    # rank r (0-based, descending u_λ) is admitted while r < cutoff; check both conventions below
    return z["ranks"][i] < z["cutoffs"][i]


def main():
    z = np.load(NPZ)
    lam = list(z["lambdas"])
    i0, i5 = lam.index(0.0), lam.index(0.5)
    res = {"lambdas": lam, "cutoffs": z["cutoffs"].tolist(), "tau_q": float(z["tau_q"]), "tau_v": float(z["tau_v"]),
           "U_tau_docs": int(z["doc_id"].size)}
    ids = z["doc_id"]
    sel0_lt, sel0_le = ids[z["ranks"][i0] < z["cutoffs"][i0]], ids[z["ranks"][i0] <= z["cutoffs"][i0]]
    sel5 = ids[selection(z, i5)]
    res["sel0_docs_lt"], res["sel0_docs_le"], res["sel05_docs"] = int(sel0_lt.size), int(sel0_le.size), int(sel5.size)
    pos = {int(d): k for k, d in enumerate(ids)}

    rw_ids, u_err, rows, by_kind = [], 0.0, 0, {}
    for p in sorted((L0 / "rewritten").glob("*.parquet")):
        t = pq.read_table(p, columns=["doc_id", "u_score", "source_prompt", "status"])
        o = t["doc_id"].to_numpy()
        u = t["u_score"].to_numpy()
        kind = p.name.split("_")[0]
        by_kind[kind] = by_kind.get(kind, 0) + len(o)
        rows += len(o)
        rw_ids.append(o)
        k = np.array([pos.get(int(x), -1) for x in o[: min(len(o), 2000)]])
        ok = k >= 0
        if ok.any():
            u_err = max(u_err, float(np.max(np.abs(u[: len(k)][ok] - z["q"][k[ok]]))))
    rw = np.unique(np.concatenate(rw_ids))
    in_U = np.isin(rw, ids)
    res.update({"rewritten_rows": rows, "rows_by_file_kind": by_kind, "rewritten_unique_sources": int(rw.size),
                "sources_in_U_tau": int(in_U.sum()),
                "sources_in_sel0_lt": int(np.isin(rw, sel0_lt).sum()), "sources_in_sel0_le": int(np.isin(rw, sel0_le).sum()),
                "sel0_lt_covered": float(np.isin(sel0_lt, rw).mean()),
                "sources_in_sel05": int(np.isin(rw, sel5).sum()),
                "max_abs_u_score_minus_q_sample": u_err})
    k = np.array([pos[int(x)] for x in rw[in_U]])
    res["min_q_sources"], res["max_v_sources"] = float(z["q"][k].min()), float(z["v"][k].max())
    res["floor_cap_hold"] = bool(res["min_q_sources"] >= res["tau_q"] and res["max_v_sources"] <= res["tau_v"])

    anchor, other, srcs = 0, [], {}
    for p in sorted((L0 / "shuffled").glob("part_*.parquet")):
        t = pq.read_table(p, columns=["doc_id", "source_prompt"])
        sp = np.array(t["source_prompt"].to_pylist(), dtype=object)
        o = t["doc_id"].to_numpy()
        a = sp == "original"
        anchor += int(a.sum())
        other.append(o[~a])
        for s in set(sp):
            srcs[s] = srcs.get(s, 0) + int((sp == s).sum())
    other = np.concatenate(other)
    res.update({"shuffled_anchor_rows": anchor, "shuffled_rewritten_rows": int(other.size), "shuffled_source_prompts": srcs,
                "shuffled_rewritten_sources_eq_rewritten_dir": bool(np.array_equal(np.unique(other), rw))})
    print(json.dumps(res, indent=1))
    (EV / "lambda0_sources.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
