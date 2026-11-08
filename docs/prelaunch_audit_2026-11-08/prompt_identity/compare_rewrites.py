#!/usr/bin/env python
"""Issue C: did the λ0 and λ0.5 rewriting runs use the same prompt text? Direct output comparison, read only.

Under greedy decoding with the same model, vLLM version and prompt, a document rewritten in both runs should give
the same text, except for rare batch-dependent numerical divergence (which typically shows as identical text up to
some point, then a different continuation). A different prompt would change outputs from the first tokens on.

Inputs
  published λ0.5 corpus  wytro/Know-Your-Sources@9e5ff24 disagreement_aware/part_00000.parquet (downloaded to
                         ../work/kys_da/; 500,000 rows of the shuffled 12.56M-row mixture)
  local λ0 corpus        <λ0>/shuffled/part_*.parquet (12,566,226 rows, verified in §1/§2 of the audit)
Rows are matched on (doc_id, source_prompt) for source_prompt in {wikipedia, distill}; doc_id is the scored-pool id
in both corpora. Every matched pair is compared (no subsampling beyond part_00000).
Output: ../evidence/prompt_identity.json and a mismatch sample in ../evidence/prompt_identity_mismatches.jsonl
"""
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"
DA = HERE.parent / "work" / "kys_da" / "disagreement_aware" / "part_00000.parquet"
L0 = Path("/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0/shuffled")


def first_diff(a, b):
    n = min(len(a), len(b))
    i = 0
    step = 4096
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    t = pq.read_table(DA, columns=["doc_id", "source_prompt", "text"])
    t = t.filter(pc.is_in(t["source_prompt"], value_set=__import__("pyarrow").array(["wikipedia", "distill"])))
    da = {(int(d), s): x for d, s, x in zip(t["doc_id"].to_numpy(), t["source_prompt"].to_pylist(), t["text"].to_pylist())}
    da_counts = Counter(s for _, s in da)
    print("DA part_00000 rewritten rows:", dict(da_counts), flush=True)
    keys = set(da)
    pairs = []
    for p in sorted(L0.glob("part_*.parquet")):
        lt = pq.read_table(p, columns=["doc_id", "source_prompt"])
        ids, sps = lt["doc_id"].to_numpy(), lt["source_prompt"].to_pylist()
        idx = [i for i, (d, s) in enumerate(zip(ids, sps)) if (int(d), s) in keys]
        if not idx:
            continue
        txt = pq.read_table(p, columns=["text"])["text"].take(idx).to_pylist()
        for i, x in zip(idx, txt):
            pairs.append((int(ids[i]), sps[i], da[(int(ids[i]), sps[i])], x))
    print("matched pairs:", len(pairs), flush=True)

    res, mism = {}, []
    for sp in ("wikipedia", "distill"):
        P = [q for q in pairs if q[1] == sp]
        exact = sum(a == b for _, _, a, b in P)
        ws = sum(" ".join(a.split()) == " ".join(b.split()) for _, _, a, b in P)
        fracs, sims, early = [], [], 0
        for d, s, a, b in P:
            if a == b:
                continue
            k = first_diff(a, b)
            frac = k / max(1, min(len(a), len(b)))
            fracs.append(frac)
            sm = SequenceMatcher(None, a[:3000], b[:3000], autojunk=False).ratio() if len(sims) < 2000 else None
            if sm is not None:
                sims.append(sm)
            early += k < 50
            if len(mism) < 200:
                mism.append({"doc_id": d, "source_prompt": s, "first_diff_char": k, "len_l05": len(a), "len_l0": len(b),
                             "frac_before_diff": round(frac, 4), "context_l05": a[max(0, k - 60):k + 60],
                             "context_l0": b[max(0, k - 60):k + 60]})
        fracs.sort()
        q = lambda v, x: v[int(x * (len(v) - 1))] if v else None  # noqa: E731
        res[sp] = {"pairs": len(P), "exact": exact, "exact_rate": exact / len(P) if P else None,
                   "whitespace_normalised_equal": ws, "mismatches": len(P) - exact,
                   "mismatch_first_diff_fraction_quantiles": {p: q(fracs, p) for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
                   "mismatches_diverging_in_first_50_chars": early,
                   "mismatch_similarity_first3000_quantiles": {p: q(sorted(sims), p) for p in (0.0, 0.1, 0.5, 0.9)}}
        print(sp, json.dumps(res[sp], indent=1), flush=True)
    res["da_part"] = str(DA)
    res["da_rows_by_prompt"] = dict(da_counts)
    (EV / "prompt_identity.json").write_text(json.dumps(res, indent=1))
    with open(EV / "prompt_identity_mismatches.jsonl", "w") as f:
        for m in mism:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
