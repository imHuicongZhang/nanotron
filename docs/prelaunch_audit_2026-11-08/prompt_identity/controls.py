#!/usr/bin/env python
"""Issue C, controls for compare_rewrites.py. Read only.

  P  λ0 (job 1734978, 2026-07) vs λ0.5 (job 1519364, 2026-06): the question. Per pair: exact?, first differing
     character, lengths, and the λ0 rewrite input shard (from <λ0>/rewritten/{wiki,distill}_<shard>.parquet), so the
     exact-match rate can be checked for uniformity across the 200 shards / 8 workers (a prompt change during either
     run would show as a step).
  S  same prompt path, different run: published diversity_oriented (job 1545473, 2026-06) vs λ0.5 (job 1519364), both
     launched in grounded mode with the same GROUNDED_PROMPT path (07_rewrite/launch_dataset.sh). The expected
     behaviour of a re-run with the same prompt.
  D  different prompt, same document: λ0's wikipedia vs distill rewrite of the same doc_id (two different prompts,
     same run). What a prompt difference looks like.
Inputs: ../work/kys_da/{disagreement_aware,diversity_oriented}/part_00000.parquet (wytro/Know-Your-Sources@9e5ff24),
<λ0>/shuffled and <λ0>/rewritten. Output: ../evidence/prompt_identity_controls.json, ../evidence/prompt_identity_pairs.parquet
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"
K = HERE.parent / "work" / "kys_da"
L0 = Path("/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0")
SP = ("wikipedia", "distill")


def first_diff(a, b):
    n, i, step = min(len(a), len(b)), 0, 4096
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_pub(name):
    t = pq.read_table(K / name / "part_00000.parquet", columns=["doc_id", "source_prompt", "text"])
    t = t.filter(pc.is_in(t["source_prompt"], value_set=pa.array(list(SP))))
    return {(int(d), s): x for d, s, x in zip(t["doc_id"].to_numpy(), t["source_prompt"].to_pylist(), t["text"].to_pylist())}


def summarize(recs):
    out = {}
    for sp in SP:
        r = [x for x in recs if x[1] == sp]
        if not r:
            continue
        ex = np.array([x[2] for x in r])
        fd = np.array([x[3] / max(1, min(x[4], x[5])) for x in r if not x[2]])
        early = sum(1 for x in r if not x[2] and x[3] < 50)
        out[sp] = {"pairs": len(r), "exact_rate": float(ex.mean()),
                   "mismatch_first_diff_fraction_median": float(np.median(fd)) if fd.size else None,
                   "mismatch_first_diff_fraction_p10": float(np.quantile(fd, 0.1)) if fd.size else None,
                   "diverge_in_first_50_chars_rate": early / len(r),
                   "diverge_at_char_0_rate": sum(1 for x in r if not x[2] and x[3] == 0) / len(r)}
    return out


def compare(a, b):
    keys = sorted(set(a) & set(b))
    return [(d, s, a[(d, s)] == b[(d, s)], first_diff(a[(d, s)], b[(d, s)]), len(a[(d, s)]), len(b[(d, s)])) for d, s in keys]


def main():
    da, do = load_pub("disagreement_aware"), load_pub("diversity_oriented")
    print("loaded DA", len(da), "DO", len(do), flush=True)

    # λ0 texts for keys present in DA, plus every λ0 wiki/distill text for the D control (same doc, both prompts)
    l0 = {}
    for p in sorted((L0 / "shuffled").glob("part_*.parquet")):
        t = pq.read_table(p, columns=["doc_id", "source_prompt"])
        ids, sps = t["doc_id"].to_numpy(), t["source_prompt"].to_pylist()
        idx = [i for i, (d, s) in enumerate(zip(ids, sps)) if (int(d), s) in da]
        if idx:
            txt = pq.read_table(p, columns=["text"])["text"].take(idx).to_pylist()
            for i, x in zip(idx, txt):
                l0[(int(ids[i]), sps[i])] = x
    print("λ0 rows matched to DA:", len(l0), flush=True)

    # λ0 rewrite shard of each doc
    shard = {}
    for p in sorted((L0 / "rewritten").glob("*.parquet")):
        kind, num = p.stem.split("_")
        sp = "wikipedia" if kind == "wiki" else "distill"
        for d in pq.read_table(p, columns=["doc_id"])["doc_id"].to_numpy():
            shard[(int(d), sp)] = int(num)

    P = compare(da, l0)
    S = compare(da, do)
    both = defaultdict(dict)
    for (d, s), x in l0.items():
        both[d][s] = x
    D = [(d, "wikipedia", v["wikipedia"] == v["distill"], first_diff(v["wikipedia"], v["distill"]),
          len(v["wikipedia"]), len(v["distill"])) for d, v in both.items() if len(v) == 2]

    res = {"P_lambda0_vs_lambda05": summarize(P), "S_same_prompt_DO_vs_lambda05": summarize(S),
           "D_wikipedia_vs_distill_same_doc_lambda0": summarize(D)["wikipedia"] if D else None}
    # uniformity across λ0 shards / workers
    for sp in SP:
        per = defaultdict(list)
        for d, s, ex, *_ in P:
            if s == sp and (d, s) in shard:
                per[shard[(d, s)]].append(ex)
        rates = np.array([np.mean(v) for _, v in sorted(per.items())])
        n = np.array([len(v) for _, v in sorted(per.items())])
        p0 = float(np.sum([sum(v) for v in per.values()]) / n.sum())
        z = (rates - p0) / np.sqrt(p0 * (1 - p0) / n)
        workers = defaultdict(list)
        for k, v in per.items():
            workers[k % 8] += v
        res[f"P_{sp}_by_lambda0_shard"] = {"shards": int(len(rates)), "pairs_per_shard_median": float(np.median(n)),
                                           "rate_min": float(rates.min()), "rate_max": float(rates.max()),
                                           "binomial_z_abs_max": float(np.abs(z).max()),
                                           "shards_with_abs_z_gt_4": int((np.abs(z) > 4).sum()),
                                           "rate_by_worker": {w: float(np.mean(v)) for w, v in sorted(workers.items())}}
    # early-divergence examples (λ0 vs λ0.5)
    ex = [(d, s, k) for d, s, e, k, *_ in P if not e and k < 50][:12]
    res["P_early_divergence_examples"] = [{"doc_id": d, "source_prompt": s, "first_diff_char": k,
                                           "lambda05_head": da[(d, s)][:160], "lambda0_head": l0[(d, s)][:160]} for d, s, k in ex]
    pq.write_table(pa.table({"doc_id": [x[0] for x in P], "source_prompt": [x[1] for x in P], "exact": [x[2] for x in P],
                             "first_diff_char": [x[3] for x in P], "len_lambda05": [x[4] for x in P],
                             "len_lambda0": [x[5] for x in P],
                             "lambda0_shard": [shard.get((x[0], x[1]), -1) for x in P]}),
                   EV / "prompt_identity_pairs.parquet")
    print(json.dumps({k: v for k, v in res.items() if k != "P_early_divergence_examples"}, indent=1))
    (EV / "prompt_identity_controls.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
