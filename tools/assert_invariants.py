#!/usr/bin/env python
"""Fail fast if the batch invariants have drifted.

Two invariants hold across all 72 runs of the grid, and both must be true or the runs are
not comparable to each other:

    micro_batch_size == 16                  (pinned for NUMERICAL comparability, not speed)
    mbs * accum * dp * seq_len == 2,097,152 tokens/step

The second is what most people check. The first is the one that actually gets broken,
because 109.7 GiB of 141 GB looks like headroom. It is not: `Loss.forward` computes
`masked_mean(loss, label_mask)` and `label_mask` drops the token before every document
boundary, so masked_mean's denominator varies per micro-batch and the effective per-token
weight is `1/(n_micro * n_j)`. Regrouping into a different mbs moves those weights by
1.43e-3 (mbs=4 vs 16) — four orders of magnitude above fp32 rounding. Nothing in a loss
curve will look wrong; the runs are simply no longer measuring the same objective.

Run it two ways:

  preflight (before any GPU time — this is the cheap one):
      python tools/assert_invariants.py --config rendered/quality-base_seed42_trunk1.yaml

  smoke test (after ~200 steps of a real run, catches an edited config or a stale env):
      python tools/assert_invariants.py --config rendered/x.yaml --log train.log --at-step 200

Exit 0 = invariants hold. Exit 1 = they do not; do not start the grid.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

EXPECTED_MBS = 16
EXPECTED_TOK_PER_STEP = 2_097_152

# nanotron/trainer.py logs this banner once at startup:
#   mbs: 16 | grad_accum: 8 | cp: 1 | sequence_length: 2048 | global_batch_size: 1024 | ...
BANNER = re.compile(
    r"mbs:\s*(\d+)\s*\|\s*grad_accum:\s*(\d+)\s*\|\s*cp:\s*(\d+)\s*\|\s*"
    r"sequence_length:\s*(\d+)\s*\|\s*global_batch_size:\s*(\d+)"
)
# and per iteration:  iteration: 200 / 4292 | consumed_tokens: 419M | ...
ITER = re.compile(r"iteration:\s*(\d+)\s*/\s*\d+.*?consumed_tokens:\s*([\d.]+)([KMBT]?)")
SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def fail(msgs):
    print("INVARIANT VIOLATION", file=sys.stderr)
    for m in msgs:
        print(f"  - {m}", file=sys.stderr)
    print("\n  mbs is pinned for numerical comparability across the 72 runs, not for speed.", file=sys.stderr)
    print("  See deploy/clusters.yaml and SOP.md before changing anything.", file=sys.stderr)
    sys.exit(1)


def check_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    errs = []
    try:
        mbs = cfg["tokens"]["micro_batch_size"]
        accum = cfg["tokens"]["batch_accumulation_per_replica"]
        seq = cfg["tokens"]["sequence_length"]
        dp = cfg["parallelism"]["dp"]
    except KeyError as e:
        fail([f"{path}: missing {e} — is this a RENDERED config, or a raw template? "
              f"Templates carry no parallelism/tokens deployment keys by design."])
    tok = mbs * accum * dp * seq
    print(f"config  : {path}")
    print(f"          mbs={mbs} accum={accum} dp={dp} seq={seq} -> {tok:,} tokens/step")
    if mbs != EXPECTED_MBS:
        errs.append(f"micro_batch_size is {mbs}, expected {EXPECTED_MBS}")
    if tok != EXPECTED_TOK_PER_STEP:
        errs.append(f"tokens/step is {tok:,}, expected {EXPECTED_TOK_PER_STEP:,}")
    if mbs * accum * dp != cfg["tokens"].get("_gbs", mbs * accum * dp):
        errs.append("global batch inconsistent")
    return errs, (mbs, accum, dp, seq)


def check_log(path: Path, at_step: int, expect):
    mbs, accum, dp, seq = expect
    errs = []
    banner = None
    iter_tok = None
    for line in path.read_text(errors="ignore").splitlines():
        m = BANNER.search(line)
        if m and banner is None:
            banner = tuple(int(x) for x in m.groups())
        m = ITER.search(line)
        if m and int(m.group(1)) == at_step:
            iter_tok = float(m.group(2)) * SUFFIX[m.group(3)]
    if banner is None:
        errs.append(f"{path}: no 'mbs: ... | grad_accum: ...' banner found — did the run start?")
    else:
        b_mbs, b_accum, b_cp, b_seq, b_gbs = banner
        print(f"log     : mbs={b_mbs} grad_accum={b_accum} cp={b_cp} seq={b_seq} gbs={b_gbs}")
        if b_mbs != EXPECTED_MBS:
            errs.append(f"log reports micro_batch_size {b_mbs}, expected {EXPECTED_MBS}")
        if (b_mbs, b_accum, b_seq) != (mbs, accum, seq):
            errs.append(f"log banner {(b_mbs, b_accum, b_seq)} disagrees with config {(mbs, accum, seq)}")
        if b_gbs * b_seq != EXPECTED_TOK_PER_STEP:
            errs.append(f"log global_batch_size {b_gbs} x seq {b_seq} = {b_gbs*b_seq:,} tokens/step, "
                        f"expected {EXPECTED_TOK_PER_STEP:,}")
    if iter_tok is None:
        errs.append(f"{path}: no 'iteration: {at_step} / ...' line — run has not reached step {at_step}")
    else:
        want = at_step * EXPECTED_TOK_PER_STEP
        # nanotron prints consumed_tokens with 3 significant figures, so compare loosely
        if abs(iter_tok - want) / want > 0.01:
            errs.append(f"consumed_tokens at step {at_step} is {iter_tok:,.0f}, expected ~{want:,} "
                        f"({at_step} x {EXPECTED_TOK_PER_STEP:,})")
        else:
            print(f"          consumed_tokens@{at_step} = {iter_tok:,.0f} (expected ~{want:,}) OK")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="a RENDERED config (not a template)")
    ap.add_argument("--log", type=Path, help="training log, for the post-step-200 check")
    ap.add_argument("--at-step", type=int, default=200)
    args = ap.parse_args()

    errs, expect = check_config(args.config)
    if args.log:
        errs += check_log(args.log, args.at_step, expect)
    if errs:
        fail(errs)
    print(f"OK: micro_batch_size={EXPECTED_MBS}, {EXPECTED_TOK_PER_STEP:,} tokens/step")


if __name__ == "__main__":
    main()
