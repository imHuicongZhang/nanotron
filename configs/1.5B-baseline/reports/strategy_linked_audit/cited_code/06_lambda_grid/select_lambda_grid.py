#!/usr/bin/env python
"""Extend the S5 (signal-disagreement) λ grid to {0, 0.25, 0.5, 0.75, 1} and dump the
raw material for the Appendix D worked examples.

λ enters the method in exactly one place: the ranking u = q + λ·√v. The candidate domain
U = A∪B∪C and the feasibility constraint (q >= τ_q, v <= τ_v) are λ-independent
(select_s5.py:220-243), so new λ values are a RE-RANK + RE-TRUNCATE over the fixed
survivor pool U_τ -- no re-scoring, no re-thresholding.

To guarantee that, this script does not reimplement any of the selection logic: it
imports the primitives from 05_select_s5_variants/select_s5.py and reproduces the
λ ∈ {0, 0.5, 1} selections byte-for-byte against their saved doc_ids.npy before writing
anything. Any mismatch aborts.

Writes (all NEW paths; nothing pre-existing is touched):
  <TRAIN>/signal-disagreement-lambda025/{doc_ids.npy,_manifest.json}
  <TRAIN>/signal-disagreement-lambda075/{doc_ids.npy,_manifest.json}
  06_lambda_grid/lambda_grid.npz     per-survivor q, v, tok, rank_λ, selection pattern
  06_lambda_grid/DOMAIN_STATS.md     U / U_τ distributions, τ placement, floor×cap table

Deliberately does NOT run PASS2 -- no parquet blocks are materialised for the new λ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '05_select_s5_variants'))

# select_s5 is __main__-guarded, so importing is side-effect free. Crucially this also
# inherits RNG_TIE == default_rng(SeedSequence(42).spawn(8)[1]) -- the tie-break stream
# must be identical or the id sets drift on ties in u.
from select_s5 import (  # noqa: E402
    N_EXPECT, TARGET, TRAIN, VAL_DIR,
    fill_to, load_doc_ids, mask_of, pass1, top10_by_tokens,
    write_variant_manifest, RNG_TIE, log, stop,
)

LAMBDAS = [(0.0, 'signal-disagreement-lambda0'),
           (0.25, 'signal-disagreement-lambda025'),
           (0.5, 'signal-disagreement-lambda05'),
           (0.75, 'signal-disagreement-lambda075'),
           (1.0, 'signal-disagreement-lambda1')]
# the three that already exist on disk -> the regression gate
KNOWN = {0.0: 'signal-disagreement-lambda0',
         0.5: 'signal-disagreement-lambda05',
         1.0: 'signal-disagreement-lambda1'}
NEW = {0.25: 'signal-disagreement-lambda025',
       0.75: 'signal-disagreement-lambda075'}

# invariants from the original run's manifests / report; the gate refuses to proceed
# unless the rebuilt domain matches these exactly.
EXPECT = dict(U_docs=14_982_068, surv_docs=10_435_667, surv_tok=16_703_787_941,
              tau_q=0.6956847310066223, tau_v=0.10364920496940616)

PCTS = [0, 1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 100]


def dist(a):
    """Percentile table + moments for a 1-D array."""
    a = np.asarray(a, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                pct={p: float(np.percentile(a, p)) for p in PCTS})


def fmt_row(name, d, fmt='.6f'):
    cells = [format(d['pct'][p], fmt) for p in PCTS]
    return f'| {name} | ' + ' | '.join(cells) + f' | {d["mean"]:{fmt}} | {d["std"]:{fmt}} |'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q-floor-pct', type=float, default=30.0)
    ap.add_argument('--v-cap-pct', type=float, default=90.0)
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    t_start = time.time()
    log(f'START: cpus={cpus}; λ grid = {[l for l, _ in LAMBDAS]}')

    val_npy = VAL_DIR / 'val_doc_ids.npy'
    if not val_npy.exists():
        stop(f'missing {val_npy}')

    # ---- identical to select_s5.main() ----------------------------------------
    tok, ft, fw, mb = pass1(cpus)
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3.0).astype(np.float32)
    tie = RNG_TIE.permutation(N_EXPECT).astype(np.int64)

    val_idx = np.load(val_npy).astype(np.int64)
    shared_idx = load_doc_ids(TRAIN / 'shared-top-5B')
    remaining_idx = np.flatnonzero(~mask_of(val_idx) & ~mask_of(shared_idx))
    log(f'EXCLUDE: val={val_idx.size:,}, shared-top={shared_idx.size:,}; '
        f'REMAINING={remaining_idx.size:,} docs, {int(tok[remaining_idx].sum()):,} tok')

    A = top10_by_tokens(remaining_idx, ft, tok, tie)
    B = top10_by_tokens(remaining_idx, fw, tok, tie)
    C = top10_by_tokens(remaining_idx, mb, tok, tie)
    cand_idx = np.flatnonzero(mask_of(A) | mask_of(B) | mask_of(C))
    log(f'U (=A∪B∪C): {cand_idx.size:,} docs, {int(tok[cand_idx].sum()):,} tok')

    tau_q = float(np.percentile(q[cand_idx], args.q_floor_pct))
    tau_v = float(np.percentile(v[cand_idx], args.v_cap_pct))
    pass_q = q[cand_idx] >= tau_q
    pass_v = v[cand_idx] <= tau_v
    surv = cand_idx[pass_q & pass_v]
    surv_tok = int(tok[surv].sum())
    log(f'CONSTRAINT over U: τ_q=Q{args.q_floor_pct:.0f}(q)={tau_q:.16g}, '
        f'τ_v=Q{args.v_cap_pct:.0f}(v)={tau_v:.16g}; survivors={surv.size:,} / {surv_tok:,} tok')

    # ---- GATE part 1: domain invariants ---------------------------------------
    got = dict(U_docs=int(cand_idx.size), surv_docs=int(surv.size), surv_tok=surv_tok,
               tau_q=tau_q, tau_v=tau_v)
    bad = {k: (got[k], EXPECT[k]) for k in EXPECT if got[k] != EXPECT[k]}
    if bad:
        stop(f'domain invariants differ from the original run: {bad}')
    log('GATE 1/2 OK: domain invariants match the original run exactly.')

    # ---- per-λ ranking (the ONLY λ-dependent step) -----------------------------
    # Work in survivor-local index space for the rank arrays; the global-id ordering
    # used for the gate is exactly select_s5.py:253-255.
    sqrtv = np.sqrt(v)
    nsurv = int(surv.size)
    tok_surv = tok[surv]
    ranks = np.empty((len(LAMBDAS), nsurv), np.int32)
    cutoffs, meta = [], {}
    for li, (lam, name) in enumerate(LAMBDAS):
        u = q[surv] + lam * sqrtv[surv]            # float32, as in select_s5
        loc = np.lexsort((tie[surv], -u.astype(np.float64)))
        order = surv[loc]
        sel, ttok, over, filled = fill_to(order, tok, TARGET)
        ranks[li][loc] = np.arange(nsurv, dtype=np.int32)
        k = int(sel.size) - 1                       # last in-budget rank (0-based)
        cutoffs.append(k)
        meta[name] = dict(lam=lam, idx=np.sort(sel), tokens=ttok, over=over, filled=filled,
                          max_if_unfilled=(None if filled else surv_tok),
                          u_cutoff=float(u[loc[k]]))
        log(f'  λ={lam:<5} {name}: {sel.size:,} docs, {ttok:,} tok, overshoot {over:,}, '
            f'filled={filled}, u@cutoff={meta[name]["u_cutoff"]:.6f}')

    # ---- GATE part 2: byte-exact reproduction of the existing blocks -----------
    for lam, name in LAMBDAS:
        if lam not in KNOWN:
            continue
        ref = np.load(TRAIN / KNOWN[lam] / 'doc_ids.npy')
        new = meta[name]['idx']
        if not np.array_equal(ref, new):
            stop(f'λ={lam}: rebuilt selection differs from saved {KNOWN[lam]}/doc_ids.npy '
                 f'(ref {ref.size:,} vs new {new.size:,}, '
                 f'{int(np.setdiff1d(new, ref).size):,} added / '
                 f'{int(np.setdiff1d(ref, new).size):,} dropped)')
        log(f'GATE 2/2 λ={lam}: byte-exact match vs {KNOWN[lam]}/doc_ids.npy '
            f'({ref.size:,} docs).')

    # ---- write ONLY the new λ blocks (ids + manifest, no parquet) --------------
    for lam, name in NEW.items():
        d = TRAIN / name
        if (d / 'doc_ids.npy').exists():
            stop(f'{d}/doc_ids.npy already exists; refusing to overwrite')
        write_variant_manifest(d, name, meta[name], tau_q, tau_v,
                               SimpleNamespace(q_floor_pct=args.q_floor_pct,
                                               v_cap_pct=args.v_cap_pct))
        np.save(d / 'doc_ids.npy', meta[name]['idx'])
        # note in the manifest that this block has ids only
        mp = d / '_manifest.json'
        m = json.loads(mp.read_text())
        m['materialised'] = 'doc_ids only (no parquet shards); analysis block for Appendix D'
        m['u_cutoff'] = meta[name]['u_cutoff']
        mp.write_text(json.dumps(m, indent=2))
        log(f'wrote {d}/doc_ids.npy ({meta[name]["idx"].size:,} docs) + _manifest.json')

    # ---- analysis dump ---------------------------------------------------------
    sel_bits = np.zeros(nsurv, np.uint8)
    for li in range(len(LAMBDAS)):
        sel_bits |= ((ranks[li] <= cutoffs[li]).astype(np.uint8) << li)
    np.savez(HERE / 'lambda_grid.npz',
             lambdas=np.array([l for l, _ in LAMBDAS], np.float64),
             doc_id=surv.astype(np.int64),
             q=q[surv], v=v[surv], tok=tok_surv.astype(np.int64),
             ranks=ranks, cutoffs=np.array(cutoffs, np.int64), pattern=sel_bits,
             tau_q=np.float64(tau_q), tau_v=np.float64(tau_v),
             u_cutoff=np.array([meta[n]['u_cutoff'] for _, n in LAMBDAS], np.float64))
    log(f'wrote {HERE / "lambda_grid.npz"}')

    write_domain_stats(args, q, v, tok, cand_idx, surv, pass_q, pass_v, meta,
                       cutoffs, sel_bits, tau_q, tau_v)
    log(f'ALL DONE in {time.time() - t_start:.0f}s.')


def write_domain_stats(args, q, v, tok, cand_idx, surv, pass_q, pass_v, meta,
                       cutoffs, sel_bits, tau_q, tau_v):
    """U / U_τ distributions of q and v, where τ_q and τ_v sit, and the floor×cap table."""
    qU, vU, tU = q[cand_idx], v[cand_idx], tok[cand_idx]
    qS, vS = q[surv], v[surv]
    rem = ~(pass_q & pass_v)

    hdr = ('| set | ' + ' | '.join(f'p{p}' for p in PCTS) + ' | mean | std |\n'
           + '|---|' + '---:|' * (len(PCTS) + 2))

    L = ['# Feasible-domain statistics for the S5 (Disagreement Aware) selection', '',
         'Generated by `06_lambda_grid/select_lambda_grid.py`. All numbers reproduce the',
         'original run exactly (domain invariants + byte-exact λ=0/0.5/1 id sets).', '',
         '## Definitions', '',
         '- `r1,r2,r3` = global percentile ranks of the fastText / FineWeb-Edu / ModernBERT',
         '  scorers over all 99,949,162 documents (`*-ranking-v2` columns).',
         '- `q = mean(r1,r2,r3)` (consensus quality), `v = var(r1,r2,r3)` (disagreement).',
         '- `U = A∪B∪C`, each scorer\'s top-10%-by-tokens over REMAINING (all docs minus val',
         '  minus shared-top-5B).',
         f'- `τ_q = Q{args.q_floor_pct:.0f}(q over U) = {tau_q:.16g}` (quality floor).',
         f'- `τ_v = Q{args.v_cap_pct:.0f}(v over U) = {tau_v:.16g}` (disagreement cap).',
         '- `U_τ = {d ∈ U : q(d) ≥ τ_q and v(d) ≤ τ_v}` (feasible domain).',
         '- λ enters only via the ranking `u = q + λ·√v`, then truncate to 10B train tokens.',
         '', '## Sizes', '',
         '| set | docs | tokens |', '|---|---:|---:|',
         f'| U (candidate domain) | {cand_idx.size:,} | {int(tU.sum()):,} |',
         f'| U_τ (feasible domain) | {surv.size:,} | {int(tok[surv].sum()):,} |',
         f'| removed by floor and/or cap | {int(rem.sum()):,} | {int(tU[rem].sum()):,} |',
         '',
         f'The two constraints keep **{100.0*surv.size/cand_idx.size:.2f}%** of U by document',
         f'count and **{100.0*int(tok[surv].sum())/int(tU.sum()):.2f}%** by token count.',
         '', '## Where τ_q and τ_v sit', '',
         'Inside `U_τ` this question is degenerate: by construction `τ_q = min(q)` and',
         '`τ_v = max(v)` there. The informative placement is within **U**, which is what the',
         'floor and the cap actually cut:', '',
         f'- `τ_q = {tau_q:.6f}` is the **{args.q_floor_pct:.0f}th percentile of q over U** — it removes the',
         f'  bottom {args.q_floor_pct:.0f}% of U by consensus quality',
         f'  ({int((~pass_q).sum()):,} docs, {int(tU[~pass_q].sum()):,} tokens =',
         f'  {100.0*int(tU[~pass_q].sum())/int(tU.sum()):.2f}% of U tokens).',
         f'- `τ_v = {tau_v:.6f}` is the **{args.v_cap_pct:.0f}th percentile of v over U** — it removes the',
         f'  top {100-args.v_cap_pct:.0f}% of U by disagreement',
         f'  ({int((~pass_v).sum()):,} docs, {int(tU[~pass_v].sum()):,} tokens =',
         f'  {100.0*int(tU[~pass_v].sum())/int(tU.sum()):.2f}% of U tokens).',
         '', '### Floor × cap contingency (the two cuts are far from independent)', '',
         '| | v ≤ τ_v (pass cap) | v > τ_v (cut by cap) | total |',
         '|---|---:|---:|---:|']
    for qlab, qm in [('q ≥ τ_q (pass floor)', pass_q), ('q < τ_q (cut by floor)', ~pass_q)]:
        a = int((qm & pass_v).sum()); b = int((qm & ~pass_v).sum())
        L.append(f'| **{qlab}** | {a:,} | {b:,} | {a+b:,} |')
    L.append(f'| **total** | {int(pass_v.sum()):,} | {int((~pass_v).sum()):,} | {cand_idx.size:,} |')
    only_cap = int((pass_q & ~pass_v).sum())
    L += ['',
          f'Independent cuts would remove {args.q_floor_pct + (100-args.v_cap_pct):.0f}% of U; the actual removal is',
          f'{100.0*int(rem.sum())/cand_idx.size:.2f}%. Only **{only_cap:,} docs '
          f'({100.0*only_cap/cand_idx.size:.2f}% of U)** are removed by the disagreement cap',
          'alone — every other doc the cap rejects is already below the quality floor. The cap',
          'is therefore a narrow guard against pathological disagreement, not a second filter.',
          '', '## Distribution of q', '', hdr]
    L.append(fmt_row('q over U', dist(qU)))
    L.append(fmt_row('q over U_τ', dist(qS)))
    L.append(fmt_row('q over U \\ U_τ (removed)', dist(qU[rem])))
    L += ['', '## Distribution of v', '', hdr]
    L.append(fmt_row('v over U', dist(vU), '.6g'))
    L.append(fmt_row('v over U_τ', dist(vS), '.6g'))
    L.append(fmt_row('v over U \\ U_τ (removed)', dist(vU[rem]), '.6g'))

    L += ['', '## Per-λ selection over U_τ', '',
          '| λ | docs | % of U_τ docs | tokens | overshoot | rank cutoff (1-based) | u at cutoff |',
          '|---:|---:|---:|---:|---:|---:|---:|']
    for (lam, name), k in zip(LAMBDAS, cutoffs):
        m = meta[name]
        L.append(f'| {lam} | {m["idx"].size:,} | {100.0*m["idx"].size/surv.size:.2f}% | '
                 f'{m["tokens"]:,} | {m["over"]:,} | {k+1:,} | {m["u_cutoff"]:.6f} |')

    counts = np.bincount(sel_bits, minlength=32)
    L += ['', '## Selection-pattern histogram over U_τ', '',
          'Bit i = in budget at λ = ' + ', '.join(str(l) for l, _ in LAMBDAS)
          + ' (bit 0 = λ=0, leftmost digit below = λ=1).', '',
          '| pattern (λ=1 … λ=0) | docs | % of U_τ |', '|---|---:|---:|']
    for b in np.argsort(-counts):
        if counts[b] == 0:
            continue
        bits = ''.join('1' if (b >> i) & 1 else '0' for i in range(4, -1, -1))
        L.append(f'| `{bits}` | {counts[b]:,} | {100.0*counts[b]/surv.size:.3f}% |')
    always_in = int(counts[0b11111]); never = int(counts[0])
    L += ['',
          f'- Selected at **every** λ: {always_in:,} docs ({100.0*always_in/surv.size:.2f}% of U_τ).',
          f'- Selected at **no** λ: {never:,} docs ({100.0*never/surv.size:.2f}% of U_τ).',
          f'- Status changes with λ: {surv.size-always_in-never:,} docs '
          f'({100.0*(surv.size-always_in-never)/surv.size:.2f}% of U_τ).', '']

    (HERE / 'DOMAIN_STATS.md').write_text('\n'.join(L))
    log(f'wrote {HERE / "DOMAIN_STATS.md"}')


if __name__ == '__main__':
    main()
