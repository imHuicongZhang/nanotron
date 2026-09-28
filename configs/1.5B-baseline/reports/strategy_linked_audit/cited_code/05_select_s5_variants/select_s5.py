#!/usr/bin/env python
"""Build 3 SAVED S5 (signal-disagreement) candidate datasets at 10B scale.

Promotes the analysis-only signal-disagreement selections from 04_select into saved
datasets. Same domain / constraint / objective as that run's analysis block:
  - REMAINING = all docs minus val minus shared-top-5B (loaded from disk for exact match).
  - A,B,C = each scorer's top-10%-by-tokens over REMAINING (r1=fasttext, r2=fineweb, r3=modernbert).
  - U = A∪B∪C  (candidate domain).
  - q(d)=mean(r1,r2,r3); v(d)=variance(r1,r2,r3).
  - constraint over U: keep q>=Q30(q over U) AND v<=Q90(v over U).
  - per λ∈{1.5,2.0,3.0}: rank survivors by u=q+λ·√v DESC, fill 10B by (tokens-llama2+1).

Saved (under experiments/train/10B/):
  signal-disagreement-lambda15  λ=1.5
  signal-disagreement-lambda2   λ=2.0
  signal-disagreement-lambda3   λ=3.0
  (λ=0.0/0.5/1.0 were built in an earlier run and are intentionally left untouched.)
each as 200 parquet shards (all source columns) + doc_ids.npy + _manifest.json.
The candidate domain U and the Q30/V90 constraint are λ-independent, so these variants
share the exact survivor pool of the earlier λ=0/0.5/1 run.

Read-only on 6_merged_clean. seed=42. Idempotent, atomic per-shard write.
Budgets use TRAIN length = (tokens-llama2 + 1).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- paths / constants
BASE   = Path('/scratch/bvandur1/zhuicon1/data_rewrite')
CLEAN  = BASE / '6_merged_clean'
EXP    = BASE / 'experiments'
TRAIN  = EXP / 'train' / '10B'
VAL_DIR = EXP / 'val'

NSHARDS, ROWS_FULL, LAST_ROWS, N_EXPECT = 200, 500_000, 449_162, 99_949_162
TARGET = 10_000_000_000          # 10B (tokens-llama2 + 1)
SEED = 42

V2_FT, V2_FW, V2_MB = 'fasttext-ranking-v2', 'fineweb-edu-ranking-v2', 'modernbert-ranking-v2'

# variants: (lambda, output folder name).
# λ=0/0.5/1 already built in an earlier run; only the new higher-λ variants are saved here.
VARIANTS = [(1.5, 'signal-disagreement-lambda15'),
            (2.0, 'signal-disagreement-lambda2'),
            (3.0, 'signal-disagreement-lambda3')]
# baseline blocks (already saved) for Jaccard reporting
BASELINE_BLOCKS = ['quality-first', 'diversity-first', 'wrap']

# seed sequence children (identical mapping to the 04_select run; only the tie-break is used here)
_SS = np.random.SeedSequence(SEED)
_CH = _SS.spawn(8)
RNG_TIE = np.random.default_rng(_CH[1])   # child1: global tie-break priority for DESC ranking


def shard_path(i): return CLEAN / f'merged_clean_{i:05d}.parquet'
def shard_rows(i): return ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS
def shard_offset(i): return i * ROWS_FULL
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- PASS 1: load numeric cols
def _read_numeric(i):
    t = pq.read_table(shard_path(i),
                      columns=['doc_id', 'tokens-llama2', V2_FT, V2_FW, V2_MB],
                      use_threads=False)
    did = t.column('doc_id').to_numpy(zero_copy_only=False)
    tok = t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64) + 1  # +1 BOS
    ft = t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32)
    fw = t.column(V2_FW).to_numpy(zero_copy_only=False).astype(np.float32)
    mb = t.column(V2_MB).to_numpy(zero_copy_only=False).astype(np.float32)
    nulls = {c: t.column(c).null_count for c in ['tokens-llama2', V2_FT, V2_FW, V2_MB]}
    return i, t.num_rows, did, tok, ft, fw, mb, nulls


def pass1(workers):
    log(f'PASS1: loading numeric cols across 200 shards with {workers} workers...')
    tok = np.empty(N_EXPECT, np.int64)
    ft = np.empty(N_EXPECT, np.float32); fw = np.empty(N_EXPECT, np.float32); mb = np.empty(N_EXPECT, np.float32)
    null_tot = Counter()
    t0 = time.time(); seen = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_read_numeric, i) for i in range(NSHARDS)]
        for fut in as_completed(futs):
            i, n, did, tk, a, b, c, nl = fut.result()
            if n != shard_rows(i): stop(f'shard {i}: {n} rows != {shard_rows(i)}')
            lo = shard_offset(i); hi = lo + n
            if not np.array_equal(did, np.arange(lo, hi, dtype=did.dtype)):
                stop(f'shard {i}: doc_id not contiguous')
            tok[lo:hi] = tk; ft[lo:hi] = a; fw[lo:hi] = b; mb[lo:hi] = c
            for k, vv in nl.items(): null_tot[k] += vv
            seen += 1
            if seen % 50 == 0: log(f'  PASS1 {seen}/200 ({time.time()-t0:.0f}s)')
    if any(null_tot.values()):
        stop(f'nulls found: {dict(null_tot)}')
    log(f'PASS1 OK ({time.time()-t0:.0f}s): N={N_EXPECT:,}, zero nulls.')
    return tok, ft, fw, mb


# ----------------------------------------------------------------------------- selection primitives
def order_desc(idxs, score, tie):
    """idxs sorted by score DESC, ties by random priority `tie` ASC (deterministic)."""
    return idxs[np.lexsort((tie[idxs], -score[idxs].astype(np.float64)))]


def fill_to(order, tok, target):
    """Accumulate (tok) along `order` until cumsum first >= target; keep last doc whole.
    Returns (selected_idx, total_tok, overshoot, filled_bool)."""
    if order.size == 0:
        return order, 0, -int(target), False
    c = np.cumsum(tok[order])
    if c[-1] < target:
        return order, int(c[-1]), int(c[-1] - target), False
    i = int(np.searchsorted(c, target, side='left'))
    return order[:i + 1], int(c[i]), int(c[i] - target), True


def jaccard(a_mask, b_mask):
    inter = int(np.count_nonzero(a_mask & b_mask))
    union = int(np.count_nonzero(a_mask | b_mask))
    return (inter / union if union else 0.0), inter, union


def mask_of(idx):
    m = np.zeros(N_EXPECT, bool); m[idx] = True; return m


def top10_by_tokens(idxs, score, tok, tie):
    order = order_desc(idxs, score, tie)
    target = 0.10 * int(tok[idxs].sum())
    sel, _, _, _ = fill_to(order, tok, target)
    return sel


def qstats(a):
    """mean/median/p10/p90 (None if empty)."""
    if a.size == 0:
        return dict(mean=None, median=None, p10=None, p90=None)
    return dict(mean=float(a.mean()), median=float(np.percentile(a, 50)),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


def lenstats(a):
    """mean/median/p90/p99 (None if empty)."""
    if a.size == 0:
        return dict(mean=None, median=None, p90=None, p99=None)
    return dict(mean=float(a.mean()), median=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)))


def load_doc_ids(block_dir):
    """Concatenate the doc_id column across a saved block's 200 parquet shards.
    doc_id is the contiguous global row index, so these double as global indices."""
    ids = []
    for k in range(NSHARDS):
        p = block_dir / f'part_{k:05d}.parquet'
        if not p.exists():           # shards with 0 selected rows produce no file
            continue
        t = pq.read_table(p, columns=['doc_id'], use_threads=False)
        ids.append(t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64))
    if not ids:
        stop(f'no parquet shards found under {block_dir}')
    return np.concatenate(ids)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q-floor-pct', type=float, default=30.0)
    ap.add_argument('--v-cap-pct', type=float, default=90.0)
    ap.add_argument('--write-mem-per-worker-gb', type=float, default=7.0)
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    write_workers = max(1, min(cpus, int((mem_gb - 16) // args.write_mem_per_worker_gb)))
    log(f'START: cpus={cpus} mem={mem_gb:.0f}G -> pass1 {cpus} workers, write {write_workers} workers; '
        f'seed={SEED}; q_floor_pct={args.q_floor_pct} v_cap_pct={args.v_cap_pct}')
    t_start = time.time()

    # require val + the four exclusion/baseline blocks to exist
    val_npy = VAL_DIR / 'val_doc_ids.npy'
    if not val_npy.exists(): stop(f'missing {val_npy}')
    for nm in ['shared-top-5B'] + BASELINE_BLOCKS:
        if not (TRAIN / nm).is_dir(): stop(f'missing baseline block dir {TRAIN / nm}')

    tok, ft, fw, mb = pass1(cpus)
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3.0).astype(np.float32)
    tie = RNG_TIE.permutation(N_EXPECT).astype(np.int64)

    # ---- exclusions loaded from disk (exact match to the 04_select run) --------
    val_idx = np.load(val_npy).astype(np.int64)
    shared_idx = load_doc_ids(TRAIN / 'shared-top-5B')
    val_mask = mask_of(val_idx)
    shared_mask = mask_of(shared_idx)
    remaining_mask = ~val_mask & ~shared_mask
    remaining_idx = np.flatnonzero(remaining_mask)
    rem_tok = int(tok[remaining_idx].sum())
    log(f'EXCLUDE: val={val_idx.size:,}, shared-top={shared_idx.size:,}; '
        f'REMAINING={remaining_idx.size:,} docs, {rem_tok:,} tok')

    # ---- STEP 0: domain A∪B∪C over REMAINING (report first) --------------------
    A = top10_by_tokens(remaining_idx, ft, tok, tie)
    B = top10_by_tokens(remaining_idx, fw, tok, tie)
    C = top10_by_tokens(remaining_idx, mb, tok, tie)
    mA, mB, mC = mask_of(A), mask_of(B), mask_of(C)
    cand_mask = mA | mB | mC
    inter3_mask = mA & mB & mC
    cand_idx = np.flatnonzero(cand_mask)
    votes = mA.astype(np.int8) + mB.astype(np.int8) + mC.astype(np.int8)   # voters over REMAINING (=A/B/C)
    one_mask, two_mask, three_mask = votes == 1, votes == 2, votes == 3

    def dt(idx):
        idx = np.flatnonzero(idx) if idx.dtype == bool else idx
        return int(idx.size), int(tok[idx].sum())
    domain = {k: dict(zip(('docs', 'tokens'), dt(x))) for k, x in
              {'A_fasttext': A, 'B_fineweb': B, 'C_modernbert': C,
               'intersection_ABC': inter3_mask, 'union_ABC': cand_idx}.items()}
    for k in ['A_fasttext', 'B_fineweb', 'C_modernbert', 'intersection_ABC', 'union_ABC']:
        log(f'  STEP0 {k}: {domain[k]["docs"]:,} docs, {domain[k]["tokens"]:,} tok')

    # ---- constraint over U -----------------------------------------------------
    Q30 = float(np.percentile(q[cand_idx], args.q_floor_pct))
    V90 = float(np.percentile(v[cand_idx], args.v_cap_pct))
    keep = (q[cand_idx] >= Q30) & (v[cand_idx] <= V90)
    surv = cand_idx[keep]
    surv_tok = int(tok[surv].sum())
    log(f'CONSTRAINT over U: Q{args.q_floor_pct:.0f}(q)={Q30:.6f}, Q{args.v_cap_pct:.0f}(v)={V90:.6g}; '
        f'survivors={surv.size:,} docs / {surv_tok:,} tok (can fill 10B: {surv_tok >= TARGET})')

    # ---- per-λ selection -------------------------------------------------------
    sqrtv = np.sqrt(v)
    sel_masks = {}      # folder name -> bool mask
    sel_meta = {}       # folder name -> dict
    for lam, name in VARIANTS:
        u = q[surv] + lam * sqrtv[surv]
        order = surv[np.lexsort((tie[surv], -u.astype(np.float64)))]
        sel, ttok, over, filled = fill_to(order, tok, TARGET)
        sel_masks[name] = mask_of(sel)
        sel_meta[name] = dict(lam=lam, idx=np.sort(sel), tokens=ttok, over=over, filled=filled,
                              max_if_unfilled=(None if filled else surv_tok))
        log(f'  {name}: λ={lam} -> {sel.size:,} docs, {ttok:,} tok, overshoot {over:,}, filled={filled}'
            + ('' if filled else f' (MAX {surv_tok:,})'))

    # ---- per-variant statistics (needs q/v/tok; build BEFORE freeing) ----------
    base_masks = {nm: mask_of(load_doc_ids(TRAIN / nm)) for nm in BASELINE_BLOCKS}
    stats = build_stats(sel_meta, sel_masks, base_masks, q, v, tok,
                        one_mask, two_mask, three_mask, val_mask, shared_mask)
    report = dict(seed=SEED, target_tokens=TARGET, q_floor_pct=args.q_floor_pct, v_cap_pct=args.v_cap_pct,
                  remaining=dict(docs=int(remaining_idx.size), tokens=rem_tok),
                  domain=domain, constraint=dict(Q30=Q30, V90=V90,
                                                 survivors=dict(docs=int(surv.size), tokens=surv_tok),
                                                 can_fill_10B=bool(surv_tok >= TARGET)),
                  variants=stats['variants'], jaccard=stats['jaccard'],
                  voter_sizes=dict(one=int(one_mask.sum()), two=int(two_mask.sum()),
                                   three=int(three_mask.sum())),
                  exclusivity=stats['exclusivity'])

    # write manifests, doc_ids, report (no longer need the big arrays after this)
    for lam, name in VARIANTS:
        write_variant_manifest(TRAIN / name, name, sel_meta[name], Q30, V90, args)
        d = TRAIN / name; d.mkdir(parents=True, exist_ok=True)
        np.save(d / 'doc_ids.npy', sel_meta[name]['idx'])
    write_report(report)
    log('manifests + doc_ids + report written.')

    # free big arrays before forking writers; keep only the selection masks
    del tok, ft, fw, mb, q, v, tie, sqrtv, mA, mB, mC, cand_mask, inter3_mask, base_masks
    import gc; gc.collect()

    write_blocks(sel_masks, write_workers)
    log(f'ALL DONE in {time.time()-t_start:.0f}s.')


# ----------------------------------------------------------------------------- stats / report
def build_stats(sel_meta, sel_masks, base_masks, q, v, tok,
                one_mask, two_mask, three_mask, val_mask, shared_mask):
    names = [n for _, n in VARIANTS]
    variants = {}
    exclusivity = {}
    for name in names:
        m = sel_masks[name]; idx = sel_meta[name]['idx']; sz = int(idx.size)
        variants[name] = dict(
            lam=sel_meta[name]['lam'], docs=sz, tokens=sel_meta[name]['tokens'],
            overshoot=sel_meta[name]['over'], filled=sel_meta[name]['filled'],
            max_tokens_if_unfilled=sel_meta[name]['max_if_unfilled'],
            q=qstats(q[idx]), v=qstats(v[idx]), doc_len=lenstats((tok[idx] - 1).astype(np.float64)),
            inside_one_pct=100.0 * int(np.count_nonzero(m & one_mask)) / sz if sz else 0.0,
            inside_two_pct=100.0 * int(np.count_nonzero(m & two_mask)) / sz if sz else 0.0,
            inside_three_pct=100.0 * int(np.count_nonzero(m & three_mask)) / sz if sz else 0.0)
        exclusivity[name] = dict(val_cap=int(np.count_nonzero(val_mask & m)),
                                 shared_cap=int(np.count_nonzero(shared_mask & m)))
    # Jaccard: 3x3 among variants + each variant vs the 3 baselines
    J = {}
    for a in names:
        row = {}
        for b in names:
            jc, inter, _ = jaccard(sel_masks[a], sel_masks[b])
            row[b] = dict(jaccard=jc, shared=inter)
        for b in BASELINE_BLOCKS:
            jc, inter, _ = jaccard(sel_masks[a], base_masks[b])
            row[b] = dict(jaccard=jc, shared=inter)
        J[a] = row
    return dict(variants=variants, jaccard=J, exclusivity=exclusivity)


def write_variant_manifest(d, name, meta, Q30, V90, args):
    d.mkdir(parents=True, exist_ok=True)
    m = dict(block=name, target_tokens=int(TARGET), lambda_=meta['lam'],
             docs=int(meta['idx'].size), train_tokens_sum=int(meta['tokens']),
             overshoot_tokens=int(meta['over']), filled_10B=bool(meta['filled']),
             max_tokens_if_unfilled=meta['max_if_unfilled'], seed=SEED,
             q_floor_pct=args.q_floor_pct, v_cap_pct=args.v_cap_pct, Q30=Q30, V90=V90,
             method='U=A∪B∪C (each scorer top-10%-by-tokens over REMAINING); '
                    'keep q>=Q30 & v<=Q90 over U; rank u=q+λ·√v DESC; fill 10B by (tokens-llama2+1)')
    (d / '_manifest.json').write_text(json.dumps(m, indent=2))


def _f(x, fmt):
    return 'n/a' if x is None else format(x, fmt)


def write_report(R):
    names = [n for _, n in VARIANTS]
    L = ['# S5 (signal-disagreement) candidate variants — selection report', '',
         f'- seed={R["seed"]}; budgets use (tokens-llama2 + 1); target = {R["target_tokens"]:,}.',
         f'- REMAINING (all minus val minus shared-top-5B): {R["remaining"]["docs"]:,} docs, '
         f'{R["remaining"]["tokens"]:,} tok.',
         f'- q=mean(r1,r2,r3); v=variance(r1,r2,r3); constraint over U: q>=Q{R["q_floor_pct"]:.0f} '
         f'AND v<=Q{R["v_cap_pct"]:.0f}.', '',
         '## STEP 0 — candidate domain A∪B∪C over REMAINING',
         '| set | docs | tokens |', '|---|---:|---:|']
    for k in ['A_fasttext', 'B_fineweb', 'C_modernbert', 'intersection_ABC', 'union_ABC']:
        L.append(f'| {k} | {R["domain"][k]["docs"]:,} | {R["domain"][k]["tokens"]:,} |')
    c = R['constraint']
    L += ['', '## Constraint over U',
          f'- Q30(q over U) = {c["Q30"]:.6f}; Q90(v over U) = {c["V90"]:.6g}',
          f'- survivors: {c["survivors"]["docs"]:,} docs / {c["survivors"]["tokens"]:,} tok '
          f'(can fill 10B: {c["can_fill_10B"]})', '',
          '## Variants (fill 10B by tokens-llama2+1)',
          '| variant | λ | docs | tokens | overshoot | filled | max if unfilled |',
          '|---|---:|---:|---:|---:|:--:|---:|']
    for name in names:
        m = R['variants'][name]
        L.append(f'| {name} | {m["lam"]} | {m["docs"]:,} | {m["tokens"]:,} | {m["overshoot"]:,} | '
                 f'{m["filled"]} | {_f(m["max_tokens_if_unfilled"], ",")} |')
    L += ['', '## q / v / doc-length distributions',
          '| variant | q̄ | q p50 | q p10 | q p90 | v̄ | v p50 | v p10 | v p90 | '
          'len μ | len p50 | len p90 | len p99 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m = R['variants'][name]; qd = m['q']; vd = m['v']; ld = m['doc_len']
        L.append(f'| {name} | {_f(qd["mean"],".4f")} | {_f(qd["median"],".4f")} | {_f(qd["p10"],".4f")} '
                 f'| {_f(qd["p90"],".4f")} | {_f(vd["mean"],".5f")} | {_f(vd["median"],".5f")} '
                 f'| {_f(vd["p10"],".5f")} | {_f(vd["p90"],".5f")} | {_f(ld["mean"],".0f")} '
                 f'| {_f(ld["median"],".0f")} | {_f(ld["p90"],".0f")} | {_f(ld["p99"],".0f")} |')
    vs = R['voter_sizes']
    L += ['', f'## Voter membership (voters over REMAINING: one={vs["one"]:,}, two={vs["two"]:,}, '
          f'three={vs["three"]:,}; one=exactly-1 of A/B/C, two=exactly-2, three=A∩B∩C)',
          '| variant | %inside 1-voter | %inside 2-voter | %inside 3-voter |',
          '|---|---:|---:|---:|']
    for name in names:
        m = R['variants'][name]
        L.append(f'| {name} | {m["inside_one_pct"]:.1f}% | {m["inside_two_pct"]:.1f}% '
                 f'| {m["inside_three_pct"]:.1f}% |')
    cols = names + BASELINE_BLOCKS
    L += ['', '## Jaccard matrix (rows = S5 variants; cols = S5 variants + baseline blocks)',
          '| | ' + ' | '.join(cols) + ' |',
          '|' + '---|' * (len(cols) + 1)]
    for a in names:
        cells = [f'{R["jaccard"][a][b]["jaccard"]:.4f}' for b in cols]
        L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Exclusivity (must be 0)', '| variant | val ∩ | shared-top ∩ |', '|---|---:|---:|']
    for name in names:
        e = R['exclusivity'][name]
        L.append(f'| {name} | {e["val_cap"]} | {e["shared_cap"]} |')
    L += ['']
    TRAIN.mkdir(parents=True, exist_ok=True)
    (TRAIN / 'S5_VARIANTS_REPORT.md').write_text('\n'.join(L))
    (TRAIN / 'S5_variants_stats.json').write_text(json.dumps(R, indent=2))


# ----------------------------------------------------------------------------- PASS 2: write saved blocks
_WMASKS = {}    # set as a module global in write_blocks(); forked workers inherit via COW


def _write_one(k):
    p = shard_path(k)
    lo = shard_offset(k); n = shard_rows(k)
    counts = {}
    plan = {}
    for name, mask in _WMASKS.items():
        sel = mask[lo:lo + n]
        cnt = int(sel.sum())
        counts[name] = cnt
        if cnt == 0:
            continue
        outp = TRAIN / name / f'part_{k:05d}.parquet'
        if outp.exists():
            try:
                if pq.ParquetFile(outp).metadata.num_rows == cnt:
                    continue
            except Exception:
                pass
        plan[name] = (sel, outp)
    if plan:
        t = pq.read_table(p, use_threads=False)
        for name, (sel, outp) in plan.items():
            outp.parent.mkdir(parents=True, exist_ok=True)
            sub = t.filter(pa.array(sel))
            tmp = outp.with_suffix(outp.suffix + '.tmp')
            pq.write_table(sub, tmp, compression='zstd')
            os.replace(tmp, outp)
            if pq.ParquetFile(outp).metadata.num_rows != int(sel.sum()):
                raise RuntimeError(f'{name} shard {k}: rowcount mismatch')
    return k, counts


def write_blocks(masks, workers):
    log(f'PASS2: writing {list(masks.keys())} with {workers} workers...')
    global _WMASKS
    _WMASKS = masks                          # set BEFORE fork; workers inherit via COW
    totals = Counter()
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_one, k): k for k in range(NSHARDS)}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                _, counts = fut.result()
            except Exception as e:   # noqa: BLE001
                stop(f'PASS2 shard {k:05d} failed: {e!r}')
            for nm, cnt in counts.items(): totals[nm] += cnt
            done += 1
            if done % 25 == 0: log(f'  PASS2 {done}/200 ({time.time()-t0:.0f}s)')
    log(f'PASS2 OK ({time.time()-t0:.0f}s): written rows per set: {dict(totals)}')
    for nm, m in masks.items():
        if int(m.sum()) != totals[nm]:
            stop(f'{nm}: written {totals[nm]} != selected {int(m.sum())}')


if __name__ == '__main__':
    main()
