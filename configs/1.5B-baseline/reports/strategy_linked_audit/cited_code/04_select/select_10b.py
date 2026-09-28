#!/usr/bin/env python
"""Build 6 experiment-setting datasets at 10B scale + a statistics-only analysis
for the 6th (signal-disagreement). Selection happens in-memory over the numeric
columns; the 6 saved blocks are written as parquet shards in a parallel pass.
Read-only on 6_merged_clean. seed=42. Idempotent, atomic per-shard write.

Saved blocks (under experiments/train/10B/):
  shared-top-5B   5B  common core, fasttext-v2 DESC over (all minus val)
  quality-base    5B  next-best by fasttext-v2 DESC after SHARED
  quality-first  10B  next-best by fasttext-v2 DESC after SHARED (deeper prefix)
  wrap           10B  uniform sample from REMAINING
  rewrite        20B  uniform sample from REMAINING (= 10B * 2)
  diversity-first 10B topic-stratified, per-category quota by token share, q DESC

signal-disagreement: STATISTICS ONLY this run (no dataset saved) -> a report under
experiments/analysis/signal-disagreement/. We decide lambda + V_cap afterwards.

Budgets use TRAIN length = (tokens-llama2 + 1)  [one leading BOS].
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
ANALYSIS_DIR = EXP / 'analysis' / 'signal-disagreement'
CODE_DIR = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/04_select')

NSHARDS, ROWS_FULL, LAST_ROWS, N_EXPECT = 200, 500_000, 449_162, 99_949_162

# per-scale token budgets (TRAIN length = tokens-llama2 + 1)
SHARED_TARGET  =  5_000_000_000   # shared-top-5B
QBASE_TARGET   =  5_000_000_000   # quality-base
TARGET         = 10_000_000_000   # variable-block default (quality-first / wrap / diversity / analysis)
REWRITE_TARGET = 20_000_000_000   # rewrite = 10B * 2

# block -> target token budget (drives manifests + report)
BLOCK_TARGETS = {
    'shared-top-5B':  SHARED_TARGET,
    'quality-base':   QBASE_TARGET,
    'quality-first':  TARGET,
    'wrap':           TARGET,
    'rewrite':        REWRITE_TARGET,
    'diversity-first': TARGET,
}
VAR_BLOCKS = ['quality-base', 'quality-first', 'wrap', 'rewrite', 'diversity-first']

VAL_SIZE = 50_000
SEED = 42

V2_FT, V2_FW, V2_MB = 'fasttext-ranking-v2', 'fineweb-edu-ranking-v2', 'modernbert-ranking-v2'
TOPIC_N = 24

# seed sequence children (documented mapping; identical to the 5B run -> identical val)
_SS = np.random.SeedSequence(SEED)
_CH = _SS.spawn(8)
RNG_VAL   = np.random.default_rng(_CH[0])   # child0: val uniform sample
RNG_TIE   = np.random.default_rng(_CH[1])   # child1: global tie-break priority for DESC ranking
RNG_WRAP  = np.random.default_rng(_CH[2])   # child2: wrap uniform sample
RNG_REWR  = np.random.default_rng(_CH[3])   # child3: rewrite uniform sample


def shard_path(i): return CLEAN / f'merged_clean_{i:05d}.parquet'
def shard_rows(i): return ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS
def shard_offset(i): return i * ROWS_FULL
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- PASS 1: load numeric cols
def _read_numeric(i):
    t = pq.read_table(shard_path(i),
                      columns=['doc_id', 'tokens-llama2', V2_FT, V2_FW, V2_MB, 'topic'],
                      use_threads=False)
    did = t.column('doc_id').to_numpy(zero_copy_only=False)
    tok = t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64) + 1  # +1 BOS
    ft = t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32)
    fw = t.column(V2_FW).to_numpy(zero_copy_only=False).astype(np.float32)
    mb = t.column(V2_MB).to_numpy(zero_copy_only=False).astype(np.float32)
    topic = t.column('topic')
    dic = topic.dictionary_encode()                      # local dict + indices
    local_vocab = dic.combine_chunks().dictionary.to_pylist()
    local_codes = dic.combine_chunks().indices.to_numpy(zero_copy_only=False).astype(np.int32)
    nulls = {c: t.column(c).null_count for c in ['tokens-llama2', V2_FT, V2_FW, V2_MB, 'topic']}
    return i, t.num_rows, did, tok, ft, fw, mb, local_vocab, local_codes, nulls


def pass1(workers):
    log(f'PASS1: loading numeric cols across 200 shards with {workers} workers...')
    tok = np.empty(N_EXPECT, np.int64)
    ft = np.empty(N_EXPECT, np.float32); fw = np.empty(N_EXPECT, np.float32); mb = np.empty(N_EXPECT, np.float32)
    topic_local = [None] * NSHARDS       # (local_vocab, local_codes) per shard
    null_tot = Counter()
    t0 = time.time(); seen = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_read_numeric, i) for i in range(NSHARDS)]
        for fut in as_completed(futs):
            i, n, did, tk, a, b, c, lv, lc, nl = fut.result()
            if n != shard_rows(i): stop(f'shard {i}: {n} rows != {shard_rows(i)}')
            lo = shard_offset(i); hi = lo + n
            if not np.array_equal(did, np.arange(lo, hi, dtype=did.dtype)):
                stop(f'shard {i}: doc_id not contiguous')
            tok[lo:hi] = tk; ft[lo:hi] = a; fw[lo:hi] = b; mb[lo:hi] = c
            topic_local[i] = (lv, lc)
            for k, v in nl.items(): null_tot[k] += v
            seen += 1
            if seen % 50 == 0: log(f'  PASS1 {seen}/200 ({time.time()-t0:.0f}s)')
    if any(null_tot.values()):
        stop(f'nulls found: {dict(null_tot)}')
    # global topic vocab (sorted union) -> int8 codes
    vocab = sorted({s for lv, _ in topic_local for s in lv})
    if len(vocab) != TOPIC_N:
        print('topic vocab:', vocab, flush=True)
        stop(f'distinct topic count {len(vocab)} != {TOPIC_N}; reported above for confirmation.')
    code_of = {s: k for k, s in enumerate(vocab)}
    topic = np.empty(N_EXPECT, np.int8)
    for i in range(NSHARDS):
        lv, lc = topic_local[i]
        remap = np.array([code_of[s] for s in lv], dtype=np.int8)
        lo = shard_offset(i)
        topic[lo:lo + lc.size] = remap[lc]
    log(f'PASS1 OK ({time.time()-t0:.0f}s): N={N_EXPECT:,}, topics={len(vocab)}, zero nulls.')
    return tok, ft, fw, mb, topic, vocab


# ----------------------------------------------------------------------------- selection primitives
def order_desc(idxs, score, tie):
    """idxs sorted by score DESC, ties by random priority `tie` ASC (deterministic)."""
    return idxs[np.lexsort((tie[idxs], -score[idxs].astype(np.float64)))]


def fill_to(order, tok, target):
    """Accumulate (tok) along `order` until cumsum first >= target; keep last doc whole.
    Returns (selected_idx, total_tok, overshoot, filled_bool)."""
    if order.size == 0:
        return order, 0, -target, False
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


def pctl(a):
    """p10/p50/p90/mean of an array (None if empty)."""
    if a.size == 0:
        return dict(p10=None, p50=None, p90=None, mean=None)
    return dict(p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), mean=float(a.mean()))


# ----------------------------------------------------------------------------- main selection
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
    tok, ft, fw, mb, topic, vocab = pass1(cpus)
    # global q (mean) and v (population variance) of the 3 v2 percentiles
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3.0).astype(np.float32)
    tie = RNG_TIE.permutation(N_EXPECT).astype(np.int64)

    # ---- STEP 0: validation set ------------------------------------------------
    val_idx = np.sort(RNG_VAL.choice(N_EXPECT, size=VAL_SIZE, replace=False)).astype(np.int64)
    val_mask = mask_of(val_idx)
    alive_mask = ~val_mask
    alive_idx = np.flatnonzero(alive_mask)
    log(f'STEP0 val: {val_idx.size:,} docs; alive={alive_idx.size:,}')

    # ---- shared-top-5B ---------------------------------------------------------
    avail_order = order_desc(alive_idx, ft, tie)                 # alive, fasttext-v2 DESC
    shared_idx, shared_tok, shared_over, shared_fill = fill_to(avail_order, tok, SHARED_TARGET)
    if not shared_fill: stop('shared-top could not reach 5B.')
    shared_mask = mask_of(shared_idx)
    i_shared = shared_idx.size
    log(f'shared-top-5B: {shared_idx.size:,} docs, {shared_tok:,} tok (overshoot {shared_over:,})')

    # ---- REMAINING = alive minus shared ---------------------------------------
    remaining_mask = alive_mask & ~shared_mask
    remaining_idx = np.flatnonzero(remaining_mask)
    rem_tok_total = int(tok[remaining_idx].sum())
    log(f'REMAINING: {remaining_idx.size:,} docs, {rem_tok_total:,} tok')

    blocks = {}   # name -> dict(idx, tok, over, method[, cats, topup])

    # quality-base: continuation of avail_order after shared, fill 5B
    qf_order = avail_order[i_shared:]
    qb_idx, qb_tok, qb_over, qb_fill = fill_to(qf_order, tok, QBASE_TARGET)
    if not qb_fill: stop('quality-base could not reach 5B.')
    blocks['quality-base'] = dict(idx=qb_idx, tok=qb_tok, over=qb_over,
                                  method='fasttext-v2 DESC (2nd-top, 5B)')

    # quality-first: same fasttext order after shared, fill 10B
    qf_idx, qf_tok, qf_over, qf_fill = fill_to(qf_order, tok, TARGET)
    if not qf_fill: stop('quality-first could not reach 10B.')
    blocks['quality-first'] = dict(idx=qf_idx, tok=qf_tok, over=qf_over,
                                   method='fasttext-v2 DESC (2nd-top, 10B)')

    # wrap: uniform sample from REMAINING, fill 10B
    w_order = remaining_idx[RNG_WRAP.permutation(remaining_idx.size)]
    w_idx, w_tok, w_over, w_fill = fill_to(w_order, tok, TARGET)
    if not w_fill: stop('wrap could not reach 10B.')
    blocks['wrap'] = dict(idx=w_idx, tok=w_tok, over=w_over, method='uniform(seed42 child2)')

    # rewrite: uniform sample from REMAINING, fill 20B (= 10B * 2)
    if rem_tok_total < REWRITE_TARGET:
        stop(f'rewrite: REMAINING tok {rem_tok_total:,} < 20B ({REWRITE_TARGET:,}); cannot fill.')
    r_order = remaining_idx[RNG_REWR.permutation(remaining_idx.size)]
    r_idx, r_tok, r_over, r_fill = fill_to(r_order, tok, REWRITE_TARGET)
    if not r_fill: stop('rewrite could not reach 20B.')
    blocks['rewrite'] = dict(idx=r_idx, tok=r_tok, over=r_over, method='uniform(seed42 child3), 20B')

    # diversity-first: stratified by topic, per-category quota by token share, top by q DESC, fill 10B
    div_sel = []
    rem_topic = topic[remaining_idx]
    cat_report = []
    selected_div_mask = np.zeros(N_EXPECT, bool)
    for c in range(TOPIC_N):
        cat_idx = remaining_idx[rem_topic == c]
        cat_tok = int(tok[cat_idx].sum())
        quota = TARGET * (cat_tok / rem_tok_total)
        order_c = order_desc(cat_idx, q, tie)
        sel_c, tok_c, over_c, fill_c = fill_to(order_c, tok, quota)
        div_sel.append(sel_c)
        selected_div_mask[sel_c] = True
        cat_report.append(dict(cat=vocab[c], docs=int(sel_c.size), tok=int(tok_c),
                               quota=float(quota), filled=bool(fill_c)))
    div_idx = np.concatenate(div_sel)
    div_tok = int(tok[div_idx].sum())
    # top-up shortfall from REMAINING (best q DESC not yet picked), if under target
    if div_tok < TARGET:
        pool = remaining_idx[~selected_div_mask[remaining_idx]]
        pool_order = order_desc(pool, q, tie)
        need = TARGET - div_tok
        add_idx, add_tok, _, _ = fill_to(pool_order, tok, need)
        div_idx = np.concatenate([div_idx, add_idx]); div_tok += int(add_tok)
        topup_docs = int(add_idx.size)
    else:
        topup_docs = 0
    div_over = div_tok - TARGET
    blocks['diversity-first'] = dict(idx=div_idx, tok=div_tok, over=div_over,
                                     method='topic-stratified, q DESC', cats=cat_report, topup=topup_docs)
    log(f'variable blocks built: quality-base {qb_idx.size:,} | quality-first {qf_idx.size:,} | '
        f'wrap {w_idx.size:,} | rewrite {r_idx.size:,} | diversity-first {div_idx.size:,} (topup {topup_docs})')

    # =====================================================================
    # SIGNAL-DISAGREEMENT ANALYSIS (no dataset saved)
    # =====================================================================
    analysis = run_analysis(tok, ft, fw, mb, q, v, tie, remaining_idx, alive_idx,
                            qf_idx, qb_idx, args.q_floor_pct, args.v_cap_pct)

    # =====================================================================
    # build saved-block masks; free big arrays; PASS2 write
    # =====================================================================
    saved = {'shared-top-5B': shared_idx, **{k: blocks[k]['idx'] for k in VAR_BLOCKS}}
    SAVE_MASKS = {name: mask_of(idx) for name, idx in saved.items()}
    SAVE_MASKS_VAL = {'__val__': val_mask, **SAVE_MASKS}

    # overlap / exclusivity report (before freeing masks)
    overlap = overlap_report(SAVE_MASKS, shared_mask, val_mask)

    # manifests for saved blocks
    write_manifests(shared_idx, shared_tok, shared_over, blocks, rem_tok_total, alive_idx.size)
    write_final_report(shared_idx, shared_tok, shared_over, blocks, rem_tok_total,
                       remaining_idx.size, overlap, val_idx)

    # save val ids + analysis outputs
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VAL_DIR / 'val_doc_ids.npy', val_idx.astype(np.int64))
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / 'signal_disagreement_overlap.json').write_text(json.dumps(analysis['json'], indent=2))
    (ANALYSIS_DIR / 'signal_disagreement_report.md').write_text(analysis['md'])
    log('analysis + manifests + val ids written.')

    # free big arrays before forking writers
    del tok, ft, fw, mb, topic, q, v, tie, avail_order, qf_order, w_order, r_order
    import gc; gc.collect()

    write_blocks(SAVE_MASKS_VAL, write_workers)
    log(f'ALL DONE in {time.time()-t_start:.0f}s.')


# ----------------------------------------------------------------------------- analysis
def top10_by_tokens(idxs, score, tok, tie):
    order = order_desc(idxs, score, tie)
    target = 0.10 * int(tok[idxs].sum())
    sel, _, _, _ = fill_to(order, tok, target)
    return sel


def run_analysis(tok, ft, fw, mb, q, v, tie, remaining_idx, alive_idx, qf_idx, qb_idx,
                 q_floor_pct, v_cap_pct):
    log('ANALYSIS (signal-disagreement, stats only)...')
    LAMBDAS = [0.0, 0.3, 0.5, 1.0]
    J = {}

    # STEP 1: candidate domain A∪B∪C over REMAINING (each scorer's top-10% by tokens)
    A = top10_by_tokens(remaining_idx, ft, tok, tie)
    B = top10_by_tokens(remaining_idx, fw, tok, tie)
    C = top10_by_tokens(remaining_idx, mb, tok, tie)
    mA, mB, mC = mask_of(A), mask_of(B), mask_of(C)
    cand_mask = mA | mB | mC
    inter3_mask = mA & mB & mC
    cand_idx = np.flatnonzero(cand_mask)

    def dt(mask_or_idx):
        idx = np.flatnonzero(mask_or_idx) if mask_or_idx.dtype == bool else mask_or_idx
        return int(idx.size), int(tok[idx].sum())
    J['domain'] = {k: dict(zip(('docs', 'tokens'), dt(x))) for k, x in
                   {'A_fasttext': A, 'B_fineweb': B, 'C_modernbert': C,
                    'intersection_ABC': inter3_mask, 'union_ABC': cand_idx}.items()}

    # STEP 3: two-sided constraint over candidate domain (percentiles over A∪B∪C)
    qc = q[cand_idx]; vc = v[cand_idx]
    Q_floor = float(np.percentile(qc, q_floor_pct))
    Vcap90 = float(np.percentile(vc, v_cap_pct))
    Vcap80 = float(np.percentile(vc, 80.0))

    def survivors(vcap):
        keep = (q[cand_idx] >= Q_floor) & (v[cand_idx] <= vcap)
        return cand_idx[keep]
    surv90 = survivors(Vcap90)
    surv80 = survivors(Vcap80)
    J['constraint'] = dict(q_floor_pct=q_floor_pct, Q_floor=Q_floor,
                           v_cap_pct=v_cap_pct, V_cap_90=Vcap90, V_cap_80=Vcap80,
                           survivors_vcap90=dict(zip(('docs', 'tokens'), dt(surv90))),
                           survivors_vcap80=dict(zip(('docs', 'tokens'), dt(surv80))),
                           can_fill_10B_vcap90=bool(int(tok[surv90].sum()) >= TARGET),
                           can_fill_10B_vcap80=bool(int(tok[surv80].sum()) >= TARGET))

    # STEP 4: per-lambda u = q + lambda*sqrt(v), top 10B (analysis only) over survivors @vcap90
    sv = surv90
    sqrtv = np.sqrt(v)
    lam_sel = {}
    lam_meta = {}
    for lam in LAMBDAS:
        u = q[sv] + lam * sqrtv[sv]
        order = sv[np.lexsort((tie[sv], -u.astype(np.float64)))]
        sel, ttok, over, fill = fill_to(order, tok, TARGET)
        rej = order[sel.size:]                       # survivors not selected (order is a perm of sv)
        lam_sel[lam] = sel
        lam_meta[lam] = dict(
            docs=int(sel.size), tokens=int(ttok), filled=bool(fill),
            max_tokens_if_unfilled=(None if fill else int(tok[sv].sum())),
            q_mean=float(q[sel].mean()), v_mean=float(v[sel].mean()),
            q=dict(selected=pctl(q[sel]), rejected=pctl(q[rej])),
            v=dict(selected=pctl(v[sel]), rejected=pctl(v[rej])))
    J['lambda_selections'] = {str(k): lam_meta[k] for k in LAMBDAS}

    lam_masks = {lam: mask_of(s) for lam, s in lam_sel.items()}
    # pairwise Jaccard among lambda selections
    pair = {}
    for a in range(len(LAMBDAS)):
        for b in range(a + 1, len(LAMBDAS)):
            la, lb = LAMBDAS[a], LAMBDAS[b]
            jc, inter, _ = jaccard(lam_masks[la], lam_masks[lb])
            pair[f'{la}_vs_{lb}'] = dict(jaccard=jc, shared=inter)
    J['lambda_pairwise_jaccard'] = pair

    # vs quality-first (10B), vs quality-base (5B), vs lambda0
    qf_mask = mask_of(qf_idx)
    qb_mask = mask_of(qb_idx)
    l0_mask = lam_masks[0.0]
    vs = {}
    for lam in LAMBDAS:
        m = lam_masks[lam]; sz = int(m.sum())
        jq, iq, _ = jaccard(m, qf_mask)
        jb, ib, _ = jaccard(m, qb_mask)
        j0, i0, _ = jaccard(m, l0_mask)
        vs[str(lam)] = dict(
            vs_quality_first=dict(jaccard=jq, inside_pct=100.0 * iq / sz if sz else 0.0),
            vs_quality_base=dict(jaccard=jb, inside_pct=100.0 * ib / sz if sz else 0.0),
            vs_lambda0=dict(jaccard=j0, inside_pct=100.0 * i0 / sz if sz else 0.0))
    J['lambda_vs_baselines'] = vs

    # voters over FULL pool minus val (alive_idx), NOT remaining
    Ap = top10_by_tokens(alive_idx, ft, tok, tie)
    Bp = top10_by_tokens(alive_idx, fw, tok, tie)
    Cp = top10_by_tokens(alive_idx, mb, tok, tie)
    votes = mask_of(Ap).astype(np.int8) + mask_of(Bp).astype(np.int8) + mask_of(Cp).astype(np.int8)
    one_mask = votes == 1; two_mask = votes == 2; three_mask = votes == 3
    voter = {}
    for lam in LAMBDAS:
        m = lam_masks[lam]; sz = int(m.sum())
        jc2, i2, _ = jaccard(m, two_mask)
        voter[str(lam)] = dict(
            inside_one_pct=100.0 * int(np.count_nonzero(m & one_mask)) / sz if sz else 0.0,
            inside_two_pct=100.0 * int(np.count_nonzero(m & two_mask)) / sz if sz else 0.0,
            inside_three_pct=100.0 * int(np.count_nonzero(m & three_mask)) / sz if sz else 0.0,
            jaccard_two_voter=jc2)
    J['voter_analysis'] = dict(
        sizes=dict(one_voter=int(one_mask.sum()), two_voter=int(two_mask.sum()),
                   three_voter=int(three_mask.sum())),
        per_lambda=voter)

    # q/v shift as lambda grows (selected means)
    J['q_v_shift'] = {str(lam): dict(sel_q_mean=lam_meta[lam]['q_mean'],
                                     sel_v_mean=lam_meta[lam]['v_mean']) for lam in LAMBDAS}

    md = render_analysis_md(J, LAMBDAS)
    return dict(json=J, md=md)


def _fmt(x, fmt):
    return 'n/a' if x is None else format(x, fmt)


def render_analysis_md(J, LAMBDAS):
    L = ['# signal-disagreement — ANALYSIS ONLY (no dataset saved this run)', '',
         '_q = mean(3 v2 percentiles); v = variance of the 3 v2 percentiles; '
         'top-fill target = 10B by (tokens-llama2+1)._', '']
    d = J['domain']
    L += ['## STEP 1 — candidate domain A∪B∪C (over REMAINING)',
          '| set | docs | tokens |', '|---|---:|---:|']
    for k in ['A_fasttext', 'B_fineweb', 'C_modernbert', 'intersection_ABC', 'union_ABC']:
        L.append(f'| {k} | {d[k]["docs"]:,} | {d[k]["tokens"]:,} |')
    c = J['constraint']
    L += ['', '## STEP 3 — two-sided constraint (percentiles over A∪B∪C)',
          f'- q_floor = {c["q_floor_pct"]:.0f}th pct of q = {c["Q_floor"]:.5f}',
          f'- V_cap(90th) = {c["V_cap_90"]:.6g}; V_cap(80th) = {c["V_cap_80"]:.6g}',
          f'- survivors @vcap90: {c["survivors_vcap90"]["docs"]:,} docs / '
          f'{c["survivors_vcap90"]["tokens"]:,} tok — can fill 10B: {c["can_fill_10B_vcap90"]}',
          f'- survivors @vcap80: {c["survivors_vcap80"]["docs"]:,} docs / '
          f'{c["survivors_vcap80"]["tokens"]:,} tok — can fill 10B: {c["can_fill_10B_vcap80"]}',
          '', '## STEP 4 — per-lambda u=q+λ·√v top-10B (analysis only, over survivors @vcap90)',
          '| λ | docs | tokens | filled 10B | max tok if unfilled | sel q̄ | sel v̄ |',
          '|---:|---:|---:|:--:|---:|---:|---:|']
    for lam in LAMBDAS:
        m = J['lambda_selections'][str(lam)]
        L.append(f'| {lam} | {m["docs"]:,} | {m["tokens"]:,} | {m["filled"]} | '
                 f'{_fmt(m["max_tokens_if_unfilled"], ",") } | {m["q_mean"]:.4f} | {m["v_mean"]:.5f} |')
    L += ['', '## Pairwise Jaccard among λ selections', '| pair | jaccard | shared docs |', '|---|---:|---:|']
    for k, vv in J['lambda_pairwise_jaccard'].items():
        L.append(f'| {k} | {vv["jaccard"]:.4f} | {vv["shared"]:,} |')
    L += ['', '## λ selection vs baselines',
          '| λ | J vs quality-first | %in QF | J vs quality-base | %in QB | J vs λ0 | %in λ0 |',
          '|---:|---:|---:|---:|---:|---:|---:|']
    for lam in LAMBDAS:
        b = J['lambda_vs_baselines'][str(lam)]
        L.append(f'| {lam} | {b["vs_quality_first"]["jaccard"]:.4f} | {b["vs_quality_first"]["inside_pct"]:.1f}% '
                 f'| {b["vs_quality_base"]["jaccard"]:.4f} | {b["vs_quality_base"]["inside_pct"]:.1f}% '
                 f'| {b["vs_lambda0"]["jaccard"]:.4f} | {b["vs_lambda0"]["inside_pct"]:.1f}% |')
    va = J['voter_analysis']
    L += ['', f'## Voter analysis (top-10% over full pool minus val): one={va["sizes"]["one_voter"]:,}, '
          f'two={va["sizes"]["two_voter"]:,}, three={va["sizes"]["three_voter"]:,}',
          '| λ | %inside 1-voter | %inside 2-voter | %inside 3-voter | J vs 2-voter |',
          '|---:|---:|---:|---:|---:|']
    for lam in LAMBDAS:
        p = va['per_lambda'][str(lam)]
        L.append(f'| {lam} | {p["inside_one_pct"]:.1f}% | {p["inside_two_pct"]:.1f}% | '
                 f'{p["inside_three_pct"]:.1f}% | {p["jaccard_two_voter"]:.4f} |')
    # q/v distributions of selected vs rejected (within survivors), per lambda
    L += ['', '## q/v distributions: selected vs rejected (within survivors @vcap90)',
          '| λ | grp | q p10 | q p50 | q p90 | q̄ | v p10 | v p50 | v p90 | v̄ |',
          '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lam in LAMBDAS:
        m = J['lambda_selections'][str(lam)]
        for grp in ('selected', 'rejected'):
            qd = m['q'][grp]; vd = m['v'][grp]
            L.append(f'| {lam} | {grp} | {_fmt(qd["p10"],".4f")} | {_fmt(qd["p50"],".4f")} | '
                     f'{_fmt(qd["p90"],".4f")} | {_fmt(qd["mean"],".4f")} | {_fmt(vd["p10"],".5f")} | '
                     f'{_fmt(vd["p50"],".5f")} | {_fmt(vd["p90"],".5f")} | {_fmt(vd["mean"],".5f")} |')
    L += ['', '## q/v shift as λ grows (selected means)', '| λ | sel q̄ | sel v̄ |', '|---:|---:|---:|']
    for lam in LAMBDAS:
        s = J['q_v_shift'][str(lam)]
        L.append(f'| {lam} | {s["sel_q_mean"]:.4f} | {s["sel_v_mean"]:.5f} |')
    L += ['', '_No signal-disagreement training dataset saved this run; decide λ + V_cap from the above._']
    return '\n'.join(L)


# ----------------------------------------------------------------------------- overlap / manifests / reports
def overlap_report(save_masks, shared_mask, val_mask):
    names = list(save_masks.keys())
    var_names = [n for n in names if n != 'shared-top-5B']
    out = {'pairwise': {}, 'exclusivity': {}}
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            a, b = var_names[i], var_names[j]
            jc, inter, _ = jaccard(save_masks[a], save_masks[b])
            out['pairwise'][f'{a}__{b}'] = dict(jaccard=jc, shared=inter)
    for n in names:
        out['exclusivity'][f'shared_cap_{n}'] = int(np.count_nonzero(shared_mask & save_masks[n])) if n != 'shared-top-5B' else None
        out['exclusivity'][f'val_cap_{n}'] = int(np.count_nonzero(val_mask & save_masks[n]))
    return out


def write_manifests(shared_idx, shared_tok, shared_over, blocks, rem_tok, alive_n):
    # shared
    _mani(TRAIN / 'shared-top-5B', 'shared-top-5B', shared_idx, shared_tok, shared_over,
          BLOCK_TARGETS['shared-top-5B'], 'fasttext-v2 DESC over (all minus val)',
          pct_base=('available', alive_n))
    for name in VAR_BLOCKS:
        bk = blocks[name]
        extra = {}
        if name == 'diversity-first':
            extra = dict(categories=bk['cats'], topup_docs=bk['topup'])
        _mani(TRAIN / name, name, bk['idx'], bk['tok'], bk['over'], BLOCK_TARGETS[name],
              bk['method'], pct_base=('remaining_tokens', rem_tok), extra=extra)


def _mani(d, name, idx, ttok, over, target, method, pct_base, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    base_label, base_val = pct_base
    m = dict(block=name, target_tokens=int(target), docs=int(idx.size),
             train_tokens_sum=int(ttok), overshoot_tokens=int(over),
             method=method, seed=SEED,
             pct_of_base=dict(base=base_label, value=100.0 * ttok / base_val if base_val else None))
    if extra: m.update(extra)
    (d / '_manifest.json').write_text(json.dumps(m, indent=2))


def write_final_report(shared_idx, shared_tok, shared_over, blocks, rem_tok,
                       rem_docs, overlap, val_idx):
    L = ['# 10B experiment selection — FINAL report', '',
         f'- seed={SEED}; budgets use (tokens-llama2 + 1).',
         f'- targets: shared-top-5B=5B, quality-base=5B, quality-first=10B, wrap=10B, '
         f'rewrite=20B, diversity-first=10B.',
         f'- val: {val_idx.size:,} docs (excluded from every block).',
         f'- REMAINING: {rem_docs:,} docs, {rem_tok:,} tok (= all minus val minus shared-top).', '',
         '## Generated blocks', '| block | target | docs | train_tokens | overshoot | % of base |',
         '|---|---:|---:|---:|---:|---:|']
    L.append(f'| shared-top-5B | {BLOCK_TARGETS["shared-top-5B"]:,} | {shared_idx.size:,} | '
             f'{shared_tok:,} | {shared_over:,} | (of available) |')
    for name in VAR_BLOCKS:
        bk = blocks[name]
        L.append(f'| {name} | {BLOCK_TARGETS[name]:,} | {bk["idx"].size:,} | {bk["tok"]:,} | '
                 f'{bk["over"]:,} | {100.0*bk["tok"]/rem_tok:.2f}% of REMAINING |')
    L += ['', '## Exclusivity (must be 0)', '| check | count |', '|---|---:|']
    for k, vv in overlap['exclusivity'].items():
        if vv is not None:
            L.append(f'| {k} | {vv} |')
    L += ['', '## Pairwise overlap across variable blocks (reported, not enforced)',
          '| pair | jaccard | shared docs |', '|---|---:|---:|']
    for k, vv in overlap['pairwise'].items():
        L.append(f'| {k} | {vv["jaccard"]:.4f} | {vv["shared"]:,} |')
    L += ['', '## diversity-first per-category', '| topic | docs | tokens | quota | filled |',
          '|---|---:|---:|---:|:--:|']
    for c in blocks['diversity-first']['cats']:
        L.append(f'| {c["cat"]} | {c["docs"]:,} | {c["tok"]:,} | {c["quota"]:.0f} | {c["filled"]} |')
    L += ['', f'- diversity-first top-up docs (rounding shortfall): {blocks["diversity-first"]["topup"]:,}']
    TRAIN.mkdir(parents=True, exist_ok=True)
    (TRAIN / 'SELECTION_REPORT.md').write_text('\n'.join(L))


# ----------------------------------------------------------------------------- PASS 2: write saved blocks
_WMASKS = {}    # set as a module global in write_blocks(); forked workers inherit via COW


def _block_out(name, k):
    d = VAL_DIR if name == '__val__' else (TRAIN / name)
    return d / f'part_{k:05d}.parquet'


def _write_one(k):
    p = shard_path(k)
    lo = shard_offset(k); n = shard_rows(k)
    counts = {}
    need_read = False
    plan = {}
    for name, mask in _WMASKS.items():
        sel = mask[lo:lo + n]
        cnt = int(sel.sum())
        counts[name] = cnt
        if cnt == 0:
            continue
        outp = _block_out(name, k)
        if outp.exists():
            try:
                if pq.ParquetFile(outp).metadata.num_rows == cnt:
                    continue
            except Exception:
                pass
        plan[name] = (sel, outp)
        need_read = True
    if need_read:
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
            for nm, c in counts.items(): totals[nm] += c
            done += 1
            if done % 25 == 0: log(f'  PASS2 {done}/200 ({time.time()-t0:.0f}s)')
    log(f'PASS2 OK ({time.time()-t0:.0f}s): written rows per set: {dict(totals)}')
    # cross-check totals == mask popcounts
    for nm, m in masks.items():
        if int(m.sum()) != totals[nm]:
            stop(f'{nm}: written {totals[nm]} != selected {int(m.sum())}')


if __name__ == '__main__':
    main()
