#!/usr/bin/env python
"""STEP 2 — assemble ~5B rewritten training tokens per setting.

For each of signal-disagreement-lambda05 and quality-first:
  * Use ALL status==2 Wikipedia-style rewrites (rewritten/, after Step 1 prefix cleanup).
  * If that is < 5B (expected), top up from the distill pass (distill/), quality-sorted, until
    cumulative (rewritten_tokens + 1) reaches 5B. Distill is NOT de-duplicated against wiki —
    the same source rewritten by two prompts is two valid training examples (intentional).
  * If even wiki+distill < 5B, report the shortfall prominently and STOP (no padding with
    status 0/1 or original text).

QUALITY-SORT RULE — always on the ORIGINAL document's precomputed columns (never score the
rewritten text), read straight from the shard (no join):
  * quality-first:                sort key = `fasttext-ranking-v2`, DESC.
  * signal-disagreement-lambda05: u = q + 0.5*sqrt(v), DESC, where (in float32)
        s = stack([fasttext-ranking-v2, fineweb-edu-ranking-v2, modernbert-ranking-v2])
        q = s.mean(axis=0); v = s.var(axis=0)  # POPULATION variance, ddof=0
        u = q + np.float32(0.5)*np.sqrt(v)
  Plain stable sort on the key DESC (no seeded tie-break — only quality ordering matters).

Budget length = (rewritten_tokens + 1) for the REWRITTEN set. tokens-llama2 (source length)
is kept strictly separate.

Output -> data_rewrite/pretrain/<setting>/rewritten/{wiki,distill}_NNNNN.parquet (atomic),
plus _assembly_manifest.json. Also writes _step2_summary.json for the final report. CPU-only,
read-only on the inputs (Step 1 already finalized them).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pp_io import atomic_write_table, paired_wiki_status

# ----------------------------------------------------------------------------- paths / constants
BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B')
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

# SETTINGS defaults to the original two but can be overridden via the PP_SETTINGS env var
# (comma-separated) so run_all.sh can target new settings without editing the list here.
SETTINGS = (os.environ['PP_SETTINGS'].split(',') if os.environ.get('PP_SETTINGS')
            else ['signal-disagreement-lambda05', 'quality-first'])
NSHARDS = 200
TARGET = 5_000_000_000          # 5B (rewritten_tokens + 1)

# signal-disagreement λ per setting. Explicit map — the folder-name suffix is ambiguous
# ('lambda05'->0.5 but 'lambda15'->1.5). Any setting listed here uses sort key u=q+λ·√var3.
LAMBDA_BY_SETTING = {
    'signal-disagreement-lambda0':  0.0,
    'signal-disagreement-lambda05': 0.5,
    'signal-disagreement-lambda1':  1.0,
    'signal-disagreement-lambda15': 1.5,
    'signal-disagreement-lambda2':  2.0,
    'signal-disagreement-lambda3':  3.0,
}


def is_signal(setting):
    return setting in LAMBDA_BY_SETTING

V2_FT, V2_FW, V2_MB = 'fasttext-ranking-v2', 'fineweb-edu-ranking-v2', 'modernbert-ranking-v2'

# columns carried into the assembled rewritten shards (keep-most; original `text` dropped,
# `rewritten` -> `text`); `source_prompt` and (signal only) `u_score` are appended.
KEEP_COLS = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2',
             'status', V2_FT, V2_FW, V2_MB, 'url', 'metadata', 'topic']


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def in_path(setting, subdir, k): return BASE / setting / subdir / f'part_{k:05d}.parquet'
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- quality-sort key
def sort_key(setting, ft, fw, mb):
    """Return the per-doc sort score (DESC). float32 throughout."""
    if setting == 'quality-first':
        return ft.astype(np.float32)
    # signal-disagreement-lambdaX: u = q + λ*sqrt(var3), population variance (ddof=0)
    lam = LAMBDA_BY_SETTING[setting]
    s = np.stack([ft, fw, mb]).astype(np.float32)
    q = s.mean(axis=0)
    v = s.var(axis=0)                                   # ddof=0 (np.var default)
    return (q + np.float32(lam) * np.sqrt(v)).astype(np.float32)


def sort_key_label(setting):
    if setting == 'quality-first':
        return 'fasttext-ranking-v2'
    return f'u=q+{LAMBDA_BY_SETTING[setting]}*sqrt(var3), ddof=0, float32'


# ----------------------------------------------------------------------------- numeric pass
def _read_numeric(job):
    setting, subdir, k = job
    t = pq.read_table(in_path(setting, subdir, k),
                      columns=['doc_id', 'status', 'rewritten_tokens',
                               V2_FT, V2_FW, V2_MB, 'tokens-llama2'],
                      use_threads=False)
    return dict(
        k=k, n=t.num_rows,
        doc_id=t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64),
        status=t.column('status').to_numpy(zero_copy_only=False).astype(np.int8),
        rtok=t.column('rewritten_tokens').to_numpy(zero_copy_only=False).astype(np.int64),
        ft=t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32),
        fw=t.column(V2_FW).to_numpy(zero_copy_only=False).astype(np.float32),
        mb=t.column(V2_MB).to_numpy(zero_copy_only=False).astype(np.float32),
        llama=t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64))


def collect_numeric(setting, subdir, workers):
    res = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_read_numeric, (setting, subdir, k)): k for k in range(NSHARDS)}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'{setting}/{subdir} numeric shard {k:05d} failed: {e!r}')
            res[r['k']] = r
            done += 1
            if done % 50 == 0:
                log(f'  numeric {setting}/{subdir}: {done}/200 ({time.time()-t0:.0f}s)')
    shards = [res[k] for k in range(NSHARDS)]
    rows = np.array([s['n'] for s in shards], dtype=np.int64)
    offs = np.zeros(NSHARDS + 1, dtype=np.int64)
    offs[1:] = np.cumsum(rows)
    cat = lambda key: np.concatenate([s[key] for s in shards])  # noqa: E731
    out = dict(offs=offs, doc_id=cat('doc_id'), status=cat('status'), rtok=cat('rtok'),
               ft=cat('ft'), fw=cat('fw'), mb=cat('mb'), llama=cat('llama'))
    out['key'] = sort_key(setting, out['ft'], out['fw'], out['mb'])
    out['len'] = out['rtok'] + 1                                 # train length (+1 BOS)
    return out


# ----------------------------------------------------------------------------- fill
def fill_desc(idx, key, length, target):
    """Order `idx` by key DESC (stable), cumulative-fill `length` until >= target (last whole).
    Returns (selected_idx, total_len, filled_bool)."""
    if idx.size == 0:
        return idx, 0, False
    order = idx[np.argsort(-key[idx], kind='stable')]           # stable DESC
    c = np.cumsum(length[order])
    if c[-1] < target:
        return order, int(c[-1]), False
    cut = int(np.searchsorted(c, target, side='left'))
    return order[:cut + 1], int(c[cut]), True


# ----------------------------------------------------------------------------- write pass
_WRITE = {}   # set before fork: {'setting','keep':{'wiki':mask,'distill':mask},'offs':...}


def _write_shard(job):
    keep_key, in_subdir, k, source_prompt, outprefix = job
    setting = _WRITE['setting']
    offs = _WRITE['offs']
    mask = _WRITE['keep'][keep_key][offs[k]:offs[k + 1]]
    cnt = int(mask.sum())
    if cnt == 0:
        return (keep_key, k, 0)
    t = pq.read_table(in_path(setting, in_subdir, k), use_threads=False)   # 'rewritten'/'distill'
    sub = t.filter(pa.array(mask))

    out = sub.select(KEEP_COLS)
    names = out.column_names
    names[names.index('rewritten')] = 'text'                    # rewrite is the training text
    out = out.rename_columns(names)
    out = out.append_column('source_prompt',
                            pa.array([source_prompt] * cnt, type=pa.large_string()))
    if is_signal(setting):
        ft = sub.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32)
        fw = sub.column(V2_FW).to_numpy(zero_copy_only=False).astype(np.float32)
        mb = sub.column(V2_MB).to_numpy(zero_copy_only=False).astype(np.float32)
        u = sort_key(setting, ft, fw, mb)
        out = out.append_column('u_score', pa.array(u, type=pa.float32()))

    outp = PRETRAIN / setting / 'rewritten' / f'{outprefix}_{k:05d}.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(out, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'{outprefix} shard {k}: rowcount mismatch')
    return (keep_key, k, cnt)


def write_blocks(setting, keep_w, keep_d, offs, workers):
    global _WRITE
    _WRITE = dict(setting=setting, offs=offs, keep={'wiki': keep_w, 'distill': keep_d})
    # job = (keep_key, input_subdir, k, source_prompt, output_prefix)
    jobs = ([('wiki', 'rewritten', k, 'wikipedia', 'wiki') for k in range(NSHARDS)]
            + [('distill', 'distill', k, 'distill', 'distill') for k in range(NSHARDS)])
    written = {'wiki': 0, 'distill': 0}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_shard, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                keep_key, k, cnt = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'write {j[4]} shard {j[2]:05d} failed: {e!r}')
            written[keep_key] += cnt
            done += 1
            if done % 50 == 0:
                log(f'  write {setting}: {done}/{len(jobs)} ({time.time()-t0:.0f}s)')
    if written['wiki'] != int(keep_w.sum()) or written['distill'] != int(keep_d.sum()):
        stop(f'{setting}: written {written} != selected '
             f'(wiki={int(keep_w.sum())}, distill={int(keep_d.sum())})')
    return written


# ----------------------------------------------------------------------------- cross-pass coverage
def cross_pass(wiki, distill):
    """wiki & distill cover the SAME doc_ids. Align wiki status to distill order via
    paired_wiki_status (asserts equality; explicit doc_id-join fallback, never silent)."""
    w = paired_wiki_status(wiki, distill)
    d, llama = distill['status'], distill['llama']

    def ct(mask): return int(mask.sum()), int(llama[mask].sum())
    s0_both_n, s0_both_tok = ct((w == 0) & (d == 0))
    s1_both_n, s1_both_tok = ct((w == 1) & (d == 1))
    recovered_n, _ = ct((w == 1) & (d == 2))
    s2w_not_d_n, _ = ct((w == 2) & ((d == 0) | (d == 1)))
    any_s2 = (w == 2) | (d == 2)
    unique_any_s2_n, covered_tok = ct(any_s2)
    total_selection_tok = int(llama.sum())
    return dict(status_0_both=s0_both_n, status_0_both_tokens=s0_both_tok,
                status_1_both=s1_both_n, status_1_both_tokens=s1_both_tok,
                recovered_by_distill=recovered_n,
                status_2_wiki_not_distill=s2w_not_d_n,
                unique_docs_with_any_status2=unique_any_s2_n,
                source_tokens_covered=covered_tok,
                total_selection_tokens=total_selection_tok,
                coverage_frac=round(covered_tok / total_selection_tok, 4)
                if total_selection_tok else 0.0)


# ----------------------------------------------------------------------------- per-setting driver
def assemble(setting, workers):
    log(f'=== {setting}: numeric pass ===')
    wiki = collect_numeric(setting, 'rewritten', workers)
    distill = collect_numeric(setting, 'distill', workers)

    s2_w = np.flatnonzero(wiki['status'] == 2)
    wiki_tokens = int(wiki['len'][s2_w].sum())
    gap = TARGET - wiki_tokens
    log(f'  wiki status2: {s2_w.size:,} docs, {wiki_tokens:,} tok; gap to 5B = {gap:,}')

    keep_w = np.zeros(wiki['doc_id'].size, dtype=bool)
    keep_d = np.zeros(distill['doc_id'].size, dtype=bool)
    shortfall = False

    if wiki_tokens >= TARGET:
        sel_w, tok_wiki, _ = fill_desc(s2_w, wiki['key'], wiki['len'], TARGET)
        keep_w[sel_w] = True
        n_wiki, n_distill, tok_distill = sel_w.size, 0, 0
        log(f'  wiki alone fills 5B -> {sel_w.size:,} docs, {tok_wiki:,} tok (no distill)')
    else:
        keep_w[s2_w] = True
        tok_wiki, n_wiki = wiki_tokens, int(s2_w.size)
        s2_d = np.flatnonzero(distill['status'] == 2)
        sel_d, tok_distill, filled = fill_desc(s2_d, distill['key'], distill['len'], gap)
        keep_d[sel_d] = True
        n_distill = sel_d.size
        if not filled:
            shortfall = True
            log(f'  *** SHORTFALL: distill exhausted at {tok_distill:,} tok '
                f'(needed {gap:,}); total {tok_wiki+tok_distill:,} < 5B ***')
        else:
            log(f'  distill top-up: {sel_d.size:,} docs, {tok_distill:,} tok (filled={filled})')

    total = tok_wiki + tok_distill
    overshoot = total - TARGET
    dual = int(np.intersect1d(wiki['doc_id'][keep_w], distill['doc_id'][keep_d]).size)

    cov = cross_pass(wiki, distill)
    log(f'  cross-pass: any-status2 docs={cov["unique_docs_with_any_status2"]:,}; '
        f'source coverage {cov["coverage_frac"]*100:.2f}% '
        f'({cov["source_tokens_covered"]:,}/{cov["total_selection_tokens"]:,})')

    log(f'  WRITING {setting}: wiki={int(keep_w.sum()):,} distill={int(keep_d.sum()):,} rows')
    write_blocks(setting, keep_w, keep_d, wiki['offs'], workers)

    manifest = {
        'setting': setting,
        'lambda': LAMBDA_BY_SETTING[setting] if is_signal(setting) else None,
        'total_docs': int(keep_w.sum() + keep_d.sum()),
        'total_tokens_plus_bos': int(total),
        'docs_from_wikipedia': int(n_wiki), 'tokens_from_wikipedia': int(tok_wiki),
        'docs_from_distill': int(n_distill), 'tokens_from_distill': int(tok_distill),
        'dual_rewrite_doc_ids': dual,
        'gap_before_distill': int(TARGET - wiki_tokens), 'overshoot': int(overshoot),
        'status_0_both_passes': cov['status_0_both'],
        'status_1_both_passes': cov['status_1_both'],
        'recovered_by_distill': cov['recovered_by_distill'],
        'unique_docs_with_any_status2': cov['unique_docs_with_any_status2'],
        'sort_key': sort_key_label(setting),
        'shortfall': shortfall,
    }
    (PRETRAIN / setting).mkdir(parents=True, exist_ok=True)
    (PRETRAIN / setting / '_assembly_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary = dict(manifest)
    summary.update(coverage=cov, target=TARGET)
    return setting, summary, shortfall


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP2 START: settings={SETTINGS} target={TARGET:,} workers={workers}')

    summaries = {}
    any_short = False
    for setting in SETTINGS:
        _, summary, short = assemble(setting, workers)
        summaries[setting] = summary
        any_short = any_short or short

    (HERE / '_step2_summary.json').write_text(json.dumps(summaries, indent=2))
    log('STEP2 DONE; wrote _step2_summary.json')
    if any_short:
        stop('one or more settings fell SHORT of 5B — see shortfall above; not padded.')


if __name__ == '__main__':
    main()
