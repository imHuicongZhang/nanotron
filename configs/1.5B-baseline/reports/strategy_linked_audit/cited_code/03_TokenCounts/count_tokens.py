#!/usr/bin/env python
"""Append a Llama-2 token-count column `tokens-llama2` (int32) to the CLEAN
merged corpus 6_merged_clean. Append-only; no selection, no training.

Recipe: tokens = len(tokenizer(text, add_special_tokens=False))  — PURE text
length, no BOS/EOS baked in. empty/None text -> 0. FAST tokenizer only.

Two passes (both parallel across the 200 shards):
  PASS 1 (tokenize):  stream `text` in batches, count tokens, assemble an int32
          array of length N in doc_id order; save 6_merged_clean_flat/tokens_llama2.npy.
          Light memory (~2.1 GB/worker) -> many workers.
  PASS 2 (append):    read each shard, attach its tokens-llama2 slice, temp-file +
          atomic replace, re-open & verify. Heavy memory (full shard) -> fewer workers.

Idempotent, atomic, re-runnable. Operates ONLY on 6_merged_clean (append one col)
and new files under 6_merged_clean_flat/. Does not touch any other dir/column.
"""
from __future__ import annotations

import argparse
import gc
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
BASE  = Path('/scratch/bvandur1/zhuicon1/data_rewrite')
CLEAN = BASE / '6_merged_clean'
FLAT  = BASE / '6_merged_clean_flat'
TOK_NPY = FLAT / 'tokens_llama2.npy'
TOKENIZER_DIR = '/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer'
REPORT = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/03_TokenCounts/count_tokens_report.md')

NSHARDS   = 200
ROWS_FULL = 500_000
LAST_ROWS = 449_162
N_EXPECT  = 99_949_162

NEW_COL = 'tokens-llama2'
BATCH = 2000

_TOK = None     # per-worker tokenizer
_TOKENS: np.ndarray = np.empty(0, dtype=np.int32)   # global, for PASS2 fork-share


def shard_path(i: int) -> Path:
    return CLEAN / f'merged_clean_{i:05d}.parquet'


def shard_rows(i: int) -> int:
    return ROWS_FULL if i < NSHARDS - 1 else LAST_ROWS


def shard_offset(i: int) -> int:
    return i * ROWS_FULL


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def stop(m: str) -> 'NoReturn':
    print(f'\n*** STOP: {m}\n', flush=True)
    sys.exit(2)


def _load_tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    if not tk.is_fast or not hasattr(tk, 'backend_tokenizer'):
        raise RuntimeError(f'tokenizer not FAST (is_fast={tk.is_fast}); refusing slow tokenizer.')
    return tk


def _init_tok_worker():
    pa.set_cpu_count(1)
    global _TOK
    _TOK = _load_tokenizer()


def _init_plain_worker():
    pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- META
def metadata_check() -> int:
    log('META: schema + per-shard row counts...')
    pf0 = pq.ParquetFile(shard_path(0))
    names = list(pf0.schema_arrow.names)
    if 'text' not in names:
        stop('no `text` column in 6_merged_clean.')
    if NEW_COL in names:
        log(f'META note: `{NEW_COL}` already present in shard 0 -> will OVERWRITE cleanly (idempotent).')
    tmps = sorted(str(p) for p in CLEAN.glob('*.tmp'))
    counts = []
    for i in range(NSHARDS):
        p = shard_path(i)
        if not p.exists():
            stop(f'missing shard {i:05d}')
        counts.append(pq.ParquetFile(p).metadata.num_rows)
    bad = [(i, c) for i, c in enumerate(counts) if c != shard_rows(i)]
    total = sum(counts)
    if tmps or bad or total != N_EXPECT:
        if tmps:
            print(f'  .tmp leftovers: {tmps}', flush=True)
        for i, c in bad:
            print(f'  shard {i:05d}: {c} rows (expected {shard_rows(i)})', flush=True)
        stop(f'row-count precondition failed (total={total:,}, expected {N_EXPECT:,}).')
    log(f'META OK: 200 shards (199x500k + {LAST_ROWS:,}), total {total:,}.')
    return total


def orig_cols() -> list[str]:
    names = list(pq.ParquetFile(shard_path(0)).schema_arrow.names)
    return [n for n in names if n != NEW_COL]


# ----------------------------------------------------------------------------- PASS 1: tokenize
def _tokenize_shard(i: int):
    p = shard_path(i)
    n = shard_rows(i)
    lengths = np.empty(n, dtype=np.int32)
    char_sum = 0
    pos = 0
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=BATCH, columns=['text']):
        texts = [s if s else '' for s in batch.column('text').to_pylist()]
        char_sum += sum(len(s) for s in texts)
        enc = _TOK(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
        for ids in enc:
            lengths[pos] = len(ids)
            pos += 1
    if pos != n:
        raise RuntimeError(f'shard {i}: tokenized {pos} != {n} rows')
    if lengths.min() < 0:
        raise RuntimeError(f'shard {i}: negative token count')
    return i, lengths, int(char_sum)


def pass1_tokenize(n_actual: int, workers: int, force: bool) -> tuple[np.ndarray, int]:
    FLAT.mkdir(parents=True, exist_ok=True)
    if TOK_NPY.exists() and not force:
        a = np.load(TOK_NPY, mmap_mode='r')
        if a.shape == (n_actual,) and a.dtype == np.int32:
            log(f'PASS1: {TOK_NPY.name} present ({n_actual:,} int32), skipping tokenize.')
            return np.asarray(np.load(TOK_NPY)), -1   # char_sum unknown on resume
    log(f'PASS1: tokenizing {n_actual:,} docs across 200 shards with {workers} workers...')
    tokens = np.empty(n_actual, dtype=np.int32)
    char_total = 0
    seen = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_tok_worker) as ex:
        futs = [ex.submit(_tokenize_shard, i) for i in range(NSHARDS)]
        for fut in as_completed(futs):
            i, lengths, csum = fut.result()
            lo = shard_offset(i)
            tokens[lo:lo + lengths.size] = lengths
            char_total += csum
            seen += 1
            if seen % 25 == 0:
                log(f'  PASS1 {seen}/200 ({time.time()-t0:.0f}s)')
    # atomic save (file handle so np.save does not append a second .npy)
    tmp = TOK_NPY.with_suffix(TOK_NPY.suffix + '.tmp')
    with open(tmp, 'wb') as fh:
        np.save(fh, tokens)
    os.replace(tmp, TOK_NPY)
    log(f'PASS1 OK ({time.time()-t0:.0f}s): saved {TOK_NPY.name}; total_chars={char_total:,}')
    return tokens, char_total


# ----------------------------------------------------------------------------- PASS 2: append
def _append_shard(i: int, ORIG: tuple, EXPECT: tuple):
    p = shard_path(i)
    tmp = p.with_suffix(p.suffix + '.tmp')
    lo = shard_offset(i)
    n = shard_rows(i)
    sl = np.ascontiguousarray(_TOKENS[lo:lo + n]).astype(np.int32)

    pf = pq.ParquetFile(p)
    names = list(pf.schema_arrow.names)
    had = NEW_COL in names

    if names == list(EXPECT) and pf.metadata.num_rows == n:
        cur = pq.read_table(p, columns=[NEW_COL], use_threads=False).column(0)
        if cur.null_count == 0 and np.array_equal(cur.to_numpy(zero_copy_only=False), sl):
            return dict(i=i, status='skip-verified', nrows=n, mn=int(sl.min()), mx=int(sl.max()),
                        tsum=int(sl.sum()))

    told = pq.read_table(p, use_threads=False)
    if told.num_rows != n:
        raise RuntimeError(f'shard {i}: {told.num_rows} rows')
    base = told
    if NEW_COL in base.column_names:
        base = base.drop_columns([NEW_COL])
    if list(base.column_names) != list(ORIG):
        raise RuntimeError(f'shard {i}: unexpected base cols {base.column_names}')
    new = base.append_column(NEW_COL, pa.array(sl, type=pa.int32()))
    pq.write_table(new, tmp, compression='zstd')
    os.replace(tmp, p)

    pfn = pq.ParquetFile(p)
    if list(pfn.schema_arrow.names) != list(EXPECT):
        raise RuntimeError(f'shard {i}: post-write schema {pfn.schema_arrow.names}')
    if pfn.metadata.num_rows != n:
        raise RuntimeError(f'shard {i}: post-write rows {pfn.metadata.num_rows}')
    col = pq.read_table(p, columns=[NEW_COL], use_threads=False).column(0)
    if col.null_count != 0 or not np.array_equal(col.to_numpy(zero_copy_only=False), sl) or sl.min() < 0:
        raise RuntimeError(f'shard {i}: {NEW_COL} mismatch/null/negative')
    del col
    for c in ORIG:
        rc = pq.read_table(p, columns=[c], use_threads=False).column(0)
        if not rc.equals(told.column(c)):
            raise RuntimeError(f'shard {i}: original col {c} changed')
        del rc
    return dict(i=i, status=('overwritten' if had else 'written'), nrows=n,
                mn=int(sl.min()), mx=int(sl.max()), tsum=int(sl.sum()))


def pass2_append(ORIG: list, EXPECT: list, workers: int) -> list[dict]:
    log(f'PASS2: appending `{NEW_COL}` to 200 shards with {workers} workers...')
    res = []
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_plain_worker) as ex:
        futs = {ex.submit(_append_shard, i, tuple(ORIG), tuple(EXPECT)): i for i in range(NSHARDS)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res.append(fut.result())
            except Exception as e:    # noqa: BLE001
                stop(f'PASS2 shard {i:05d} failed: {e!r}')
            done += 1
            if done % 25 == 0:
                log(f'  PASS2 {done}/200 ({time.time()-t0:.0f}s)')
    res.sort(key=lambda d: d['i'])
    log(f'PASS2 OK ({time.time()-t0:.0f}s): {dict(Counter(r["status"] for r in res))}')
    return res


# ----------------------------------------------------------------------------- report
def special_token_manifest() -> dict:
    tk = _load_tokenizer()
    probe = 'The quick brown fox.'
    f = tk(probe, add_special_tokens=False)['input_ids']
    t = tk(probe, add_special_tokens=True)['input_ids']
    # Llama-2 adds specials only at the ends; count how many are leading vs trailing.
    n_added = len(t) - len(f)
    specials = set(tk.all_special_ids)
    lead = 0
    while lead < n_added and t[lead] in specials:
        lead += 1
    trail = n_added - lead
    return dict(klass=type(tk).__name__, is_fast=tk.is_fast,
                bos_token=tk.bos_token, bos_id=tk.bos_token_id,
                eos_token=tk.eos_token, eos_id=tk.eos_token_id,
                add_bos=getattr(tk, 'add_bos_token', None), add_eos=getattr(tk, 'add_eos_token', None),
                n_specials_added=len(t) - len(f), leading=lead, trailing=trail)


def spot_examples(tokens: np.ndarray) -> list[dict]:
    order = np.argsort(tokens, kind='stable')
    nz = order[tokens[order] > 0]
    shorts = nz[:3] if nz.size >= 3 else nz
    longs = order[-3:][::-1]
    picks = list(shorts) + list(longs)
    by_shard = {}
    for g in picks:
        by_shard.setdefault(int(g) // ROWS_FULL, []).append(int(g))
    out = {}
    for k, gs in by_shard.items():
        t = pq.read_table(shard_path(k), columns=['text', 'doc_id'], use_threads=False)
        for g in gs:
            r = g - shard_offset(k)
            s = t.column('text')[r].as_py() or ''
            out[g] = dict(doc_id=int(t.column('doc_id')[r].as_py()), chars=len(s),
                          tokens=int(tokens[g]), kind=('short' if g in shorts else 'long'))
    return [out[int(g)] for g in picks]


def write_report(tokens, char_total, results, manifest, spots, worker_info):
    total = int(tokens.sum(dtype=np.int64))
    n = tokens.size
    pct = lambda q: float(np.percentile(tokens, q))
    L = ['# tokens-llama2 — manifest & validation', '',
         f'- SLURM job {os.environ.get("SLURM_JOB_ID","?")}. {worker_info}',
         f'- Dataset: `{CLEAN}` (clean). N docs = **{n:,}** (asserted == {N_EXPECT:,}).',
         f'- New column **`{NEW_COL}`** (int32, append-only). Flat: `{TOK_NPY}`.',
         f'- Tokenizer: `{TOKENIZER_DIR}` — class `{manifest["klass"]}`, is_fast={manifest["is_fast"]}.',
         '- Recipe: `len(tokenizer(text, add_special_tokens=False))` — PURE text length (no BOS/EOS baked in).',
         '', '## Special tokens Llama-2 WOULD add (for later packing offset)',
         f'- BOS: `{manifest["bos_token"]}` (id {manifest["bos_id"]}), EOS: `{manifest["eos_token"]}` '
         f'(id {manifest["eos_id"]}).',
         f'- add_bos_token={manifest["add_bos"]}, add_eos_token={manifest["add_eos"]}.',
         f'- With add_special_tokens=True: **+{manifest["n_specials_added"]}** token(s) per doc '
         f'({manifest["leading"]} leading BOS, {manifest["trailing"]} trailing EOS). '
         'So packed/per-doc training length = `tokens-llama2 + ' f'{manifest["n_specials_added"]}` if BOS/EOS added.',
         '', '## Corpus token statistics (pure-text, add_special_tokens=False)',
         f'- total docs: **{n:,}**',
         f'- total tokens: **{total:,}**',
         f'- mean tokens/doc: **{total / n:.2f}**   median: **{pct(50):.0f}**',
         f'- p10 / p50 / p90 / p99: {pct(10):.0f} / {pct(50):.0f} / {pct(90):.0f} / {pct(99):.0f}',
         f'- min / max: {int(tokens.min())} / {int(tokens.max()):,}',
         f'- docs with 0 tokens (empty text): {int((tokens == 0).sum()):,}', '']
    if char_total >= 0:
        L += [f'- total chars: {char_total:,}; **tokens/char = {total / char_total:.4f}** '
              f'(chars/token = {char_total / total:.3f}). Use this to rescale char-based budgets to real Llama-2 tokens.', '']
    else:
        L += ['- total chars: (PASS1 skipped on resume — re-run with --force-rank to recompute the ratio).', '']
    L += ['## Budget rescale helper',
          f'- A 5B-token budget  ≈ {5e9 / (total / n):,.0f} docs at this mean.',
          f'- A 10B-token budget ≈ {10e9 / (total / n):,.0f} docs at this mean.', '',
          '## Spot-checks (3 short, 3 long)',
          '| kind | doc_id | chars | tokens-llama2 | chars/token |', '|---|---:|---:|---:|---:|']
    for s in spots:
        cpt = s['chars'] / s['tokens'] if s['tokens'] else 0.0
        L.append(f'| {s["kind"]} | {s["doc_id"]:,} | {s["chars"]:,} | {s["tokens"]:,} | {cpt:.2f} |')
    nn = all(r['nrows'] == shard_rows(r['i']) for r in results)
    c = Counter(r['status'] for r in results)
    L += ['', '## Per-shard verification (all 200)',
          f'- shards: **{len(results)}/200**, status {dict(c)}; row counts correct: '
          f'**{"PASS" if nn else "FAIL"}**; new col non-null & >=0; schema == orig + `{NEW_COL}`.',
          '- all original columns verified byte/value-identical after rewrite (`Array.equals` on re-read).', '',
          '| shard | rows | tok min | tok max | tok sum |', '|---:|---:|---:|---:|---:|']
    for r in results[:5] + results[-5:]:
        L.append(f'| {r["i"]:05d} | {r["nrows"]:,} | {r["mn"]} | {r["mx"]:,} | {r["tsum"]:,} |')
    REPORT.write_text('\n'.join(L))
    log(f'REPORT -> {REPORT}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-rank', action='store_true', help='recompute tokens npy even if present')
    ap.add_argument('--tok-mem-per-worker-gb', type=float, default=3.0)
    ap.add_argument('--append-mem-per-worker-gb', type=float, default=7.0)
    args = ap.parse_args()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8))
    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_gb = (mem_mb / 1024.0) if mem_mb else 256.0
    tok_workers = max(1, min(cpus, int((mem_gb - 24) // args.tok_mem_per_worker_gb)))
    app_workers = max(1, min(cpus, int((mem_gb - 16) // args.append_mem_per_worker_gb)))
    worker_info = (f'cpus={cpus}, mem={mem_gb:.0f}G -> tokenize {tok_workers} workers '
                   f'(~{args.tok_mem_per_worker_gb:g}G ea), append {app_workers} workers '
                   f'(~{args.append_mem_per_worker_gb:g}G ea).')
    log(f'START: {worker_info}')

    t0 = time.time()
    n_actual = metadata_check()
    assert n_actual == N_EXPECT
    ORIG = orig_cols()
    EXPECT = ORIG + [NEW_COL]
    log(f'orig cols ({len(ORIG)}): {ORIG}')

    tokens, char_total = pass1_tokenize(n_actual, tok_workers, force=args.force_rank)
    if tokens.shape != (n_actual,):
        stop(f'tokens length {tokens.shape} != {n_actual}')

    manifest = special_token_manifest()
    log(f'special tokens: +{manifest["n_specials_added"]} (lead {manifest["leading"]} / trail {manifest["trailing"]})')

    global _TOKENS
    _TOKENS = tokens
    results = pass2_append(ORIG, EXPECT, app_workers)

    if sum(r['tsum'] for r in results) != int(tokens.sum(dtype=np.int64)):
        stop('per-shard token sums != global token sum.')

    spots = spot_examples(tokens)
    write_report(tokens, char_total, results, manifest, spots, worker_info)
    log(f'ALL DONE in {time.time()-t0:.0f}s. `{NEW_COL}` on 200 clean shards; '
        f'total tokens={int(tokens.sum(dtype=np.int64)):,}; report at {REPORT}')


if __name__ == '__main__':
    main()
