#!/usr/bin/env python
"""STEP 3 — mix shared-top-5B (original text) with the 5B rewritten set, shuffle, report.

Each arm trains on: shared-top-5B (ORIGINAL text) + the Step-2 rewritten 5B = ~10B tokens.
For each of signal-disagreement-lambda05 and quality-first:

  3a. Physically COPY the shared-top-5B shards into pretrain/<setting>/shared-top-5B/ (copy,
      not symlink), adding source_prompt="original". `text` stays the original text.
  3b. Document-level RANDOM SHUFFLE (seed=42) of (shared-top + rewritten), written to
      pretrain/<setting>/shuffled/ as ~500k-row parquet shards. This interleaves the two
      sources so training never sees all of one then all of the other.
  3c. Verify ZERO doc_id overlap between shared-top-5B and the ORIGINAL source docs of the
      rewritten 5B (dedup the dual-rewrite rows by doc_id). Should be 0 by construction.
  3d. Combined budget: shared_top (tokens-llama2 + 1) + rewritten (rewritten_tokens + 1) ~ 10B.
  3e. Write _pretrain_manifest.json.

Finally, assemble postprocess_report.md from the Step 1/2/3 summaries. CPU-only.
Shuffle uses pp_io.bucketed_shuffle (memory-bounded two-pass; safe under the 256 GB cpu cap).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pp_io import atomic_write_table, bucketed_shuffle, stream_shuffle_stats

# ----------------------------------------------------------------------------- paths / constants
BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B')
SHARED_SRC = BASE / 'shared-top-5B'
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

# SETTINGS defaults to the original two but can be overridden via the PP_SETTINGS env var
# (comma-separated) so run_all.sh can target new settings without editing the list here.
SETTINGS = (os.environ['PP_SETTINGS'].split(',') if os.environ.get('PP_SETTINGS')
            else ['signal-disagreement-lambda05', 'quality-first'])
SHUFFLE_SEED = 42
ROWS_PER_SHARD = 500_000
TARGET_10B = 10_000_000_000

# unified shuffle schema (kept deliberately small; `text` is the heavy column)
UNIFIED = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()),
                     ('text', pa.large_string()), ('source_prompt', pa.large_string()),
                     ('train_tokens', pa.int32())])


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- 3a copy shared-top
def _copy_shared_shard(job):
    setting, src = job
    t = pq.read_table(src, use_threads=False)
    n = t.num_rows
    t = t.append_column('source_prompt', pa.array(['original'] * n, type=pa.large_string()))
    outp = PRETRAIN / setting / 'shared-top-5B' / Path(src).name
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(t, outp)
    if pq.ParquetFile(outp).metadata.num_rows != n:
        raise RuntimeError(f'shared-top copy {Path(src).name}: rowcount mismatch')
    return n


def copy_shared_top(setting, workers):
    srcs = sorted(glob.glob(str(SHARED_SRC / 'part_*.parquet')))
    if not srcs:
        stop(f'no shared-top shards under {SHARED_SRC}')
    total = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_copy_shared_shard, (setting, s)): s for s in srcs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                total += fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'shared-top copy {Path(s).name} failed: {e!r}')
    log(f'  3a copied {len(srcs)} shared-top shards -> {total:,} rows')
    return len(srcs), total


# ----------------------------------------------------------------------------- unified loader
def load_unified(path, tok_col):
    """Read one parquet shard into the UNIFIED schema. `text` is taken from the `text` column
    (original for shared-top copies; the rewrite for rewritten shards, already renamed)."""
    t = pq.read_table(path, columns=['doc_id', 'orig_doc_id', 'text', 'source_prompt', tok_col],
                      use_threads=False)
    return pa.table({
        'doc_id': t.column('doc_id').cast(pa.int64()),
        'orig_doc_id': t.column('orig_doc_id').cast(pa.int64()),
        'text': t.column('text').cast(pa.large_string()),
        'source_prompt': t.column('source_prompt').cast(pa.large_string()),
        'train_tokens': t.column(tok_col).cast(pa.int32()),
    }, schema=UNIFIED)


# ----------------------------------------------------------------------------- 3b shuffle
def shuffle_and_write(setting):
    """Memory-bounded two-pass bucketed shuffle (pp_io.bucketed_shuffle); never holds the full
    corpus + a copy. Own default_rng(42), independent of Step 2's selection."""
    shared_dir = PRETRAIN / setting / 'shared-top-5B'
    rewritten_dir = PRETRAIN / setting / 'rewritten'
    specs = [(p, 'tokens-llama2')
             for p in sorted(glob.glob(str(shared_dir / 'part_*.parquet')))]
    rew_files = sorted(glob.glob(str(rewritten_dir / 'wiki_*.parquet'))
                       + glob.glob(str(rewritten_dir / 'distill_*.parquet')))
    if not rew_files:
        stop(f'no rewritten shards under {rewritten_dir} (run Step 2 first)')
    specs += [(p, 'rewritten_tokens') for p in rew_files]

    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_bytes = mem_mb * 1024 * 1024 if mem_mb else None
    total_rows, n_shards, B = bucketed_shuffle(
        specs, lambda s: load_unified(s[0], s[1]),
        PRETRAIN / setting / 'shuffled', PRETRAIN / setting / '_shuffle_tmp',
        seed=SHUFFLE_SEED, rows_per_shard=ROWS_PER_SHARD, mem_bytes=mem_bytes, log=log)
    log(f'  3b shuffled {total_rows:,} rows into {n_shards} shards (B={B}, seed={SHUFFLE_SEED})')
    return total_rows, n_shards


# ----------------------------------------------------------------------------- per-setting driver
def mix(setting, workers, assembly):
    log(f'=== {setting}: Step 3 ===')
    shared_docs, shared_rows = copy_shared_top(setting, workers)
    total_rows, n_shards = shuffle_and_write(setting)

    # 3c-3d: stream the final shards' light columns (no full-corpus table)
    st = stream_shuffle_stats(PRETRAIN / setting / 'shuffled')
    overlap = st['overlap']
    log(f'  3c doc_id overlap (shared-top ∩ rewritten-source) = {overlap} (expect 0)')
    shared_top_tokens = st['shared_top_tokens']; rewritten_tokens = st['rewritten_tokens']
    total = st['total_tokens']
    log(f'  3d shared_top={shared_top_tokens:,} + rewritten={rewritten_tokens:,} '
        f'= {total:,} (target 10B, overshoot {total-TARGET_10B:,})')

    manifest = {
        'setting': setting,
        'shared_top_docs': st['shared_top_docs'], 'shared_top_tokens': shared_top_tokens,
        'rewritten_docs': st['rewritten_docs'], 'rewritten_tokens': rewritten_tokens,
        'rewritten_from_wikipedia': assembly['docs_from_wikipedia'],
        'rewritten_from_distill': assembly['docs_from_distill'],
        'dual_rewrite_docs': assembly['dual_rewrite_doc_ids'],
        'total_docs_in_shuffled': int(total_rows), 'total_tokens': total,
        'target': TARGET_10B, 'overshoot': total - TARGET_10B,
        'shuffle_seed': SHUFFLE_SEED,
    }
    (PRETRAIN / setting / '_pretrain_manifest.json').write_text(json.dumps(manifest, indent=2))

    return dict(manifest=manifest, overlap=overlap, n_shards=n_shards,
                shared_docs_copied=shared_docs, shared_rows_copied=shared_rows)


# ----------------------------------------------------------------------------- final report
def write_report(step1, step2, step3):
    L = ['# Post-process report — final 10B pretrain-ready datasets', '',
         f'_Generated {time.strftime("%Y-%m-%d %H:%M:%S")}_', '',
         'Settings: ' + ', '.join(SETTINGS) + '. Budget convention: per-doc train length =',
         '`rewritten_tokens + 1` (rewritten) / `tokens-llama2 + 1` (shared-top). 5B per source '
         '→ 10B per arm.', '']

    # ---- Step 1
    L += ['## Step 1 — Wikipedia-prefix strip (in-place)', '',
          'Prefix stripped (start-anchored only): `"Here is a paraphrased version:\\n\\n"`.', '',
          '| setting | status2 | stripped | stripped % | no-prefix | no-prefix % | '
          'rewritten_tokens before | after | saved |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SETTINGS:
        d = step1[s]
        L.append(f'| {s} | {d["wiki_status2"]:,} | {d["wiki_stripped"]:,} | '
                 f'{d["wiki_stripped_pct"]:.2f}% | {d["wiki_no_prefix"]:,} | '
                 f'{d["wiki_no_prefix_pct"]:.2f}% | {d["wiki_tok_before"]:,} | '
                 f'{d["wiki_tok_after"]:,} | {d["wiki_tok_saved"]:,} |')
    L += ['', '### Distill template-preamble strip (shared extended rule)']
    for s in SETTINGS:
        dd = step1[s]['distill']
        top = '; '.join(f'{h!r}×{c}' for h, c in dd['top_first50'][:3])
        if dd.get('verdict') == 'STRIPPED':
            L.append(f'- **{s}**: verdict = STRIPPED; stripped = {dd.get("stripped", 0):,}/'
                     f'{dd.get("status2", 0):,} ({dd.get("stripped_pct", 0):.3f}%), '
                     f'tok saved = {dd.get("tok_saved", 0):,}; top first-50 (pre-strip): {top}')
        else:
            L.append(f'- **{s}**: verdict = {dd["verdict"]}; dominant_frac='
                     f'{dd.get("dominant_frac", "n/a")}; top first-50: {top}')
        if dd.get('candidate_prefix'):
            L.append(f'  - candidate preamble (not stripped): {dd["candidate_prefix"]!r}')

    # ---- Step 2
    L += ['', '## Step 2 — 5B rewritten assembly', '',
          'Sort key always on the ORIGINAL document\'s precomputed columns (rewritten text is '
          'NEVER scored):',
          '- **quality-first**: `fasttext-ranking-v2` DESC.',
          '- **signal-disagreement-lambdaX**: `u = q + λ·√v` DESC, where `q = mean(3 v2 cols)`, '
          '`v = population variance (ddof=0)`, computed in **float32** (per-setting λ — see the '
          'sort_key column).', '',
          '| setting | sort_key | wiki docs | wiki tok | distill docs | distill tok | '
          'dual doc_ids | total docs | total tok | gap before distill | overshoot |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SETTINGS:
        m = step2[s]
        L.append(f'| {s} | {m["sort_key"]} | {m["docs_from_wikipedia"]:,} | '
                 f'{m["tokens_from_wikipedia"]:,} | {m["docs_from_distill"]:,} | '
                 f'{m["tokens_from_distill"]:,} | {m["dual_rewrite_doc_ids"]:,} | '
                 f'{m["total_docs"]:,} | {m["total_tokens_plus_bos"]:,} | '
                 f'{m["gap_before_distill"]:,} | {m["overshoot"]:,} |')
    L += ['', '### Cross-pass coverage (wiki ↔ distill, joined on doc_id)',
          '| setting | status0 both (docs / src-tok) | status1 both (docs / src-tok) | '
          'recovered by distill | status2 wiki not distill | unique docs ≥1 status2 | '
          'source coverage |',
          '|---|---|---|---:|---:|---:|---|']
    for s in SETTINGS:
        c = step2[s]['coverage']
        L.append(f'| {s} | {c["status_0_both"]:,} / {c["status_0_both_tokens"]:,} | '
                 f'{c["status_1_both"]:,} / {c["status_1_both_tokens"]:,} | '
                 f'{c["recovered_by_distill"]:,} | {c["status_2_wiki_not_distill"]:,} | '
                 f'{c["unique_docs_with_any_status2"]:,} | '
                 f'{c["source_tokens_covered"]:,} / {c["total_selection_tokens"]:,} '
                 f'({c["coverage_frac"]*100:.2f}%) |')

    # ---- Step 3
    L += ['', '## Step 3 — shared-top mix + shuffle', '',
          '| setting | shared-top tok | rewritten tok | total | overshoot vs 10B | '
          'doc_id overlap | shuffled shards | shuffle seed |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in SETTINGS:
        m = step3[s]['manifest']
        L.append(f'| {s} | {m["shared_top_tokens"]:,} | {m["rewritten_tokens"]:,} | '
                 f'{m["total_tokens"]:,} | {m["overshoot"]:,} | {step3[s]["overlap"]} | '
                 f'{step3[s]["n_shards"]} | {m["shuffle_seed"]} |')

    # ---- confirmations / warnings
    warns = []
    for s in SETTINGS:
        if step2[s].get('shortfall'):
            warns.append(f'{s}: rewritten set fell SHORT of 5B (not padded).')
        if step3[s]['overlap'] != 0:
            warns.append(f'{s}: NON-ZERO doc_id overlap ({step3[s]["overlap"]}) between '
                         'shared-top and rewritten source — investigate.')
    L += ['', '## Confirmations',
          '- Final rewritten shards contain **only status==2** docs (status 0/1 excluded at '
          'assembly; verified by construction).',
          '- `tokens-llama2` (source length) and `rewritten_tokens` (output length) kept '
          'strictly separate throughout.']
    for s in SETTINGS:
        m = step2[s]
        ok = '≥ ~5B ✅' if not m.get('shortfall') else 'SHORTFALL ⚠️'
        L.append(f'- **{s}** rewritten total = {m["total_tokens_plus_bos"]:,} tokens ({ok}).')
    L += ['', '## Warnings / anomalies']
    L += ['- ' + w for w in warns] if warns else ['- None.']
    L += ['']
    (HERE / 'postprocess_report.md').write_text('\n'.join(L))
    log('wrote postprocess_report.md')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP3 START: settings={SETTINGS} workers={workers} seed={SHUFFLE_SEED}')

    step1 = json.loads((HERE / '_step1_summary.json').read_text())
    step2 = json.loads((HERE / '_step2_summary.json').read_text())

    step3 = {}
    for setting in SETTINGS:
        step3[setting] = mix(setting, workers, step2[setting])

    write_report(step1, step2, step3)
    log('STEP3 DONE.')

    bad = [s for s in SETTINGS if step3[s]['overlap'] != 0]
    if bad:
        stop(f'doc_id overlap non-zero for: {bad}')


if __name__ == '__main__':
    main()
