#!/usr/bin/env python
"""STEP 5 (rewrite / ReWire-inspired) — mix shared-top-5B + shuffle, and assemble the report.

  5a. Physically COPY shared-top-5B -> pretrain/rewrite/shared-top-5B/ (copy, not symlink),
      adding source_prompt="original"; `text` stays the original text.
  5b. Document-level RANDOM SHUFFLE (seed=42) of (shared-top + filtered rewritten) ->
      pretrain/rewrite/shuffled/ (~500k rows/shard).
  5c. Verify 0 doc_id overlap between shared-top and the rewrite set's ORIGINAL source docs.
  5d. Combined budget: shared-top (tokens-llama2 + 1) + rewritten (rewritten_tokens + 1).
  5e. Final source_prompt distribution (original / wikipedia / distill).
  5f. _pretrain_manifest.json.

Then append a `rewrite` section to postprocess_report.md (idempotent, marker-delimited),
including the full ReWire funnel and cross-pass coverage. Shuffle uses pp_io.bucketed_shuffle
(memory-bounded two-pass; safe under the 256 GB cpu cap). CPU-only.
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

BASE = Path('/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/10B')
SHARED_SRC = BASE / 'shared-top-5B'
SETTING = 'rewrite'
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

SHUFFLE_SEED = 42
ROWS_PER_SHARD = 500_000
TARGET_10B = 10_000_000_000
SOURCE_ORDER = ['original', 'wikipedia', 'distill']

UNIFIED = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()),
                     ('text', pa.large_string()), ('source_prompt', pa.large_string()),
                     ('train_tokens', pa.int32())])

MARK_START = '<!-- REWRITE START -->'
MARK_END = '<!-- REWRITE END -->'


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


def _copy_shared_shard(src):
    t = pq.read_table(src, use_threads=False)
    n = t.num_rows
    t = t.append_column('source_prompt', pa.array(['original'] * n, type=pa.large_string()))
    outp = PRETRAIN / SETTING / 'shared-top-5B' / Path(src).name
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(t, outp)
    if pq.ParquetFile(outp).metadata.num_rows != n:
        raise RuntimeError(f'shared-top copy {Path(src).name}: rowcount mismatch')
    return n


def copy_shared_top(workers):
    srcs = sorted(glob.glob(str(SHARED_SRC / 'part_*.parquet')))
    if not srcs:
        stop(f'no shared-top shards under {SHARED_SRC}')
    total = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_copy_shared_shard, s): s for s in srcs}
        for fut in as_completed(futs):
            try:
                total += fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'shared-top copy {Path(futs[fut]).name} failed: {e!r}')
    log(f'  5a copied {len(srcs)} shared-top shards -> {total:,} rows')
    return len(srcs), total


def load_unified(path, tok_col):
    t = pq.read_table(path, columns=['doc_id', 'orig_doc_id', 'text', 'source_prompt', tok_col],
                      use_threads=False)
    return pa.table({
        'doc_id': t.column('doc_id').cast(pa.int64()),
        'orig_doc_id': t.column('orig_doc_id').cast(pa.int64()),
        'text': t.column('text').cast(pa.large_string()),
        'source_prompt': t.column('source_prompt').cast(pa.large_string()),
        'train_tokens': t.column(tok_col).cast(pa.int32()),
    }, schema=UNIFIED)


def shuffle_and_write():
    """Memory-bounded two-pass bucketed shuffle (pp_io.bucketed_shuffle); never holds the full
    corpus + a copy. Independent of Step 2/4 RNGs (its own default_rng(42))."""
    shared_dir = PRETRAIN / SETTING / 'shared-top-5B'
    rewritten_dir = PRETRAIN / SETTING / 'rewritten'
    specs = [(p, 'tokens-llama2')
             for p in sorted(glob.glob(str(shared_dir / 'part_*.parquet')))]
    rew_files = sorted(glob.glob(str(rewritten_dir / 'wiki_*.parquet'))
                       + glob.glob(str(rewritten_dir / 'distill_*.parquet')))
    if not rew_files:
        stop(f'no rewritten shards under {rewritten_dir} (run Step 4 first)')
    specs += [(p, 'rewritten_tokens') for p in rew_files]

    mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE') or 0)
    mem_bytes = mem_mb * 1024 * 1024 if mem_mb else None
    total_rows, n_shards, B = bucketed_shuffle(
        specs, lambda s: load_unified(s[0], s[1]),
        PRETRAIN / SETTING / 'shuffled', PRETRAIN / SETTING / '_shuffle_tmp',
        seed=SHUFFLE_SEED, rows_per_shard=ROWS_PER_SHARD, mem_bytes=mem_bytes, log=log)
    log(f'  5b shuffled {total_rows:,} rows into {n_shards} shards (B={B}, seed={SHUFFLE_SEED})')
    return total_rows, n_shards


# ----------------------------------------------------------------------------- report append
def render_block(s1, s2, s3, s4, s5):
    L = [MARK_START, '', '# Post-process report — rewrite (ReWire-inspired)', '',
         f'_Generated {time.strftime("%Y-%m-%d %H:%M:%S")}_', '',
         'Pipeline: rewrite broadly → FastText-score the **rewritten** output → keep top 5B. '
         'Score = identical preprocessing & definition as `fasttext-ranking-v2` '
         '(`01_explore/score_fasttext.py` + `00_TMP/clean_v2_ranks.py`): '
         '`raw = p if __label__hq else 1-p`; v2 = `rankdata(raw,"average")/99,949,162` vs the '
         'original 100M-doc raw fasttext distribution.', '']

    # Step 1
    L += ['## Step 1 — prefix strip',
          f'- wiki shards={s1["wiki_shards"]}, distill shards={s1["distill_shards"]}'
          + ('  ⚠️ **distill empty**' if s1['distill_shards'] == 0 else ''),
          f'- wiki status2={s1["wiki_status2"]:,}; stripped={s1["wiki_stripped"]:,} '
          f'({s1["wiki_stripped_pct"]:.2f}%); rewritten_tokens before={s1["wiki_tok_before"]:,} '
          f'after={s1["wiki_tok_after"]:,} saved={s1["wiki_tok_saved"]:,}',
          f'- distill prefix check: **{s1["distill"]["verdict"]}**', '']

    # Step 2 pool
    L += ['## Step 2 — 15B pool (pre-FastText)',
          f'- wiki status2: {s2["pool_from_wikipedia"]["docs"]:,} docs / '
          f'{s2["pool_from_wikipedia"]["tokens"]:,} tok; distill status2: '
          f'{s2["pool_from_distill"]["docs"]:,} docs / {s2["pool_from_distill"]["tokens"]:,} tok.',
          f'- combined pool: {s2["pool_docs"]:,} docs / {s2["pool_tokens"]:,} tok '
          f'(target ~15B; shortfall {s2["pool_shortfall_vs_15B"]:,}).',
          f'- dual-rewrite doc_ids (status2 both): {s2["dual_rewrite_doc_ids"]:,}.', '']

    # Step 3 scores
    def drow(d):
        return ('n/a' if d['n'] == 0 else
                f'min={d["min"]:.4f} p10={d["p10"]:.4f} p25={d["p25"]:.4f} '
                f'median={d["median"]:.4f} p75={d["p75"]:.4f} p90={d["p90"]:.4f} '
                f'max={d["max"]:.4f} (n={d["n"]:,})')
    L += ['## Step 3 — FastText scoring of rewritten text (raw P(hq))',
          f'- model: {s3["fasttext_model"]}; reference N={s3["reference_N"]:,}; '
          f'scored {s3["scored_docs"]:,} docs.',
          f'- overall: {drow(s3["distribution_overall"])}',
          f'- wikipedia: {drow(s3["distribution_wikipedia"])}',
          f'- distill: {drow(s3["distribution_distill"])}', '']

    # Step 4 filter
    kw = s4['kept_from_wikipedia']; kd = s4['kept_from_distill']
    kt = s4['kept_tokens'] or 1

    def d2(d):
        return ('n/a' if d['n'] == 0 else
                f'min={d["min"]:.4f} p10={d["p10"]:.4f} median={d["median"]:.4f} '
                f'p90={d["p90"]:.4f} max={d["max"]:.4f}')
    db = s4['dual_breakdown']
    L += ['## Step 4 — keep top 5B by rewritten_fasttext_score (global sort, DESC)',
          f'- kept {s4["kept_docs"]:,} docs / {s4["kept_tokens"]:,} tok (target 5B; overshoot '
          f'{s4["overshoot"]:,}); **score cutoff = {s4["fasttext_score_cutoff"]:.6f}**.',
          f'- from wikipedia: {kw["docs"]:,} docs / {kw["tokens"]:,} tok '
          f'({100.0*kw["tokens"]/kt:.1f}%); from distill: {kd["docs"]:,} docs / '
          f'{kd["tokens"]:,} tok ({100.0*kd["tokens"]/kt:.1f}%).',
          f'- kept score dist: {d2(s4["kept_score_dist"])}',
          f'- rejected score dist: {d2(s4["rejected_score_dist"])}',
          f'- dual rewrites in pool: {s4["dual_rewrite_doc_ids_in_pool"]:,} → '
          f'both_kept={db["both_kept"]:,}, both_rejected={db["both_rejected"]:,}, '
          f'wiki_kept_distill_rejected={db["wiki_kept_distill_rejected"]:,}, '
          f'distill_kept_wiki_rejected={db["distill_kept_wiki_rejected"]:,}.', '']

    # Funnel
    L += ['## ReWire funnel', '| stage | docs | tokens |', '|---|---:|---:|']
    for row in s2['funnel_pre_fasttext']:
        L.append(f'| {row["stage"]} | {row["docs"]:,} | {row["tokens"]:,} |')
    ff = s4['funnel_final']
    L.append(f'| {ff["stage"]} | {ff["docs"]:,} | {ff["tokens"]:,} |')

    # Cross-pass
    c = s2['coverage']
    L += ['', '## Cross-pass coverage (wiki ↔ distill)',
          f'- paired docs={c["paired_docs"]:,}; wiki docs without distill='
          f'{c["wiki_docs_without_distill"]:,}.',
          f'- status0-both={c["status0_both"]:,} ({c["status0_both_tokens"]:,} src-tok); '
          f'status1-both={c["status1_both"]:,} ({c["status1_both_tokens"]:,} src-tok); '
          f'recovered-by-distill={c["recovered_by_distill"]:,}; unique docs ≥1 status2='
          f'{c["unique_docs_with_any_status2"]:,}.']

    # Step 5
    m = s5['manifest']
    L += ['', '## Step 5 — shared-top mix + shuffle',
          f'- shared-top {m["shared_top_tokens"]:,} tok + rewritten {m["rewritten_tokens"]:,} '
          f'tok = **{m["total_tokens"]:,}** tok (target 10B; overshoot {m["overshoot"]:,}).',
          f'- doc_id overlap (shared-top ∩ rewrite source) = {s5["overlap"]} (expect 0).',
          f'- shuffled into {s5["n_shards"]} shards (seed {SHUFFLE_SEED}).', '',
          '### Final source distribution',
          '| source_prompt | docs | tokens | % of total tokens |',
          '|---|---:|---:|---:|']
    sd = s5['source_dist']; ttot = sum(v['tokens'] for v in sd.values()) or 1
    for s in SOURCE_ORDER:
        if s in sd:
            v = sd[s]
            L.append(f'| {s} | {v["docs"]:,} | {v["tokens"]:,} | '
                     f'{100.0*v["tokens"]/ttot:.2f}% |')

    warns = []
    if s2['pool_shortfall_vs_15B'] > 0:
        warns.append(f'pool reached {s2["pool_tokens"]:,} (< 15B target) — distill '
                     f'{s2["distill_shards"]} shards; top-5B still satisfied from wiki.')
    if m['total_tokens'] < TARGET_10B:
        warns.append(f'final total {m["total_tokens"]:,} < 10B.')
    if s5['overlap'] != 0:
        warns.append(f'NON-ZERO doc_id overlap ({s5["overlap"]}).')
    L += ['', '## Confirmations',
          '- Only status==2 docs scored/kept; no padding.',
          '- FastText scored the REWRITTEN text (not the original); preprocessing + score '
          'definition identical to fasttext-ranking-v2.',
          '- Global top-5B sort with no wiki/distill preference.',
          '', '## Warnings / anomalies']
    L += ['- ' + w for w in warns] if warns else ['- None.']
    L += ['', MARK_END, '']
    return '\n'.join(L)


def append_report(block):
    path = HERE / 'postprocess_report.md'
    existing = path.read_text() if path.exists() else ''
    if MARK_START in existing and MARK_END in existing:
        pre = existing[:existing.index(MARK_START)].rstrip('\n')
        post = existing[existing.index(MARK_END) + len(MARK_END):]
        existing = (pre + '\n' if pre else '') + post.lstrip('\n')
    if not existing.strip():
        existing = '# Post-process report\n'
    path.write_text(existing.rstrip('\n') + '\n\n' + block)
    log(f'appended rewrite section to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP5 (rewrite) START: workers={workers} seed={SHUFFLE_SEED}')

    s1 = json.loads((HERE / '_step1_rewrite_summary.json').read_text())
    s2 = json.loads((HERE / '_step2_rewrite_summary.json').read_text())
    s3 = json.loads((HERE / '_step3_rewrite_summary.json').read_text())
    s4 = json.loads((HERE / '_step4_rewrite_summary.json').read_text())

    copy_shared_top(workers)
    total_rows, n_shards = shuffle_and_write()

    # post-shuffle stats by streaming the final shards' light columns (no full-corpus table)
    st = stream_shuffle_stats(PRETRAIN / SETTING / 'shuffled')
    overlap = st['overlap']
    log(f'  5c doc_id overlap (shared-top ∩ rewrite source) = {overlap} (expect 0)')
    shared_top_tokens = st['shared_top_tokens']; rewritten_tokens = st['rewritten_tokens']
    total = st['total_tokens']
    log(f'  5d shared_top={shared_top_tokens:,} + rewritten={rewritten_tokens:,} = {total:,} '
        f'(target 10B, overshoot {total-TARGET_10B:,})')
    source_dist = st['source_dist']

    manifest = {
        'setting': SETTING,
        'shared_top_docs': st['shared_top_docs'], 'shared_top_tokens': shared_top_tokens,
        'rewritten_docs': st['rewritten_docs'], 'rewritten_tokens': rewritten_tokens,
        'rewritten_from_wikipedia': s4['kept_from_wikipedia']['docs'],
        'rewritten_from_distill': s4['kept_from_distill']['docs'],
        'fasttext_score_cutoff': s4['fasttext_score_cutoff'],
        'total_docs_in_shuffled': int(total_rows), 'total_tokens': total,
        'target': TARGET_10B, 'overshoot': total - TARGET_10B, 'shuffle_seed': SHUFFLE_SEED,
    }
    (PRETRAIN / SETTING / '_pretrain_manifest.json').write_text(json.dumps(manifest, indent=2))
    log('  wrote _pretrain_manifest.json')

    append_report(render_block(s1, s2, s3, s4,
                               dict(manifest=manifest, overlap=overlap, n_shards=n_shards,
                                    source_dist=source_dist)))
    log('STEP5 DONE.')
    if overlap != 0:
        stop(f'doc_id overlap non-zero ({overlap})')


if __name__ == '__main__':
    main()
