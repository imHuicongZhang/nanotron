#!/usr/bin/env python
"""STEP 3 (diversity-first) — mix shared-top-5B (original text) with the topic-balanced 5B
rewritten set, shuffle, and report topic distributions.

  3a. Physically COPY shared-top-5B -> pretrain/diversity-first/shared-top-5B/ (copy, not
      symlink), adding source_prompt="original"; `text` stays the original text.
  3b. Document-level RANDOM SHUFFLE (seed=42) of (shared-top + rewritten) ->
      pretrain/diversity-first/shuffled/ (~500k rows/shard).
  3c. Verify 0 doc_id overlap between shared-top and the rewritten set's ORIGINAL source docs
      (dedup the dual-rewrite rows by doc_id).
  3d. Combined budget: shared-top (tokens-llama2 + 1) + rewritten (rewritten_tokens + 1).
  3e. THREE topic distributions side by side — rewritten (realized, in rewritten-token space),
      shared-top (quality-ranked, not balanced), combined — plus the quota proportions
      (original tokens-llama2 space). Compression differs by topic, so the realized rewritten
      distribution will not exactly equal the quotas; both are reported so the deviation is
      explicit.
  3f. Write _pretrain_manifest.json (incl. all four distributions).

Finally, append a diversity-first section to postprocess_report.md (idempotent, delimited by
markers — does not disturb the quality-first/signal-disagreement content). In-memory shuffle
(pp_io.bucketed_shuffle: memory-bounded, safe under the 256 GB cpu cap). CPU-only.
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
SETTING = 'diversity-first'
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

SHUFFLE_SEED = 42
ROWS_PER_SHARD = 500_000
TARGET_10B = 10_000_000_000

UNIFIED = pa.schema([('doc_id', pa.int64()), ('orig_doc_id', pa.int64()),
                     ('text', pa.large_string()), ('source_prompt', pa.large_string()),
                     ('train_tokens', pa.int32()), ('topic', pa.large_string())])

MARK_START = '<!-- DIVERSITY-FIRST START -->'
MARK_END = '<!-- DIVERSITY-FIRST END -->'


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


# ----------------------------------------------------------------------------- 3a copy shared-top
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
    log(f'  3a copied {len(srcs)} shared-top shards -> {total:,} rows')
    return len(srcs), total


# ----------------------------------------------------------------------------- unified loader
def load_unified(path, tok_col):
    t = pq.read_table(path, columns=['doc_id', 'orig_doc_id', 'text', 'source_prompt',
                                      tok_col, 'topic'], use_threads=False)
    return pa.table({
        'doc_id': t.column('doc_id').cast(pa.int64()),
        'orig_doc_id': t.column('orig_doc_id').cast(pa.int64()),
        'text': t.column('text').cast(pa.large_string()),
        'source_prompt': t.column('source_prompt').cast(pa.large_string()),
        'train_tokens': t.column(tok_col).cast(pa.int32()),
        'topic': t.column('topic').cast(pa.large_string()),
    }, schema=UNIFIED)


# ----------------------------------------------------------------------------- 3b shuffle
def shuffle_and_write():
    """Memory-bounded two-pass bucketed shuffle (pp_io.bucketed_shuffle); never holds the full
    corpus + a copy. Own default_rng(42), independent of Step 2's selection."""
    shared_dir = PRETRAIN / SETTING / 'shared-top-5B'
    rewritten_dir = PRETRAIN / SETTING / 'rewritten'
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
        PRETRAIN / SETTING / 'shuffled', PRETRAIN / SETTING / '_shuffle_tmp',
        seed=SHUFFLE_SEED, rows_per_shard=ROWS_PER_SHARD, mem_bytes=mem_bytes, log=log)
    log(f'  3b shuffled {total_rows:,} rows into {n_shards} shards (B={B}, seed={SHUFFLE_SEED})')
    return total_rows, n_shards


# ----------------------------------------------------------------------------- 3e distributions
def dist_from_topic_tok(topic_tok, topics_all):
    """Build the three token-weighted topic distributions (realized-rewritten, shared-top,
    combined) from stream_shuffle_stats' topic_tok dict keyed by (is_original, topic). Token
    weights already include the +1 BOS."""
    rew = {t: topic_tok.get((False, t), 0) for t in topics_all}
    sha = {t: topic_tok.get((True, t), 0) for t in topics_all}
    rew_tot = sum(rew.values()) or 1
    sha_tot = sum(sha.values()) or 1
    comb_tot = rew_tot + sha_tot
    rr = {t: rew[t] / rew_tot for t in topics_all}
    sd = {t: sha[t] / sha_tot for t in topics_all}
    cd = {t: (rew[t] + sha[t]) / comb_tot for t in topics_all}
    return rr, sd, cd


# ----------------------------------------------------------------------------- report append
def render_block(step1, step2, step3):
    topics = list(step2['per_topic'].keys())
    L = [MARK_START, '', '# Post-process report — diversity-first', '',
         f'_Generated {time.strftime("%Y-%m-%d %H:%M:%S")}_  '
         f'(topup policy: A — no cross-topic backfill; within-topic sort: fasttext-ranking-v2 DESC)',
         '']

    # Step 1
    s1 = step1
    L += ['## Step 1 — prefix strip',
          f'- wiki shards = {s1["wiki_shards"]}, distill shards = {s1["distill_shards"]}'
          + ('  ⚠️ **distill incomplete**' if s1['distill_shards'] < s1['wiki_shards'] else ''),
          f'- wiki status2 = {s1["wiki_status2"]:,}; stripped = {s1["wiki_stripped"]:,} '
          f'({s1["wiki_stripped_pct"]:.2f}%); no-prefix = {s1["wiki_no_prefix"]:,} '
          f'({s1["wiki_no_prefix_pct"]:.2f}%)',
          f'- rewritten_tokens before = {s1["wiki_tok_before"]:,}; after = '
          f'{s1["wiki_tok_after"]:,}; saved = {s1["wiki_tok_saved"]:,}',
          f'- distill prefix scan: verdict = **{s1["distill"]["verdict"]}**'
          + (f'; stripped {s1["distill"].get("stripped", 0):,} rows, saved '
             f'{s1["distill"].get("tok_saved", 0):,} tokens'
             if s1['distill']['verdict'] == 'STRIPPED' else ''), '']

    # Step 2 per-topic table
    L += ['## Step 2 — topic-balanced 5B assembly', '',
          '| topic | proportion | quota | wiki_docs | wiki_tokens | distill_docs | '
          'distill_tokens | total_docs | total_tokens | shortfall | overshoot |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    pt_full = step2['per_topic_full']
    for t in topics:
        v = pt_full[t]; p = step2['proportions'][t]
        L.append(f'| {t} | {p:.4f} | {v["quota"]:,.0f} | {v["wiki_docs"]:,} | '
                 f'{v["wiki_tokens"]:,} | {v["distill_docs"]:,} | {v["distill_tokens"]:,} | '
                 f'{v["total_docs"]:,} | {v["total_tokens"]:,} | {v["shortfall"]:,} | '
                 f'{v["overshoot"]:,} |')
    L += ['',
          f'- **Overall**: wikipedia {step2["docs_from_wikipedia"]:,} docs / '
          f'{step2["tokens_from_wikipedia"]:,} tok; distill {step2["docs_from_distill"]:,} docs / '
          f'{step2["tokens_from_distill"]:,} tok; dual doc_ids {step2["dual_rewrite_doc_ids"]:,}.',
          f'- Total assembled: {step2["total_docs"]:,} docs / {step2["total_tokens_plus_bos"]:,} '
          f'tok (target 5B; shortfall {step2["total_shortfall_vs_target"]:,}; overshoot '
          f'{step2["overshoot"]:,}).',
          f'- Topics fully filled by wiki alone (no distill): '
          f'{step2["wiki_only_topics"] if step2["wiki_only_topics"] else "none"}.',
          f'- Topics with shortfall: '
          + (', '.join(f'{t} ({s:,})' for t, s in sorted(step2['topics_with_shortfall'].items(),
                                                          key=lambda x: -x[1]))
             if step2['topics_with_shortfall'] else 'none') + '.', '']

    # Step 2 cross-pass coverage by topic
    co = step2['coverage_overall']
    L += ['### Cross-pass coverage (wiki ↔ distill)',
          f'- distill is paired on {co["paired_docs"]:,} docs; '
          f'{co["wiki_docs_without_distill"]:,} wiki docs have NO distill counterpart '
          f'(distill shards: {step2["distill_shards_present"]}/{step2["wiki_shards_present"]}).',
          f'- overall: status0-both = {co["status0_both"]:,} ({co["status0_both_tokens"]:,} '
          f'src-tok); status1-both = {co["status1_both"]:,} ({co["status1_both_tokens"]:,} '
          f'src-tok); recovered-by-distill = {co["recovered_by_distill"]:,}; '
          f'status2-wiki-not-distill = {co["status2_wiki_not_distill"]:,}; unique docs with '
          f'≥1 status2 = {co["unique_docs_with_any_status2"]:,}.', '',
          '| topic | status0_both (n/tok) | status1_both (n/tok) | recovered | '
          's2_wiki_not_distill | unique ≥1 status2 |',
          '|---|---|---|---:|---:|---:|']
    cbt = step2['coverage_by_topic']
    for t in topics:
        c = cbt[t]
        L.append(f'| {t} | {c["status0_both"]:,} / {c["status0_both_tokens"]:,} | '
                 f'{c["status1_both"]:,} / {c["status1_both_tokens"]:,} | '
                 f'{c["recovered_by_distill"]:,} | {c["status2_wiki_not_distill"]:,} | '
                 f'{c["unique_docs_with_any_status2"]:,} |')

    # Step 3
    m = step3['manifest']
    L += ['', '## Step 3 — shared-top mix + shuffle',
          f'- shared-top {m["shared_top_tokens"]:,} tok + rewritten {m["rewritten_tokens"]:,} '
          f'tok = **{m["total_tokens"]:,}** tok (target 10B; overshoot {m["overshoot"]:,}; '
          'may be under 10B if Step 2 fell short of 5B).',
          f'- doc_id overlap (shared-top ∩ rewritten source) = {step3["overlap"]} (expect 0).',
          f'- shuffled into {step3["n_shards"]} shards (seed {SHUFFLE_SEED}).', '',
          '### Topic distributions (token-weighted) — quota vs realized vs shared-top vs combined',
          '| topic | quota_prop (orig space) | rewritten_realized | shared_top | combined |',
          '|---|---:|---:|---:|---:|']
    qd = m['rewritten_topic_distribution_quota']
    rr = m['rewritten_topic_distribution_realized']
    sd = m['shared_top_topic_distribution']
    cd = m['combined_topic_distribution']
    for t in topics:
        L.append(f'| {t} | {qd.get(t,0):.4f} | {rr.get(t,0):.4f} | {sd.get(t,0):.4f} | '
                 f'{cd.get(t,0):.4f} |')

    # confirmations
    warns = []
    if step2['total_tokens_plus_bos'] < step2['target']:
        warns.append(f'rewritten total fell short of 5B by '
                     f'{step2["total_shortfall_vs_target"]:,} tok (policy A: not padded).')
    if not step2['distill_complete']:
        warns.append(f'distill incomplete ({step2["distill_shards_present"]}/'
                     f'{step2["wiki_shards_present"]} shards) — coverage/top-up are partial.')
    if step3['overlap'] != 0:
        warns.append(f'NON-ZERO doc_id overlap ({step3["overlap"]}).')
    L += ['', '## Confirmations',
          '- Only status==2 docs in the assembled rewritten set.',
          '- No cross-topic backfill; no padding with status 0/1 or original text.',
          '- tokens-llama2 (source) and rewritten_tokens (output) kept separate.',
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
    log(f'appended diversity-first section to {path}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP3 ({SETTING}) START: workers={workers} seed={SHUFFLE_SEED}')

    step1 = json.loads((HERE / '_step1_diversity_summary.json').read_text())
    step2 = json.loads((HERE / '_step2_diversity_summary.json').read_text())

    shared_docs, _ = copy_shared_top(workers)
    total_rows, n_shards = shuffle_and_write()

    # 3c-3e: stream the final shards' light columns (no full-corpus table)
    st = stream_shuffle_stats(PRETRAIN / SETTING / 'shuffled', with_topic=True)
    overlap = st['overlap']
    log(f'  3c doc_id overlap (shared-top ∩ rewritten-source) = {overlap} (expect 0)')
    shared_top_tokens = st['shared_top_tokens']; rewritten_tokens = st['rewritten_tokens']
    total = st['total_tokens']
    log(f'  3d shared_top={shared_top_tokens:,} + rewritten={rewritten_tokens:,} = {total:,} '
        f'(target 10B, overshoot {total-TARGET_10B:,})')

    # 3e topic distributions (token-weighted; weights from the streamed topic_tok)
    topics_all = list(step2['per_topic'].keys())
    rr, sd, cd = dist_from_topic_tok(st['topic_tok'], topics_all)
    qd = {t: step2['proportions'][t] for t in topics_all}

    manifest = {
        'setting': SETTING,
        'shared_top_docs': st['shared_top_docs'], 'shared_top_tokens': shared_top_tokens,
        'rewritten_docs': st['rewritten_docs'], 'rewritten_tokens': rewritten_tokens,
        'rewritten_from_wikipedia': step2['docs_from_wikipedia'],
        'rewritten_from_distill': step2['docs_from_distill'],
        'dual_rewrite_docs': step2['dual_rewrite_doc_ids'],
        'total_docs_in_shuffled': int(total_rows), 'total_tokens': total,
        'target': TARGET_10B, 'overshoot': total - TARGET_10B, 'shuffle_seed': SHUFFLE_SEED,
        'rewritten_topic_distribution_quota': qd,
        'rewritten_topic_distribution_realized': rr,
        'shared_top_topic_distribution': sd,
        'combined_topic_distribution': cd,
    }
    (PRETRAIN / SETTING / '_pretrain_manifest.json').write_text(json.dumps(manifest, indent=2))
    log('  wrote _pretrain_manifest.json')

    append_report(render_block(step1, step2,
                               dict(manifest=manifest, overlap=overlap, n_shards=n_shards,
                                    shared_docs=shared_docs)))
    log('STEP3 DONE.')
    if overlap != 0:
        stop(f'doc_id overlap non-zero ({overlap})')


if __name__ == '__main__':
    main()
