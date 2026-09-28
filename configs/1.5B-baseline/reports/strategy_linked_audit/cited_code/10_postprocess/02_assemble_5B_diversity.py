#!/usr/bin/env python
"""STEP 2 (diversity-first) — TOPIC-BALANCED 5B assembly.

diversity-first was selected with per-topic proportional quotas (each topic's share of
REMAINING's total tokens). The 5B rewritten assembly MUST preserve those proportions:
  * NO global top-5B sort (that destroys topic balance),
  * NO cross-topic backfill (policy A — a topic's shortfall is NOT redistributed),
  * within each topic the quality sort is `fasttext-ranking-v2` DESC (fastText only; there is
    NO u(d) / mean-of-three here — that belongs to signal-disagreement).

Per topic t:
  proportion[t] = sum(tokens-llama2 + 1 over t) / sum(tokens-llama2 + 1 over all)   [orig space]
  quota[t]      = 5e9 * proportion[t]
  Fill quota[t] from t's own status==2 docs: ALL wiki first (fasttext DESC, capped at quota),
  then top up from distill (fasttext DESC) until cumulative (rewritten_tokens + 1) reaches the
  gap (last doc kept whole). Same doc_id may appear in both passes — kept (two examples).
  If wiki+distill cannot reach quota[t], record the shortfall and move on (no backfill, no pad).

The distill pass may be INCOMPLETE (shards discovered by glob). Default: hard-stop if distill
has fewer shards than wiki; pass --allow-partial-distill to assemble on whatever exists (the
partial coverage is recorded in the manifest and report).

Output -> pretrain/diversity-first/rewritten/{wiki,distill}_NNNNN.parquet (atomic) +
_assembly_manifest.json + _step2_diversity_summary.json. CPU-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
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
SETTING = 'diversity-first'
SEL_DIR = BASE / SETTING                       # original selection shards (part_*.parquet)
PRETRAIN = Path('/scratch/bvandur1/zhuicon1/data_rewrite/pretrain')
HERE = Path('/scratch/bvandur1/zhuicon1/projects/rewrite/10_postprocess')

TARGET = 5_000_000_000
N_TOPICS_EXPECTED = 24
V2_FT = 'fasttext-ranking-v2'
SHARD_RE = re.compile(r'part_(\d+)\.parquet$')

# keep-most columns carried into the assembled shards (original `text` dropped; `rewritten`->`text`)
KEEP_COLS = ['doc_id', 'orig_doc_id', 'rewritten', 'rewritten_tokens', 'tokens-llama2',
             'status', 'topic', V2_FT, 'url', 'metadata']

TOPIC2CODE = {}                                 # set in main before forking; inherited by workers


def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
def stop(m): print(f'\n*** STOP: {m}\n', flush=True); sys.exit(2)
def _init_worker(): pa.set_cpu_count(1)


def list_shards(subdir):
    out = []
    for p in sorted(glob.glob(str(SEL_DIR / subdir / 'part_*.parquet'))):
        out.append((int(SHARD_RE.search(os.path.basename(p)).group(1)), p))
    return out


# ----------------------------------------------------------------------------- 2a topic proportions
def _read_topic_tokens(path):
    t = pq.read_table(path, columns=['topic', 'tokens-llama2'], use_threads=False)
    return (t.column('topic').to_pylist(),
            t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64))


def topic_proportions(workers):
    """proportion[t] = sum(tokens-llama2 + 1 over t) / sum over all, from the ORIGINAL selection."""
    sel = sorted(glob.glob(str(SEL_DIR / 'part_*.parquet')))
    if not sel:
        stop(f'no selection shards under {SEL_DIR}')
    tok_by_topic = {}
    doc_by_topic = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_read_topic_tokens, p) for p in sel]
        for fut in as_completed(futs):
            topics, toks = fut.result()
            toks_bos = toks + 1
            # accumulate per topic
            arr = np.asarray(topics, dtype=object)
            for tp in set(topics):
                m = arr == tp
                tok_by_topic[tp] = tok_by_topic.get(tp, 0) + int(toks_bos[m].sum())
                doc_by_topic[tp] = doc_by_topic.get(tp, 0) + int(m.sum())
            done += 1
            if done % 50 == 0:
                log(f'  2a proportions: {done}/{len(sel)} selection shards ({time.time()-t0:.0f}s)')

    topics_sorted = sorted(tok_by_topic)
    if len(topics_sorted) != N_TOPICS_EXPECTED:
        stop(f'expected {N_TOPICS_EXPECTED} topics, found {len(topics_sorted)}: {topics_sorted}')
    total = sum(tok_by_topic.values())
    prop = {t: tok_by_topic[t] / total for t in topics_sorted}
    log(f'2a: {len(topics_sorted)} topics, total orig tokens(+BOS)={total:,}')
    for t in topics_sorted:
        log(f'    {t:<22} prop={prop[t]:.4f}  quota={TARGET*prop[t]:,.0f}  '
            f'(orig docs={doc_by_topic[t]:,}, tok={tok_by_topic[t]:,})')
    log(f'2a: proportions sum = {sum(prop.values()):.6f}  quota sum = {sum(TARGET*p for p in prop.values()):,.0f}')
    return topics_sorted, prop, doc_by_topic, tok_by_topic


# ----------------------------------------------------------------------------- numeric pass
def _read_numeric(job):
    subdir, k, path = job
    t = pq.read_table(path, columns=['doc_id', 'status', 'rewritten_tokens', V2_FT,
                                      'topic', 'tokens-llama2'], use_threads=False)
    topics = t.column('topic').to_pylist()
    code = np.fromiter((TOPIC2CODE[x] for x in topics), count=len(topics), dtype=np.int16)
    return dict(
        k=k, n=t.num_rows,
        doc_id=t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64),
        status=t.column('status').to_numpy(zero_copy_only=False).astype(np.int8),
        rtok=t.column('rewritten_tokens').to_numpy(zero_copy_only=False).astype(np.int64),
        ft=t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32),
        code=code,
        llama=t.column('tokens-llama2').to_numpy(zero_copy_only=False).astype(np.int64))


def collect_numeric(subdir, shards, workers):
    res = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_read_numeric, (subdir, k, p)): k for k, p in shards}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'{subdir} numeric shard {k:05d} failed: {e!r}')
            res[r['k']] = r
            done += 1
            if done % 50 == 0:
                log(f'  numeric {subdir}: {done}/{len(shards)} ({time.time()-t0:.0f}s)')
    present = [k for k, _ in shards]
    ordered = [res[k] for k in present]
    rows = np.array([s['n'] for s in ordered], dtype=np.int64)
    offs = np.zeros(len(present) + 1, dtype=np.int64)
    offs[1:] = np.cumsum(rows)
    cat = lambda key: np.concatenate([s[key] for s in ordered])  # noqa: E731
    out = dict(present=present, pos={k: i for i, k in enumerate(present)}, offs=offs,
               paths={k: p for k, p in shards},
               doc_id=cat('doc_id'), status=cat('status'), rtok=cat('rtok'),
               ft=cat('ft'), code=cat('code'), llama=cat('llama'))
    out['len'] = out['rtok'] + 1
    return out


# ----------------------------------------------------------------------------- per-topic assembly
def fill_desc(idx, ft, length, target):
    """Order idx by ft DESC (stable), fill length until >= target (last whole)."""
    if idx.size == 0:
        return idx, 0, False
    order = idx[np.argsort(-ft[idx], kind='stable')]
    c = np.cumsum(length[order])
    if c[-1] < target:
        return order, int(c[-1]), False
    cut = int(np.searchsorted(c, target, side='left'))
    return order[:cut + 1], int(c[cut]), True


def assemble_topics(wiki, distill, quota_by_code, code2topic):
    keep_w = np.zeros(wiki['doc_id'].size, dtype=bool)
    keep_d = np.zeros(distill['doc_id'].size, dtype=bool)
    w_s2 = wiki['status'] == 2
    d_s2 = distill['status'] == 2
    per_topic = {}
    for c in range(len(code2topic)):
        quota = quota_by_code[c]
        widx = np.flatnonzero(w_s2 & (wiki['code'] == c))
        wiki_avail = int(wiki['len'][widx].sum()) if widx.size else 0
        ddocs = dtok = shortfall = 0

        if widx.size and wiki_avail >= quota:                  # wiki alone fills the quota
            sel, wtok, _ = fill_desc(widx, wiki['ft'], wiki['len'], quota)
            keep_w[sel] = True
            wdocs = int(sel.size)
        else:                                                  # keep all wiki, top up from distill
            keep_w[widx] = True
            wtok, wdocs = wiki_avail, int(widx.size)
            gap = quota - wtok
            didx = np.flatnonzero(d_s2 & (distill['code'] == c))
            if didx.size and gap > 0:
                dsel, dtok, filled = fill_desc(didx, distill['ft'], distill['len'], gap)
                keep_d[dsel] = True
                ddocs = int(dsel.size)
                if not filled:
                    shortfall = int(round(gap - dtok))
            elif gap > 0:
                shortfall = int(round(gap))                    # no distill docs for this topic

        total = wtok + dtok
        per_topic[code2topic[c]] = dict(
            quota=float(quota), wiki_docs=wdocs, wiki_tokens=int(wtok),
            distill_docs=ddocs, distill_tokens=int(dtok),
            total_docs=wdocs + ddocs, total_tokens=int(total),
            shortfall=int(max(0, shortfall)), overshoot=int(max(0, total - quota)))
    return keep_w, keep_d, per_topic


# ----------------------------------------------------------------------------- write pass
_WRITE = {}


def _write_shard(job):
    subdir, k, source_prompt, outprefix = job
    d = _WRITE[subdir]
    i = d['pos'][k]
    mask = d['keep'][d['offs'][i]:d['offs'][i + 1]]
    cnt = int(mask.sum())
    if cnt == 0:
        return (subdir, 0)
    t = pq.read_table(d['paths'][k], use_threads=False)
    sub = t.filter(pa.array(mask))
    out = sub.select(KEEP_COLS)
    names = out.column_names
    names[names.index('rewritten')] = 'text'
    out = out.rename_columns(names)
    out = out.append_column('source_prompt', pa.array([source_prompt] * cnt, type=pa.large_string()))
    outp = PRETRAIN / SETTING / 'rewritten' / f'{outprefix}_{k:05d}.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(out, outp)
    if pq.ParquetFile(outp).metadata.num_rows != cnt:
        raise RuntimeError(f'{outprefix} shard {k}: rowcount mismatch')
    return (subdir, cnt)


def write_blocks(wiki, distill, keep_w, keep_d, workers):
    global _WRITE
    _WRITE = {
        'wiki': dict(keep=keep_w, offs=wiki['offs'], pos=wiki['pos'], paths=wiki['paths']),
        'distill': dict(keep=keep_d, offs=distill['offs'], pos=distill['pos'],
                        paths=distill['paths']),
    }
    jobs = ([('wiki', k, 'wikipedia', 'wiki') for k in wiki['present']]
            + [('distill', k, 'distill', 'distill') for k in distill['present']])
    written = {'wiki': 0, 'distill': 0}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_write_shard, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                subdir, cnt = fut.result()
            except Exception as e:  # noqa: BLE001
                stop(f'write {j[3]} shard {j[1]:05d} failed: {e!r}')
            written[subdir] += cnt
            done += 1
            if done % 50 == 0:
                log(f'  write: {done}/{len(jobs)} ({time.time()-t0:.0f}s)')
    if written['wiki'] != int(keep_w.sum()) or written['distill'] != int(keep_d.sum()):
        stop(f'written {written} != selected '
             f'(wiki={int(keep_w.sum())}, distill={int(keep_d.sum())})')
    return written


# ----------------------------------------------------------------------------- cross-pass coverage
def cross_pass(wiki, distill, code2topic):
    """Cross-pass metrics over doc_ids present in BOTH passes (distill may be incomplete).

    paired_wiki_status asserts per-shard doc_id equality and falls back to an explicit doc_id
    join (never silently proceeds); metrics are computed over those paired docs, plus an
    overall 'unique docs with >=1 status2' that also credits wiki-only shards.
    """
    w_for_d = paired_wiki_status(wiki, distill)
    d = distill['status']; llama = distill['llama']; code = distill['code']

    def metrics(sel):
        w, dd, ll, cc = w_for_d[sel], d[sel], llama[sel], code[sel]

        def ct(mask): return int(mask.sum()), int(ll[mask].sum())
        s0n, s0t = ct((w == 0) & (dd == 0))
        s1bn, s1bt = ct((w == 1) & (dd == 1))
        recn, _ = ct((w == 1) & (dd == 2))
        s2wn, _ = ct((w == 2) & ((dd == 0) | (dd == 1)))
        return dict(status0_both=s0n, status0_both_tokens=s0t,
                    status1_both=s1bn, status1_both_tokens=s1bt,
                    recovered_by_distill=recn, status2_wiki_not_distill=s2wn)

    overall = metrics(np.arange(distill['doc_id'].size))
    # unique docs with >=1 status2 (all wiki shards + distill-only recoveries on paired shards)
    wiki_s2_total = int((wiki['status'] == 2).sum())
    extra = int(((d == 2) & (w_for_d != 2)).sum())
    overall['unique_docs_with_any_status2'] = wiki_s2_total + extra
    overall['paired_docs'] = int(distill['doc_id'].size)
    overall['wiki_docs_without_distill'] = int(wiki['doc_id'].size - distill['doc_id'].size)

    by_topic = {}
    for c in range(len(code2topic)):
        sel = np.flatnonzero(code == c)
        m = metrics(sel)
        w_topic_s2 = int(((wiki['status'] == 2) & (wiki['code'] == c)).sum())
        m['unique_docs_with_any_status2'] = w_topic_s2 + int(
            ((d == 2) & (w_for_d != 2) & (code == c)).sum())
        by_topic[code2topic[c]] = m
    return overall, by_topic


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK') or (os.cpu_count() or 8)))
    ap.add_argument('--allow-partial-distill', action='store_true',
                    help='assemble even if distill has fewer shards than wiki (records partial '
                         'coverage). Default: hard-stop when distill is incomplete.')
    args = ap.parse_args()
    workers = max(1, args.workers)
    log(f'STEP2 ({SETTING}) START: target={TARGET:,} workers={workers}')

    wiki_shards = list_shards('rewritten')
    distill_shards = list_shards('distill')
    log(f'  wiki shards={len(wiki_shards)}  distill shards={len(distill_shards)}')
    if len(distill_shards) < len(wiki_shards):
        msg = (f'distill is INCOMPLETE: {len(distill_shards)}/{len(wiki_shards)} shards present.')
        if not args.allow_partial_distill:
            stop(msg + ' Re-run with --allow-partial-distill to assemble on partial data, '
                       'or wait for the distill pass to finish.')
        log(f'  ⚠️ {msg} Proceeding (--allow-partial-distill); shortfalls will be larger.')

    # 2a/2b proportions + quotas
    topics, prop, sel_docs, sel_tok = topic_proportions(workers)
    global TOPIC2CODE
    TOPIC2CODE = {t: i for i, t in enumerate(topics)}
    code2topic = {i: t for t, i in TOPIC2CODE.items()}
    quota_by_code = {c: TARGET * prop[code2topic[c]] for c in range(len(topics))}

    # 2c numeric collection
    log('=== numeric pass ===')
    wiki = collect_numeric('rewritten', wiki_shards, workers)
    distill = collect_numeric('distill', distill_shards, workers)

    # 2d/2e per-topic assembly
    log('=== per-topic assembly (no cross-topic backfill) ===')
    keep_w, keep_d, per_topic = assemble_topics(wiki, distill, quota_by_code, code2topic)

    # overall tallies
    tok_wiki = sum(v['wiki_tokens'] for v in per_topic.values())
    tok_distill = sum(v['distill_tokens'] for v in per_topic.values())
    n_wiki = sum(v['wiki_docs'] for v in per_topic.values())
    n_distill = sum(v['distill_docs'] for v in per_topic.values())
    total = tok_wiki + tok_distill
    total_shortfall = sum(v['shortfall'] for v in per_topic.values())
    overshoot = max(0, total - TARGET)
    shortfall_topics = {t: v['shortfall'] for t, v in per_topic.items() if v['shortfall'] > 0}
    wiki_only_topics = [t for t, v in per_topic.items() if v['distill_docs'] == 0 and v['shortfall'] == 0]
    dual = int(np.intersect1d(wiki['doc_id'][keep_w], distill['doc_id'][keep_d]).size)
    log(f'  total assembled: {n_wiki+n_distill:,} docs, {total:,} tok '
        f'(target {TARGET:,}; shortfall {max(0,TARGET-total):,}; overshoot {overshoot:,})')
    if shortfall_topics:
        log(f'  ⚠️ shortfall topics ({len(shortfall_topics)}): '
            + ', '.join(f'{t}:{s:,}' for t, s in sorted(shortfall_topics.items(),
                                                        key=lambda x: -x[1])))

    # 2j cross-pass coverage
    cov_overall, cov_by_topic = cross_pass(wiki, distill, code2topic)
    log(f'  cross-pass (paired docs={cov_overall["paired_docs"]:,}, '
        f'wiki-only docs={cov_overall["wiki_docs_without_distill"]:,}): '
        f'unique any-status2={cov_overall["unique_docs_with_any_status2"]:,}')

    # 2g write
    log('=== writing assembled rewritten shards ===')
    write_blocks(wiki, distill, keep_w, keep_d, workers)

    # 2k manifest
    manifest = {
        'setting': SETTING,
        'topup_policy': 'A_no_cross_topic_backfill',
        'sort_key': 'fasttext-ranking-v2 (DESC, within topic)',
        'distill_shards_present': len(distill_shards),
        'wiki_shards_present': len(wiki_shards),
        'distill_complete': len(distill_shards) == len(wiki_shards),
        'total_docs': int(n_wiki + n_distill),
        'total_tokens_plus_bos': int(total),
        'docs_from_wikipedia': int(n_wiki), 'tokens_from_wikipedia': int(tok_wiki),
        'docs_from_distill': int(n_distill), 'tokens_from_distill': int(tok_distill),
        'dual_rewrite_doc_ids': dual,
        'target': TARGET,
        'total_shortfall_vs_target': int(max(0, TARGET - total)),
        'overshoot': int(overshoot),
        'topics_with_shortfall': shortfall_topics,
        'per_topic': {t: {'proportion': prop[t], **{k: per_topic[t][k] for k in (
            'quota', 'wiki_docs', 'wiki_tokens', 'distill_docs', 'distill_tokens',
            'total_tokens', 'shortfall')}} for t in topics},
    }
    (PRETRAIN / SETTING).mkdir(parents=True, exist_ok=True)
    (PRETRAIN / SETTING / '_assembly_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary = dict(manifest)
    summary.update(per_topic_full=per_topic, proportions=prop,
                   wiki_only_topics=wiki_only_topics,
                   coverage_overall=cov_overall, coverage_by_topic=cov_by_topic,
                   quota_proportion={t: prop[t] for t in topics})
    (HERE / '_step2_diversity_summary.json').write_text(json.dumps(summary, indent=2))
    log('STEP2 DONE; wrote _assembly_manifest.json + _step2_diversity_summary.json')
    if total < TARGET:
        log(f'  NOTE: total {total:,} < 5B by {TARGET-total:,} (per-topic shortfalls; '
            f'acceptable under policy A — NOT padded).')


if __name__ == '__main__':
    main()
