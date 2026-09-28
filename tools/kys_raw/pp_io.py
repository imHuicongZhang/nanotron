#!/usr/bin/env python
"""Shared I/O + correctness helpers for the 10_postprocess pipeline (PREFLIGHT2).

  * atomic_write_table / atomic_copy — write to `<dest>.tmp` IN THE SAME DIRECTORY then
    os.replace; on ANY exception the half-written .tmp is unlinked (try/finally) so a killed
    job never leaves a stale .tmp that a rerun could mistake for finished output.
  * paired_wiki_status — align a wiki status array to distill order by asserting per-shard
    doc_id equality, falling back to an EXPLICIT doc_id join (never silently proceeds).
  * bucketed_shuffle — memory-bounded two-pass document-level shuffle (seed=42) that never
    holds the whole corpus plus a copy; safe under the 256 GB cpu cap.
  * stream_shuffle_stats — post-shuffle stats (source dist, doc_id overlap, budget, optional
    per-topic token weights) computed by streaming the final shards' light columns only.

Imported by the 10_postprocess scripts; fork-safe (pure functions, no global state).
"""
from __future__ import annotations

import glob
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GIB = 1024 ** 3
MEM_CAP_BYTES = 256 * GIB          # hard cpu-partition memory cap


# ----------------------------------------------------------------------------- atomic writes
def atomic_write_table(table, dest, compression='zstd'):
    """Write `table` to `dest` atomically (.tmp in the SAME dir + os.replace).
    On any exception the partial .tmp is removed so it can't be mistaken for output."""
    dest = str(dest)
    tmp = dest + '.tmp'
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_copy(src, dest):
    """Byte-for-byte copy src -> dest atomically (.tmp in the SAME dir + os.replace)."""
    dest = str(dest)
    tmp = dest + '.tmp'
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------- preamble / leak strip
# Shared, fork-safe strip rules used by EVERY arm's Step-1 script so preamble hygiene is identical
# across all settings. Two artifacts are removed, both START-ANCHORED (never .replace/.lstrip):
#   * strip_instruction_leak     — the rephrasing SYSTEM-PROMPT that leaked verbatim into the start
#                                  of some wiki rewrites (a fixed instruction paragraph).
#   * strip_distill_preamble     — leading paraphrase/condense TEMPLATE preambles in distill, the
#                                  canonical "extended" rule (first-paragraph + single-'\n' first-line).
INSTRUCTION_LEAK_ANCHOR = (
    "Important: Do not add any information, claims, or details that are not")
DISTILL_MAX_PREAMBLE_CHARS = 120
DISTILL_PREAMBLE_WORDS = ('paraphras', 'condensed', 'rewrit', 'summary', 'version')
DISTILL_OPENERS = ('here is', "here's", 'here are', 'certainly', 'sure', 'below is',
                   'of course', 'the following is')


def strip_instruction_leak(text):
    """Remove a leaked rephrasing-instruction block from the START of `text`.

    Some wiki rewrites begin with the (fixed) system-prompt instruction paragraph that opens with
    INSTRUCTION_LEAK_ANCHOR and ends at the first '\\n\\n', after which the real article follows.
    START-ANCHORED only (mid-text occurrences are genuine content and are left untouched). If the
    whole doc is the leak (no '\\n\\n'), returns ('', True). Returns (new_text, stripped?)."""
    if text and text.startswith(INSTRUCTION_LEAK_ANCHOR):
        cut = text.find('\n\n')
        return (text[cut + 2:], True) if cut >= 0 else ('', True)
    return text, False


def strip_distill_preamble(text):
    """Remove a leading paraphrase/condense TEMPLATE preamble from distill text (start-anchored).

    (a) first-PARAGRAPH rule: the first paragraph (up to '\\n\\n', within DISTILL_MAX_PREAMBLE_CHARS)
        is a meta-preamble — a markdown header (###/##/**) OR a DISTILL_OPENERS opener, in either
        case containing a DISTILL_PREAMBLE_WORDS word, OR a bare 'Paraphrased ...:' label. Remove
        the paragraph + its '\\n\\n'.
    (b) first-LINE rule (single-'\\n' forms): the first line (up to '\\n', within
        DISTILL_MAX_PREAMBLE_CHARS) ENDS WITH ':' AND contains a meta word AND starts with a header
        (###/##/**) or a DISTILL_OPENERS opener (or 'paraphrased'). Remove that line + break.
    Never .replace/.lstrip on the body; never touches docs that open directly with content (no meta
    word / no ':' line). Returns (new_text, stripped?)."""
    if not text:
        return text, False
    # (a) first-paragraph meta preamble
    cut = text.find('\n\n')
    if 0 <= cut <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:cut].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if ((head.startswith(('###', '##', '**')) and has_word)
                or (low.startswith(DISTILL_OPENERS) and has_word)
                or (low.startswith('paraphrased') and head.endswith(':'))):
            return text[cut + 2:], True
    # (b) first-line meta preamble ending in ':' (single-'\n' forms)
    nl = text.find('\n')
    if 0 <= nl <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:nl].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if head.endswith(':') and has_word and (
                head.startswith(('###', '##', '**'))
                or low.startswith(DISTILL_OPENERS)
                or low.startswith('paraphrased')):
            return text[nl + 1:].lstrip('\n'), True
    return text, False


# ----------------------------------------------------------------------------- cross-pass pairing
def paired_wiki_status(wiki, distill):
    """Return the wiki `status` aligned row-for-row to distill order.

    Fast paths (assert, never silent): identical full doc_id arrays, or per-shard doc_id
    equality when both dicts carry shard metadata ('present','pos','offs'). On ANY mismatch or
    missing metadata, fall back to an EXPLICIT doc_id join (sort wiki by doc_id, searchsorted)
    and verify every distill doc_id is present — raise if not. Inputs are dicts with at least
    'doc_id' (int64) and 'status' (int8); optionally 'present' (shard ids), 'pos' (k->index),
    'offs' (row offsets)."""
    d_id = distill['doc_id']
    if d_id.size == 0:
        return np.empty(0, dtype=np.int8)
    w_id, w_st = wiki['doc_id'], wiki['status']

    # fast path A: identical full arrays (row-aligned across all shards)
    if w_id.shape == d_id.shape and np.array_equal(w_id, d_id):
        return w_st.astype(np.int8, copy=False)

    # fast path B: per-shard doc_id equality (distill may be a subset of wiki shards)
    if all(k in wiki for k in ('present', 'pos', 'offs')) and \
       all(k in distill for k in ('present', 'offs')):
        w_for_d = np.empty(d_id.size, dtype=np.int8)
        ok = True
        for i, k in enumerate(distill['present']):
            wi = wiki['pos'].get(k)
            if wi is None:
                ok = False
                break
            a = w_id[wiki['offs'][wi]:wiki['offs'][wi + 1]]
            b = d_id[distill['offs'][i]:distill['offs'][i + 1]]
            if a.shape != b.shape or not np.array_equal(a, b):
                ok = False
                break
            w_for_d[distill['offs'][i]:distill['offs'][i + 1]] = \
                w_st[wiki['offs'][wi]:wiki['offs'][wi + 1]]
        if ok:
            return w_for_d

    # fallback: explicit doc_id join (never silently proceed)
    order = np.argsort(w_id, kind='stable')
    wd = w_id[order]
    ws = w_st[order]
    idx = np.searchsorted(wd, d_id)
    idxc = np.clip(idx, 0, wd.size - 1)
    found = (idx < wd.size) & (wd[idxc] == d_id)
    if not bool(found.all()):
        raise RuntimeError(f'cross-pass join: {int((~found).sum())} distill doc_id(s) '
                           f'absent from wiki — cannot pair')
    return ws[idx].astype(np.int8, copy=False)


# ----------------------------------------------------------------------------- bucketed shuffle
def choose_buckets(specs, mem_bytes=None, inflate=4.0, log=print):
    """B = max(16, ceil(2*text_bytes_est / (0.55*mem))). text_bytes_est = on-disk parquet bytes
    * inflate (zstd->Arrow in-memory). mem capped at the 256 GB cpu limit."""
    disk = sum(os.path.getsize(p) for p, _ in specs)
    est = disk * inflate
    mem = min(mem_bytes or MEM_CAP_BYTES, MEM_CAP_BYTES)
    B = max(16, math.ceil(2.0 * est / (0.55 * mem)))
    log(f'  shuffle sizing: on-disk={disk/GIB:.1f}GiB est_in_mem={est/GIB:.1f}GiB '
        f'(x{inflate}) mem_cap={mem/GIB:.0f}GiB -> B={B} (peak ~{2*est/B/GIB:.1f}GiB)')
    return B, disk, est


def bucketed_shuffle(specs, load_fn, out_dir, tmp_dir, seed=42, rows_per_shard=500_000,
                     mem_bytes=None, inflate=4.0, log=print):
    """Memory-bounded two-pass document-level shuffle (deterministic, seed-based).

    specs: list of (path, aux) ; load_fn(spec) -> unified pa.Table (same schema for all).
    Pass 1 scatters each input shard's rows into B on-disk bucket files by a random bucket id
    (one rng over inputs in sorted order). Pass 2 reads each bucket, shuffles within it
    (rng seeded per bucket), and writes ~rows_per_shard part_NNNNN.parquet shards with a global
    counter; a carry buffer keeps shard sizes uniform across bucket boundaries. Returns
    (total_rows, n_shards, B)."""
    specs = sorted(specs, key=lambda s: str(s[0]))
    if not specs:
        raise RuntimeError('bucketed_shuffle: no inputs')
    out_dir = Path(out_dir); tmp_dir = Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    B, _, _ = choose_buckets(specs, mem_bytes=mem_bytes, inflate=inflate, log=log)

    # ---- Pass 1: scatter into B bucket files ----
    rng = np.random.default_rng(seed)
    schema = None
    writers = {}
    bucket_path = {b: str(tmp_dir / f'bucket_{b:05d}.parquet') for b in range(B)}
    total_rows = 0
    try:
        for si, spec in enumerate(specs):
            t = load_fn(spec)
            if schema is None:
                schema = t.schema
            n = t.num_rows
            total_rows += n
            bk = rng.integers(0, B, size=n)
            for b in np.unique(bk):
                sub = t.filter(pa.array(bk == b))
                w = writers.get(int(b))
                if w is None:
                    w = pq.ParquetWriter(bucket_path[int(b)], schema, compression='zstd')
                    writers[int(b)] = w
                w.write_table(sub)
            del t
            if (si + 1) % 50 == 0:
                log(f'  shuffle pass1: {si+1}/{len(specs)} shards scattered')
    finally:
        for w in writers.values():
            w.close()
    log(f'  shuffle pass1 done: {total_rows:,} rows scattered into {len(writers)}/{B} buckets')

    # ---- Pass 2: gather + within-bucket shuffle -> uniform 500k shards ----
    shard_idx = 0
    leftover = None
    written = 0
    for b in range(B):
        bp = bucket_path[b]
        if not os.path.exists(bp):
            continue
        t = pq.read_table(bp)
        if t.num_rows:
            perm = np.random.default_rng([seed, b]).permutation(t.num_rows)
            t = t.take(pa.array(perm))
            if leftover is not None and leftover.num_rows:
                t = pa.concat_tables([leftover, t])
            leftover = None
            off = 0
            n = t.num_rows
            while n - off >= rows_per_shard:
                atomic_write_table(t.slice(off, rows_per_shard),
                                   out_dir / f'part_{shard_idx:05d}.parquet')
                written += rows_per_shard
                shard_idx += 1
                off += rows_per_shard
            leftover = t.slice(off) if off < n else None
        os.unlink(bp)
        if (b + 1) % 50 == 0:
            log(f'  shuffle pass2: {b+1}/{B} buckets gathered, {shard_idx} shards written')
    if leftover is not None and leftover.num_rows:
        atomic_write_table(leftover, out_dir / f'part_{shard_idx:05d}.parquet')
        written += leftover.num_rows
        shard_idx += 1
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    if written != total_rows:
        raise RuntimeError(f'bucketed_shuffle: wrote {written} != scattered {total_rows}')
    log(f'  shuffle pass2 done: {shard_idx} shards, {written:,} rows')
    return total_rows, shard_idx, B


# ----------------------------------------------------------------------------- post-shuffle stats
def stream_shuffle_stats(out_dir, with_topic=False):
    """Stream the final shuffled shards reading ONLY light columns (doc_id, source_prompt,
    train_tokens [+ topic]); never holds the full corpus. Returns source distribution, doc_id
    overlap (shared-top 'original' vs the rest), the +1-BOS token budget, and optional per-topic
    token weights (keyed by (is_original, topic))."""
    cols = ['doc_id', 'source_prompt', 'train_tokens'] + (['topic'] if with_topic else [])
    src_docs = Counter(); src_tok = Counter()
    orig_ids = []; rew_ids = []
    topic_tok = {}
    total_rows = 0
    for p in sorted(glob.glob(str(Path(out_dir) / 'part_*.parquet'))):
        t = pq.read_table(p, columns=cols, use_threads=False)
        n = t.num_rows; total_rows += n
        sp = np.asarray(t.column('source_prompt').to_pylist(), dtype=object)
        did = t.column('doc_id').to_numpy(zero_copy_only=False).astype(np.int64)
        tok = t.column('train_tokens').to_numpy(zero_copy_only=False).astype(np.int64) + 1  # +BOS
        is_orig = sp == 'original'
        orig_ids.append(did[is_orig]); rew_ids.append(did[~is_orig])
        for s in np.unique(sp):
            m = sp == s
            src_docs[str(s)] += int(m.sum()); src_tok[str(s)] += int(tok[m].sum())
        if with_topic:
            tp = np.asarray(t.column('topic').to_pylist(), dtype=object)
            for is_o, mask in ((True, is_orig), (False, ~is_orig)):
                tpm = tp[mask]; tkm = tok[mask]
                for u in np.unique(tpm):
                    key = (is_o, str(u))
                    topic_tok[key] = topic_tok.get(key, 0) + int(tkm[tpm == u].sum())
    orig = np.unique(np.concatenate(orig_ids)) if orig_ids else np.empty(0, np.int64)
    rew = np.unique(np.concatenate(rew_ids)) if rew_ids else np.empty(0, np.int64)
    overlap = int(np.intersect1d(orig, rew).size)

    shared_top_tokens = int(src_tok.get('original', 0))
    rewritten_tokens = int(sum(v for k, v in src_tok.items() if k != 'original'))
    n_orig = int(src_docs.get('original', 0))
    n_rew = int(sum(v for k, v in src_docs.items() if k != 'original'))
    source_dist = {s: dict(docs=int(src_docs[s]), tokens=int(src_tok[s])) for s in src_docs}
    return dict(total_rows=total_rows, overlap=overlap,
                shared_top_docs=n_orig, shared_top_tokens=shared_top_tokens,
                rewritten_docs=n_rew, rewritten_tokens=rewritten_tokens,
                total_tokens=shared_top_tokens + rewritten_tokens,
                source_dist=source_dist, topic_tok=topic_tok)
