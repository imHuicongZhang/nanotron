#!/usr/bin/env python
"""Leakage checks for the raw-selected baseline corpora. Run BEFORE tokenization; all must pass.

  check 1  provenance : static scan of the build code — text is read only from the 100M pool, and
                        no published or rewrite text column is ever read to build a corpus
  check 2  anchor     : ALL 4,120,164 anchor rows of the corpus — id set == published anchor ids,
                        sha256(text) == pool text at orig_doc_id for every row, re-tokenized total
                        (tokens + 1) == 5,000,002,332 exactly, and every text also equals the
                        published anchor text (so the published anchor was raw too). With
                        --anchor-ref the anchor rows are compared by (id, sha256) digest to a
                        corpus already fully checked, instead of re-reading everything.
  check 3  hashes     : 200 sampled strategy-half docs == pool text (sha256) and != every rewrite of
                        the same orig_doc_id in the published arm
  check 4  style      : markdown heading at start / markdown list markers / no URL or web
                        boilerplate — corpus sample vs 200 random pool docs vs 200 rewritten docs
  check 5  lengths    : mean / median tokens (+1 convention) — full strategy half of the corpus,
                        full rewritten half of the arm, and the 200-doc samples

Exit 1 if check 1, 2 or 3 fails, or if check 4/5 contradict raw text (corpus closer to the
rewritten arm than to the pool). Everything is written to <out>/verify_<setting>.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ARM = {'raw_diversity_oriented': 'diversity_oriented', 'raw_disagreement_aware': 'disagreement_aware',
       'raw_random': 'wrap_inspired', 'raw_rewire_inspired': 'rewire_inspired'}
POOL_FILE = 'dclm_refinedweb_sample_{:05d}.parquet'
ROWS = 500_000
N = 200
ANCHOR_DOCS, ANCHOR_TOKENS = 4_120_164, 5_000_002_332
TOOLS = Path(__file__).resolve().parent

HEADING = re.compile(r'^\s*#{1,6}\s')
LIST = re.compile(r'^\s*(?:[-*+]|\d{1,3}[.)])\s+\S', re.M)
WEBBY = re.compile(r'https?://|www\.|©|\bcopyright\b|all rights reserved|privacy policy|cookies?\b|'
                   r'\blog ?in\b|\bsign (?:in|up)\b|\bsubscribe\b|click here|terms of (?:use|service)|'
                   r'\bposted (?:by|on)\b|\breply\b|\bshare this\b|\bnewsletter\b', re.I)


def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def pool_column(pool, shard):
    return pq.read_table(Path(pool) / POOL_FILE.format(shard), columns=['text']).column('text')


def corpus_files(corpus):
    return sorted((corpus / 'shuffled').glob('*.parquet')) or sorted((corpus / '_by_pool_shard').glob('*.parquet'))


# ------------------------------------------------------------------------------ check 1
READ_CALL = re.compile(r'\b(?:pq\.read_table|pq\.ParquetFile|read_row_group|ParquetReader)\s*\(')


def check_provenance():
    """Every parquet read in the two build files, classified by the line that performs it.

    Rule: a read that requests the `text` column must name the 100M pool on that same line
    (`pool` / `POOL_FILE`); a read of the published arm (`parquet_root`) must not request text.
    The corpus text is written from exactly one read site, assemble_raw_corpus.pool_texts().
    """
    errs, sites = [], []
    for name in ('build_raw_sources.py', 'assemble_raw_corpus.py'):
        for i, line in enumerate((TOOLS / name).read_text().splitlines(), 1):
            if not READ_CALL.search(line) or line.lstrip().startswith('#'):
                continue
            cols = re.search(r"columns=\[([^\]]*)\]", line)
            reads_text = bool(cols and "'text'" in cols.group(1))
            from_pool = 'pool' in line or 'POOL_FILE' in line
            from_published = 'parquet_root' in line
            # the assembler's shuffle re-reads the shards it staged itself from pool text
            from_staged = name == 'assemble_raw_corpus.py' and '# staged' in line
            sites.append({'file': name, 'line': i, 'columns': cols.group(1) if cols else 'ALL',
                          'source': 'pool' if from_pool else 'staged' if from_staged
                          else 'published' if from_published else 'other'})
            if from_staged:
                continue
            if not cols and not from_pool:
                errs.append(f'{name}:{i} reads all columns from a non-pool parquet: {line.strip()}')
            if reads_text and not from_pool:
                errs.append(f'{name}:{i} reads a text column from a non-pool source: {line.strip()}')
    asm_text = [s for s in sites if s['file'] == 'assemble_raw_corpus.py' and "'text'" in s['columns']]
    if len(asm_text) != 1:
        errs.append(f'assemble_raw_corpus.py has {len(asm_text)} text read sites, expected exactly 1 (pool_texts)')
    return errs, {'parquet_read_sites': sites}


# ------------------------------------------------------------------------------ check 2
def anchor_rows(corpus):
    """(orig_doc_id, text) of every anchor row in the assembled corpus, grouped for pool reads."""
    ids, texts = [], []
    for f in corpus_files(corpus):
        t = pq.read_table(f, columns=['orig_doc_id', 'text', 'source'])
        m = np.asarray(t.column('source').to_pylist(), dtype=object) == 'anchor'
        ids.append(t.column('orig_doc_id').to_numpy()[m])
        texts += t.column('text').filter(m).to_pylist()
    ids = np.concatenate(ids)
    return ids, texts


def check_anchor(corpus, pool, parquet_root, arm, sources, tokenizer, anchor_ref):
    errs, out = [], {}
    ids, texts = anchor_rows(corpus)
    order = np.argsort(ids)
    ids = ids[order]
    hashes = [sha(texts[i]) for i in order]
    del texts
    digest = hashlib.sha256()
    for o, h in zip(ids, hashes):
        digest.update(f'{int(o)}:{h}\n'.encode())
    out['anchor_docs'] = int(ids.size)
    out['anchor_id_hash_digest'] = digest.hexdigest()
    if ids.size != ANCHOR_DOCS:
        errs.append(f'anchor has {ids.size:,} docs, expected {ANCHOR_DOCS:,}')
    pub = np.load(Path(sources) / 'anchor_doc_ids.npy')
    if not np.array_equal(ids, np.sort(pub)):
        errs.append('anchor doc id set differs from the published source_prompt == original ids')

    if anchor_ref:
        ref = json.loads(Path(anchor_ref).read_text())['check2_anchor']
        out['compared_to'] = anchor_ref
        if ref['anchor_id_hash_digest'] != out['anchor_id_hash_digest']:
            errs.append('anchor (id, sha256) digest differs from the fully verified reference corpus')
        out.update({k: ref[k] for k in ('pool_mismatches', 'published_text_mismatches', 'anchor_tokens_plus1')})
        return errs, out

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path(tokenizer) / 'tokenizer.json'))
    pos = {int(o): i for i, o in enumerate(ids)}
    pool_bad, tok_total, examples = 0, 0, []
    for shard in np.unique(ids // ROWS):
        col = pool_column(pool, int(shard))
        sel = ids[(ids // ROWS) == shard]
        ptxt = col.take(sel % ROWS).to_pylist()
        for i in range(0, len(ptxt), 2_000):   # chunked to bound memory
            tok_total += sum(len(e.ids) + 1 for e in tk.encode_batch(ptxt[i:i + 2_000], add_special_tokens=False))
        for o, t in zip(sel, ptxt):
            if sha(t) != hashes[pos[int(o)]]:
                pool_bad += 1
                if len(examples) < 10:
                    examples.append(int(o))
    pub_bad = 0
    for f in sorted(glob.glob(str(Path(parquet_root) / arm / '*.parquet'))):
        t = pq.read_table(f, columns=['orig_doc_id', 'source_prompt', 'text'])   # verification only
        m = np.asarray(t.column('source_prompt').to_pylist(), dtype=object) == 'original'
        for o, txt in zip(t.column('orig_doc_id').to_numpy()[m], t.column('text').filter(m).to_pylist()):
            i = pos.get(int(o))
            if i is None or sha(txt) != hashes[i]:
                pub_bad += 1
    out.update(pool_mismatches=pool_bad, pool_mismatch_examples=examples,
               published_text_mismatches=pub_bad, anchor_tokens_plus1=tok_total)
    if pool_bad:
        errs.append(f'{pool_bad:,} anchor docs differ from the pool text at their position')
    if pub_bad:
        errs.append(f'{pub_bad:,} anchor docs differ from the published anchor text')
    if tok_total != ANCHOR_TOKENS:
        errs.append(f'anchor re-tokenized total {tok_total:,} != {ANCHOR_TOKENS:,}')
    return errs, out


# ------------------------------------------------------------------------------ checks 3-5
def style(texts):
    n = len(texts)
    return {'starts_with_md_heading': sum(bool(HEADING.match(t)) for t in texts) / n,
            'has_md_list_markers': sum(bool(LIST.search(t)) for t in texts) / n,
            'no_url_or_web_boilerplate': sum(not WEBBY.search(t) for t in texts) / n}


def lengths(tok):
    tok = np.asarray(tok, dtype=np.int64)
    return {'n': int(tok.size), 'mean_tokens': round(float(tok.mean()), 1), 'median_tokens': float(np.median(tok))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=sorted(ARM))
    ap.add_argument('--corpus', type=Path, required=True, help='<assemble out>/<setting>')
    ap.add_argument('--sources', type=Path, required=True, help='build_raw_sources.py output dir')
    ap.add_argument('--parquet-root', type=Path, required=True, help='published wytro/Know-Your-Sources snapshot')
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--anchor-ref', type=Path, help='verify_<setting>.json of a corpus whose anchor was fully checked')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    ntok = lambda ts: [len(e.ids) + 1 for e in tk.encode_batch(ts, add_special_tokens=False)]
    rng = np.random.default_rng(20260915)
    arm = ARM[args.setting]
    report, failed = {'setting': args.setting, 'counterpart': arm}, []

    errs, facts = check_provenance()
    report['check1_provenance'] = {**facts, 'errors': errs}
    failed += ['check1'] if errs else []

    errs, a = check_anchor(args.corpus, args.pool, args.parquet_root, arm, args.sources, args.tokenizer, args.anchor_ref)
    report['check2_anchor'] = {**a, 'errors': errs}
    failed += ['check2'] if errs else []

    # corpus strategy half: full-population lengths, and a 200-doc sample
    files = corpus_files(args.corpus)
    strat_tok, sample = [], []
    for f in files:
        t = pq.read_table(f, columns=['source', 'train_tokens'])
        m = np.asarray(t.column('source').to_pylist(), dtype=object) == 'raw_selected'
        strat_tok.append(t.column('train_tokens').to_numpy()[m].astype(np.int64) + 1)
    # Draw files with replacement until exactly N distinct documents are collected, so the sample
    # size never depends on how many shards the corpus happens to have.
    per_file = max(1, N // 10)
    seen = set()
    for _ in range(10 * N):
        if len(sample) >= N:
            break
        f = files[int(rng.integers(len(files)))]
        t = pq.read_table(f, columns=['orig_doc_id', 'text', 'source'])
        oid = t.column('orig_doc_id').to_numpy()
        cand = [int(r) for r in np.flatnonzero(np.asarray(t.column('source').to_pylist(), dtype=object) == 'raw_selected')
                if int(oid[r]) not in seen]
        if not cand:
            continue
        for r in rng.choice(cand, min(per_file, N - len(sample), len(cand)), replace=False):
            seen.add(int(oid[int(r)]))
            sample.append((int(oid[int(r)]), t.column('text')[int(r)].as_py()))
    ids = np.array([o for o, _ in sample])

    # rewritten arm: rewrites of the sampled ids, full rewritten-half lengths, 200 rewritten docs
    arm_files = sorted(glob.glob(str(args.parquet_root / arm / '*.parquet')))
    rewrites, rw_tok = {int(o): [] for o in ids}, []
    for f in arm_files:
        t = pq.read_table(f, columns=['orig_doc_id', 'source_prompt', 'train_tokens'])
        m = np.asarray(t.column('source_prompt').to_pylist(), dtype=object) != 'original'
        rw_tok.append(t.column('train_tokens').to_numpy()[m].astype(np.int64) + 1)
        hit = np.flatnonzero(m & np.isin(t.column('orig_doc_id').to_numpy(), ids))
        if hit.size:
            txt = pq.read_table(f, columns=['text']).column('text')           # verification only
            for r in hit:
                rewrites[int(t.column('orig_doc_id')[int(r)].as_py())].append(txt[int(r)].as_py())
    rw_txt = []
    for _ in range(10 * N):
        if len(rw_txt) >= N:
            break
        f = arm_files[int(rng.integers(len(arm_files)))]
        t = pq.read_table(f, columns=['source_prompt', 'text'])
        cand = np.flatnonzero(np.asarray(t.column('source_prompt').to_pylist(), dtype=object) != 'original')
        rw_txt += [t.column('text')[int(r)].as_py() for r in rng.choice(cand, min(per_file, N - len(rw_txt)), replace=False)]

    # check 3
    raw_match = rw_match = no_rw = 0
    bad = []
    by_shard = {}
    for o, _ in sample:
        by_shard.setdefault(o // ROWS, []).append(o)
    pool_txt_for = {}
    for s, os_ in by_shard.items():
        col = pool_column(args.pool, s)
        for o in os_:
            pool_txt_for[o] = col[o % ROWS].as_py()
    for o, txt in sample:
        h = sha(txt)
        if h == sha(pool_txt_for[o]):
            raw_match += 1
        else:
            bad.append({'orig_doc_id': o, 'failure': 'differs_from_pool'})
        if not rewrites[o]:
            no_rw += 1
            bad.append({'orig_doc_id': o, 'failure': 'no_rewrite_in_arm'})
        if any(h == sha(r) for r in rewrites[o]):
            rw_match += 1
            bad.append({'orig_doc_id': o, 'failure': 'equals_a_rewrite'})
    report['check3_hashes'] = {'sampled': len(sample), 'exact_match_raw_pool': raw_match,
                               'match_rewritten': rw_match, 'sampled_with_no_rewrite_in_arm': no_rw,
                               'failures': bad[:20]}
    if len(sample) != N or raw_match != N or rw_match or no_rw:
        failed.append('check3')

    # checks 4-5 comparison sample from the pool
    pool_ids = np.sort(rng.choice(100_000_000, N, replace=False))
    pool_txt = []
    for s in np.unique(pool_ids // ROWS):
        col = pool_column(args.pool, int(s))
        pool_txt += col.take(pool_ids[(pool_ids // ROWS) == s] % ROWS).to_pylist()
    corpus_txt = [t for _, t in sample]
    st = {'raw_corpus_sample': style(corpus_txt), 'raw_pool_sample': style(pool_txt), 'rewritten_arm_sample': style(rw_txt)}
    report['check4_style'] = st
    dist = lambda a, b: sum(abs(st[a][k] - st[b][k]) for k in st[a])
    report['check4_style']['L1_corpus_to_pool'] = round(dist('raw_corpus_sample', 'raw_pool_sample'), 4)
    report['check4_style']['L1_corpus_to_rewritten'] = round(dist('raw_corpus_sample', 'rewritten_arm_sample'), 4)
    if report['check4_style']['L1_corpus_to_pool'] >= report['check4_style']['L1_corpus_to_rewritten']:
        failed.append('check4')

    ln = {'raw_corpus_full_strategy_half': lengths(np.concatenate(strat_tok)),
          'rewritten_arm_full_rewritten_half': lengths(np.concatenate(rw_tok)),
          'raw_corpus_sample': lengths(ntok(corpus_txt)),
          'raw_pool_sample': lengths(ntok(pool_txt)),
          'rewritten_arm_sample': lengths(ntok(rw_txt))}
    report['check5_lengths'] = ln
    if ln['raw_corpus_full_strategy_half']['mean_tokens'] <= ln['rewritten_arm_full_rewritten_half']['mean_tokens']:
        failed.append('check5')

    report['failed'] = failed
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'verify_{args.setting}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
