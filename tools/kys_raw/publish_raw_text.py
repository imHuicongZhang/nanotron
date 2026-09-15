#!/usr/bin/env python
"""Publish one raw-selected corpus to blab-jhu/KYS-Pre-Rewritten as raw text, with checks.

    python tools/kys_raw/publish_raw_text.py --kys-root <dir> --setting raw_random --code-commit <sha> [--dry-run]

Refuses unless:
  * verify/verify_<setting>.json lists no failed leakage checks;
  * the assembled corpus holds exactly anchor docs + subsampled strategy docs, and its re-tokenized
    total (tokens + 1 per document) equals anchor tokens + subsampled strategy tokens EXACTLY;
  * the three 1.5B init checkpoints (seeds 42/43/44) and their hash manifests are on the Hub.

Export: the assembled corpus, already shuffled with pp_io.bucketed_shuffle (seed 42), is rewritten
in its exact post-shuffle document order into 16 contiguous, near-equal parquet files

    raw_text/<setting>/part-00000.parquet ... part-00015.parquet
        orig_doc_id  int64    position in the 100M DCLM-RefinedWeb pool
        source       string   'anchor' | 'strategy'
        text         string   original text read from the pool

Exactly 16 files so that datatrove's file-to-task assignment (files[rank::16]) gives task i file i:
tokenizing with 16 tasks (tools/kys_raw/tokenize_raw_text.sh) yields shards 00000..00015 whose
concatenation is the post-shuffle order.

Then manifest.json (all settings published so far) and README.md are refreshed, raw_text/<setting>/
is uploaded, and one parquet file is downloaded back and its sha256 compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = 'blab-jhu/KYS-Pre-Rewritten'
COUNTERPART = {
    'raw_diversity_oriented': 'diversity_oriented',
    'raw_disagreement_aware': 'disagreement_aware',
    'raw_random': 'wrap_inspired',
    'raw_rewire_inspired': 'rewire_inspired',
}
ROLE = {
    'raw_diversity_oriented': 'no-rewrite control for diversity_oriented',
    'raw_disagreement_aware': 'no-rewrite control for disagreement_aware',
    'raw_random': 'no-selection reference (uniform random source sample) and no-rewrite control for wrap_inspired',
    'raw_rewire_inspired': "no-rewrite control for rewire_inspired: source documents of the 5B kept by REWIRE's post-rewrite filter",
}
N_FILES = 16
ANCHOR_DOCS, ANCHOR_TOKENS = 4_120_164, 5_000_002_332
SCHEMA = pa.schema([('orig_doc_id', pa.int64()), ('source', pa.string()), ('text', pa.large_string())])
INIT_REPO = 'wytro/Know-Your-Sources-init'
INIT_ROLLING = {42: '2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba',
                43: '78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3',
                44: 'a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247'}
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer_report.json')


def sha256(path: Path, buf=16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def export(corpus: Path, out: Path, n_rows: int):
    """Rewrite shuffled/part_*.parquet in order into N_FILES contiguous files."""
    parts = sorted((corpus / 'shuffled').glob('part_*.parquet'))
    if not parts:
        sys.exit(f'no shuffled parquet under {corpus}')
    out.mkdir(parents=True, exist_ok=True)
    per = math.ceil(n_rows / N_FILES)
    fi, in_file, written, writer = 0, 0, 0, None
    files = []
    for p in parts:
        t = pq.read_table(p, columns=['orig_doc_id', 'source', 'text'])
        src = pa.array(['anchor' if s == 'anchor' else 'strategy' for s in t.column('source').to_pylist()], pa.string())
        t = pa.table({'orig_doc_id': t.column('orig_doc_id').cast(pa.int64()), 'source': src,
                      'text': t.column('text').cast(pa.large_string())}, schema=SCHEMA)
        off = 0
        while off < t.num_rows:
            if writer is None:
                path = out / f'part-{fi:05d}.parquet'
                writer = pq.ParquetWriter(path, SCHEMA, compression='zstd')
                files.append(path)
            take = min(per - in_file, t.num_rows - off)
            writer.write_table(t.slice(off, take), row_group_size=50_000)
            off += take
            in_file += take
            written += take
            if in_file == per:
                writer.close()
                writer, in_file, fi = None, 0, fi + 1
    if writer is not None:
        writer.close()
    if written != n_rows or len(files) != N_FILES:
        sys.exit(f'export wrote {written:,} rows into {len(files)} files, expected {n_rows:,} rows in {N_FILES}')
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kys-root', type=Path, required=True)
    ap.add_argument('--setting', required=True, choices=sorted(COUNTERPART))
    ap.add_argument('--code-commit', required=True)
    ap.add_argument('--readme', type=Path, default=Path(__file__).with_name('KYS-Pre-Rewritten.README.md'))
    ap.add_argument('--dry-run', action='store_true', help='export and write the manifest, but do not upload')
    args = ap.parse_args()
    K, s = args.kys_root, args.setting
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    # --- gates -------------------------------------------------------------------------------------
    v = json.loads((K / 'verify' / f'verify_{s}.json').read_text())
    if v.get('failed') != []:
        sys.exit(f'{s}: leakage checks failed {v.get("failed")}; not publishing')
    src = json.loads((K / 'raw_sources' / 'manifest.json').read_text())
    a, st = src['anchor'], src['settings'][s]
    if (a['docs'], a['tokens']) != (ANCHOR_DOCS, ANCHOR_TOKENS):
        sys.exit(f'anchor record unexpected: {a}')
    asm = json.loads((K / 'raw_corpus' / s / '_raw_manifest.json').read_text())
    exp_docs, exp_tokens = ANCHOR_DOCS + st['docs_after'], ANCHOR_TOKENS + st['tokens_after']
    if (asm['docs'], asm['shuffled_rows'], asm['tokens_plus1']) != (exp_docs, exp_docs, exp_tokens):
        sys.exit(f'{s}: assembled docs/rows/tokens {asm["docs"]:,}/{asm["shuffled_rows"]:,}/{asm["tokens_plus1"]:,} '
                 f'!= expected {exp_docs:,}/{exp_docs:,}/{exp_tokens:,}')
    files = api.list_repo_files(INIT_REPO)
    init = {}
    for seed, rolling in INIT_ROLLING.items():
        d, h = f'_init_1.5B_seed{seed}/0/', f'init_1.5B_seed{seed}.hash.json'
        if not any(f.startswith(d) for f in files) or h not in files:
            sys.exit(f'init checkpoint for seed {seed} missing on {INIT_REPO} ({d}, {h}); stopping')
        init[str(seed)] = {'path': d, 'hash_manifest': h, 'parameters': 1_504_299_008, 'rolling_sha256': rolling}
    hub = set(api.list_repo_files(REPO_ID, repo_type='dataset'))
    missing_tok = [f for f in TOKENIZER_FILES if f'tokenizer/{f}' not in hub]
    if missing_tok and not args.dry_run:
        sys.exit(f'tokenizer/ incomplete on {REPO_ID} (missing {missing_tok}); upload the grid tokenizer directory first')

    # --- export ------------------------------------------------------------------------------------
    stage = K / 'hf_stage'
    out = stage / 'raw_text' / s
    done = out / '_export.json'
    if done.is_file() and json.loads(done.read_text()).get('rows') == exp_docs:
        rec = json.loads(done.read_text())
        print(f'{s}: export already done')
    else:
        paths = export(K / 'raw_corpus' / s, out, exp_docs)
        rec = {'rows': exp_docs, 'files': {p.name: {'rows': pq.ParquetFile(p).metadata.num_rows, 'bytes': p.stat().st_size,
                                                    'sha256': sha256(p)} for p in paths}}
        done.write_text(json.dumps(rec, indent=2))
        print(f'{s}: exported {exp_docs:,} rows into {len(paths)} files')

    # --- data upload (per setting; independent of other settings) -----------------------------------
    if not args.dry_run:
        api.upload_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(out), path_in_repo=f'raw_text/{s}',
                          allow_patterns=['part-*.parquet'], commit_message=f'raw_text/{s}: 16 parquet files')
        print(f'{s}: uploaded raw_text/{s}/')

    # --- manifest: read-modify-write-upload under a lock shared by jobs on every node --------------
    mpath = stage / 'manifest.json'
    lock = ManifestLock(stage / '.manifest.lockdir')
    lock.acquire()
    try:
        _update_and_upload_manifest(api, args, K, s, stage, mpath, src, a, st, exp_docs, exp_tokens, init, rec, v)
    finally:
        lock.release()
    if args.dry_run:
        print('dry run: not uploading')
        return

    # --- round trip ----------------------------------------------------------------------------------
    name = 'part-00000.parquet'
    with tempfile.TemporaryDirectory(dir=K) as dl:
        got = sha256(Path(hf_hub_download(REPO_ID, f'raw_text/{s}/{name}', repo_type='dataset', local_dir=dl)))
    want = rec['files'][name]['sha256']
    ok = got == want
    (stage / f'published_{s}.json').write_text(json.dumps({'setting': s, 'hf_path': f'https://huggingface.co/datasets/{REPO_ID}/tree/main/raw_text/{s}',
                                                           'roundtrip_file': name, 'sha256_local': want, 'sha256_hub': got,
                                                           'roundtrip_ok': ok, 'code_commit': args.code_commit}, indent=2))
    print(f'{s}: round-trip raw_text/{s}/{name}: {"OK" if ok else "MISMATCH"} ({got})')
    if not ok:
        sys.exit(1)


class ManifestLock:
    """mkdir-based lock: atomic on the shared filesystem across nodes, unlike flock on some network FS."""

    def __init__(self, path: Path, stale_s: int = 1800):
        self.path, self.stale_s = path, stale_s

    def acquire(self):
        import os
        import time
        while True:
            try:
                os.mkdir(self.path)
                (self.path / 'owner').write_text(f'{os.uname().nodename} pid {os.getpid()} {time.ctime()}\n')
                return
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        print(f'breaking stale manifest lock {self.path}')
                        for f in self.path.iterdir():
                            f.unlink()
                        self.path.rmdir()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(5)

    def release(self):
        for f in self.path.glob('*'):
            f.unlink()
        self.path.rmdir()


def _update_and_upload_manifest(api, args, K, s, stage, mpath, src, a, st, exp_docs, exp_tokens, init, rec, v):
    tok = K / 'nanotron_tokenized' / 'tokenizer'
    manifest =json.loads(mpath.read_text()) if mpath.is_file() else {}
    manifest.update({
        'repo': REPO_ID,
        'description': 'Raw text of the four raw-selected (unrewritten) baseline corpora for the Know-Your-Sources 1.5B grid.',
        'layout': ('raw_text/<setting>/part-000NN.parquet, 16 files per setting; columns orig_doc_id (int64), '
                   "source ('anchor' | 'strategy'), text; rows in the exact post-shuffle document order, anchor merged in"),
        'code': {'repo': 'https://github.com/imHuicongZhang/nanotron', 'branch': 'huicong-dev', 'commit': args.code_commit},
        'source_selection': ("unique orig_doc_id of every non-anchor row (source_prompt != 'original') of the published "
                             'rewritten arm wytro/Know-Your-Sources/<counterpart>/*.parquet, deduplicated across prompts'),
        'text_source': ('DCLM-RefinedWeb 100M reservoir sample (200 parquet x 500,000 rows); text read by position '
                        'orig_doc_id: shard = id // 500000, row = id % 500000. No rewritten text is used.'),
        'token_convention': 'len(llama2_tokenizer(text, add_special_tokens=False)) + 1 per document (the +1 is the </s> appended at tokenization)',
        'budget_tokens': src['budget_tokens'],
        'subsampling': {'seed': src['seed'], 'rule': src['subsample_rule']},
        'anchor': {'docs': ANCHOR_DOCS, 'tokens': ANCHOR_TOKENS,
                   'source': "rows with source_prompt == 'original' in the published arms (identical ids in all four); text from the pool",
                   'identical_across_arms': a['identical_across']},
        'shuffle': {'function': 'pp_io.bucketed_shuffle (projects/rewrite/10_postprocess/pp_io.py), the shuffle every published arm used',
                    'seed': 42, 'scope': 'anchor + strategy documents together'},
        'tokenizer': {'name': 'llama-2 (llama2-unsloth), vocab 32000, 2-byte tokens', 'repo': REPO_ID, 'repo_type': 'dataset',
                      'path': 'tokenizer/',
                      'origin': 'tokenizer/ of wytro/Know-Your-Sources-tokenized, copied unchanged: the directory the grid was tokenized with',
                      'sha256': {f: sha256(tok / f) for f in TOKENIZER_FILES},
                      'tokenizer.json_sha256': sha256(tok / 'tokenizer.json')},
        'tokenization': ('done by the consumer: tools/kys_raw/tokenize_raw_text.sh (datatrove 0.5.0, </s> per document, '
                         '16 tasks, no shuffling) -> 16 shards whose concatenation is the file order; see RUNBOOK.md'),
        'init_checkpoints': {'repo': INIT_REPO, 'repo_type': 'model', 'seeds': init,
                             'verify_with': 'tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed<S>/0 --check <init_root>/init_1.5B_seed<S>.hash.json'},
    })
    c3 = v['check3_hashes']
    manifest.setdefault('settings', {})[s] = {
        'rewritten_counterpart': COUNTERPART[s],
        'role': ROLE[s],
        'source_docs_dedup': st['source_docs_dedup'],
        'raw_tokens_before_subsampling': st['raw_tokens_before'],
        'docs_after_subsampling': st['docs_after'],
        'tokens_after_subsampling': st['tokens_after'],
        'subsampling_seed': src['seed'],
        'anchor_docs': ANCHOR_DOCS,
        'anchor_tokens': ANCHOR_TOKENS,
        'final_merged_docs': exp_docs,
        'expected_total_tokens': exp_tokens,
        'shards_after_tokenization': N_FILES,
        'leakage_checks': {'passed': ['provenance', 'anchor', 'hashes', 'style', 'lengths'],
                           'hash_spot_check': f"{c3['exact_match_raw_pool']}/{c3['sampled']} equal the raw pool, "
                                              f"{c3['match_rewritten']} equal a rewrite"},
        'files': rec['files'],
    }
    tmp = mpath.with_name('.manifest.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=2) + '\n')
    import os
    os.replace(tmp, mpath)
    (stage / 'README.md').write_text(args.readme.read_text())
    print(f'{s}: manifest updated ({mpath})')
    if args.dry_run:
        return
    api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=str(mpath), path_in_repo='manifest.json',
                    commit_message=f'manifest.json: add {s}')
    api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=str(stage / 'README.md'), path_in_repo='README.md',
                    commit_message='README.md')
    print(f'{s}: manifest.json and README.md uploaded')


if __name__ == '__main__':
    main()
