#!/usr/bin/env python
"""Export, verify and publish the three global Top-10B corpora to blab-jhu/KYS-Pre-Rewritten (additively).

Stages (each skipped when its record already exists):

  export     <out>/stage/raw_text/<setting>/part-000NN.parquet: the shuffled corpus of
             assemble_global_top10b.py rewritten in its exact post-shuffle order into 16 contiguous,
             near-equal files (ceil(rows / 16) rows each; files[rank::16] gives datatrove task i file i).
             Columns: orig_doc_id int64 (raw-pool position), doc_id int64 (scored-pool id, the selection key),
             source string ('selected'), text. The four existing settings keep their own schema.
  manifest   a copy of the Hub's manifest.json with, ADDED and nothing existing changed: settings.<setting>
             for each new setting, `settings_overview` (all settings: family, anchor, comparator) and
             `global_top10b` (rules, universe, digests, code commit).
  verify     the consumer path, unchanged: tools/kys_raw/tokenize_raw_text.sh on a local data root that holds
             that manifest, the tokenizer/ files and raw_text/<setting>; it checks every sha256 and requires
             the datatrove token total to equal settings.<setting>.expected_total_tokens exactly.
  upload     raw_text/<setting>/ (16 files), selection/<setting>/ (doc-id arrays), reports/, then manifest.json
             and README.md last, in separate commits.
  roundtrip  every uploaded file's LFS sha256 on the Hub equals the local sha256; manifest.json downloaded back
             equals the local one; the resulting Hub revision is recorded.

    python tools/kys_raw/publish_global_top10b.py --corpus <assemble out> --selection <select out> \
        --tokenizer <dir> --out <dir> --code-commit <sha> [--reports <dir>] [--stage export|verify|upload]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import RAW_SETTINGS  # noqa: E402

REPO_ID = 'blab-jhu/KYS-Pre-Rewritten'
REPO = Path(__file__).resolve().parents[2]
SETTINGS = [s for s, v in RAW_SETTINGS.items() if v['family'] == 'global_top10b']
N_FILES = 16
SCHEMA = pa.schema([('orig_doc_id', pa.int64()), ('doc_id', pa.int64()), ('source', pa.string()), ('text', pa.large_string())])
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer_report.json')


def sha256(p, buf=16 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def export(corpus: Path, out: Path, n_rows: int):
    parts = sorted((corpus / 'shuffled').glob('part_*.parquet'))
    out.mkdir(parents=True, exist_ok=True)
    per = math.ceil(n_rows / N_FILES)
    fi = in_file = written = 0
    writer, files = None, []
    for p in parts:
        t = pq.read_table(p, columns=['orig_doc_id', 'doc_id', 'source', 'text'])
        t = pa.table({'orig_doc_id': t['orig_doc_id'].cast(pa.int64()), 'doc_id': t['doc_id'].cast(pa.int64()),
                      'source': t['source'].cast(pa.string()), 'text': t['text'].cast(pa.large_string())}, schema=SCHEMA)
        off = 0
        while off < t.num_rows:
            if writer is None:
                files.append(out / f'part-{fi:05d}.parquet')
                writer = pq.ParquetWriter(files[-1], SCHEMA, compression='zstd')
            take = min(per - in_file, t.num_rows - off)
            writer.write_table(t.slice(off, take), row_group_size=50_000)
            off, in_file, written = off + take, in_file + take, written + take
            if in_file == per:
                writer.close()
                writer, in_file, fi = None, 0, fi + 1
    if writer is not None:
        writer.close()
    if written != n_rows or len(files) != N_FILES:
        sys.exit(f'export wrote {written:,} rows into {len(files)} files, expected {n_rows:,} in {N_FILES}')
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True, help='the grid tokenizer directory (sha256 must match the Hub manifest)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--code-commit', required=True, help='pushed nanotron commit holding the generation code')
    ap.add_argument('--reports', type=Path, help='directory of report files to publish under reports/')
    ap.add_argument('--export-verify', type=Path, help='verify_global_top10b.py output; required for --stage upload')
    ap.add_argument('--stage', choices=['export', 'verify', 'upload'], default='upload', help='run up to and including this stage')
    ap.add_argument('--settings', default=','.join(SETTINGS))
    args = ap.parse_args()
    settings = args.settings.split(',')
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    stage, vroot = args.out / 'stage', args.out / 'verify_root'
    sel = json.loads((args.selection / 'selection_manifest.json').read_text())

    # --- export -------------------------------------------------------------------------------------
    recs = {}
    for s in settings:
        asm = json.loads((args.corpus / s / '_raw_manifest.json').read_text())
        want = sel['settings'][s]
        if (asm['docs'], asm['shuffled_rows'], asm['train_tokens']) != (want['docs'], want['docs'], want['train_tokens']):
            sys.exit(f'{s}: assembled {asm["docs"]}/{asm["shuffled_rows"]}/{asm["train_tokens"]} != selection {want["docs"]}/{want["train_tokens"]}')
        done = stage / 'raw_text' / s / '_export.json'
        if done.is_file():
            recs[s] = json.loads(done.read_text())
        else:
            paths = export(args.corpus / s, stage / 'raw_text' / s, asm['docs'])
            recs[s] = {'rows': asm['docs'], 'files': {p.name: {'rows': pq.ParquetFile(p).metadata.num_rows, 'bytes': p.stat().st_size,
                                                               'sha256': sha256(p)} for p in paths}}
            done.write_text(json.dumps(recs[s], indent=2))
        print(f'{s}: exported {recs[s]["rows"]:,} rows in {len(recs[s]["files"])} files')

    # --- manifest (additive) ---------------------------------------------------------------------------
    rev = api.repo_info(REPO_ID, repo_type='dataset').sha
    hub_man = json.loads(Path(hf_hub_download(REPO_ID, 'manifest.json', repo_type='dataset', revision=rev,
                                              local_dir=args.out / 'hub_before')).read_text())
    man = json.loads(json.dumps(hub_man))
    for f in TOKENIZER_FILES:
        if sha256(args.tokenizer / f) != man['tokenizer']['sha256'][f]:
            sys.exit(f'{args.tokenizer / f} differs from the tokenizer recorded in the Hub manifest')
    man['settings_overview'] = {
        s: {'family': v['family'], 'contains_anchor': v['contains_anchor'], 'comparator': v['comparator'],
            'path': f'raw_text/{s}/', 'published': s in man['settings'] or s in settings}
        for s, v in RAW_SETTINGS.items()}
    man['global_top10b'] = {
        'description': ('Three anchor-free raw baselines for the 1.5B grid. Each setting is ONE global Top-10B selection over the '
                        'Quality-Base universe; the whole ~10B TRAIN-token corpus is selected by that score. No shared anchor, '
                        'no rewriting, no floors, quotas, variance terms or domain restrictions.'),
        'layout': ("raw_text/<setting>/part-000NN.parquet, 16 files; columns orig_doc_id (int64, raw-pool position), doc_id "
                   "(int64, scored-pool id = selection key), source ('selected'), text; rows in the exact post-shuffle order"),
        'scored_pool': sel['scored_pool'],
        'universe': sel['universe'] | {'definition': 'all scored rows minus the 50,000-doc validation holdout '
                                                     '(SeedSequence(42).spawn(8)[0]); the 5M analysis sample is not excluded'},
        'scorer_training_data': ('The ModernBERT score is a Ridge head fit on 50,427 Claude-labelled documents (49,998 from '
                                 'DCLM-RefinedWeb); every one of those was removed from the pool before scoring (the 50,838-row '
                                 'first-4000-character match removal), so 0 are in any corpus. The ~5M analysis sample was not '
                                 'used to fit the head and was NOT removed (4,997,471 docs remain); each new selection holds it at '
                                 'its pool rate (5.00%), as does Quality-Base. Kept for comparability with the original 1.5B '
                                 'universe. Benchmark contamination was not tested. See reports/GLOBAL_TOP10B_SELECTION_REPORT.md §2b.'),
        'rules': sel['rules'],
        'digest_conventions': sel['digest_conventions'],
        'deterministic_rerun_identical': sel['deterministic_rerun_identical'],
        'fasttext_reference': sel['reference'],
        'text_source': 'DCLM-RefinedWeb 100M reservoir pool, by orig_doc_id (shard = id // 500000, row = id % 500000); '
                       'each document re-tokenized at assembly and its length checked equal to tokens-llama2',
        'shuffle': {'function': 'tools/kys_raw/pp_io.py bucketed_shuffle (byte-identical copy of the pp_io.py every arm used)',
                    'seed': 42, 'rows_per_shard': 500_000, 'buckets': 16},
        'code': {'repo': 'https://github.com/imHuicongZhang/nanotron', 'branch': 'huicong-dev', 'commit': args.code_commit,
                 'scripts': ['tools/kys_raw/extract_scored_columns.py', 'tools/kys_raw/select_global_top10b.py',
                             'tools/kys_raw/report_global_top10b.py', 'tools/kys_raw/assemble_global_top10b.py',
                             'tools/kys_raw/verify_global_top10b.py', 'tools/kys_raw/publish_global_top10b.py']},
        'reports': 'reports/ (selection report, provenance report); selection/<setting>/ (doc-id arrays, see digest_conventions)',
    }
    for s in settings:
        w = sel['settings'][s]
        asm = json.loads((args.corpus / s / '_raw_manifest.json').read_text())
        man['settings'][s] = {
            'family': 'global_top10b', 'contains_anchor': False, 'comparator': 'quality_base',
            'role': f'global Top-10B by {w["score"]} over the Quality-Base universe (no anchor)',
            'score': w['score'], 'anchor_docs': 0, 'anchor_tokens': 0,
            'final_merged_docs': w['docs'], 'expected_total_tokens': w['train_tokens'], 'shards_after_tokenization': N_FILES,
            'overshoot': w['overshoot'], 'boundary_score': w['boundary_score'],
            'selection_order_sha256': w['selection_order_sha256'], 'docset_sha256': w['docset_sha256'],
            'orig_docset_sha256': w['orig_docset_sha256'], 'shuffled_parts_sha256': asm['shuffled_sha256'],
            'files': recs[s]['files'],
        }
    if args.export_verify:
        ev = json.loads(args.export_verify.read_text())
        for s in settings:
            v = ev[s]
            if not v['ok'] or v['docset_sha256'] != man['settings'][s]['docset_sha256']:
                sys.exit(f'{s}: export verification failed: {v["problems"]}')
            man['settings'][s]['file_order_sha256'] = v['file_order_sha256']
            man['settings'][s]['export_verification'] = {
                'script': 'tools/kys_raw/verify_global_top10b.py', 'rows': v['rows'], 'train_tokens': v['train_tokens'],
                'docset_equals_selection': True, 'duplicate_doc_ids': v['duplicate_doc_ids'],
                'order_equals_assembled_shuffle': v['order_equals_assembled_shuffle'],
                'text_sample_byte_equal_to_pool': f'{v["text_sample"]["rows_checked"] - v["text_sample"]["mismatches"]}/'
                                                  f'{v["text_sample"]["rows_checked"]}',
                'every_doc_retokenized_length_equals_tokens_llama2': True}
    elif args.stage == 'upload':
        sys.exit('--export-verify is required before uploading')
    mpath = args.out / 'manifest.json'
    mpath.write_text(json.dumps(man, indent=2) + '\n')
    # nothing that existed may change
    for k, v in hub_man.items():
        if k != 'settings' and man[k] != v:
            sys.exit(f'manifest key {k} changed')
    for s, v in hub_man['settings'].items():
        if s not in settings and man['settings'][s] != v:
            sys.exit(f'manifest settings.{s} changed')
    if args.stage == 'export':
        return

    # --- verify: the consumer's tokenization path, on a local data root --------------------------------
    (vroot / 'tokenizer').mkdir(parents=True, exist_ok=True)
    for f in TOKENIZER_FILES:
        shutil.copy2(args.tokenizer / f, vroot / 'tokenizer' / f)
    shutil.copy2(mpath, vroot / 'manifest.json')
    (vroot / 'raw_text').mkdir(exist_ok=True)
    verified = {}
    for s in settings:
        link = vroot / 'raw_text' / s
        if not link.exists():
            link.symlink_to((stage / 'raw_text' / s).resolve())
        rec = args.out / f'tokenized_{s}.json'
        if not rec.is_file():
            r = subprocess.run(['bash', str(REPO / 'tools/kys_raw/tokenize_raw_text.sh'), str(vroot), s],
                               env={**os.environ, 'PY': sys.executable}, capture_output=True, text=True)
            print(r.stdout[-2000:], r.stderr[-2000:])
            if r.returncode:
                sys.exit(f'{s}: tokenize_raw_text.sh failed; not publishing')
            meta = sorted((vroot / s / 'tokenized').glob('*.ds.metadata'))
            total = sum(int(m.read_text().splitlines()[1]) for m in meta)
            rec.write_text(json.dumps({'setting': s, 'shards': len(meta), 'datatrove_total_tokens': total,
                                       'expected_total_tokens': man['settings'][s]['expected_total_tokens'],
                                       'shard_tokens': {m.name: int(m.read_text().splitlines()[1]) for m in meta},
                                       'ds_bytes': {p.name: p.stat().st_size for p in sorted((vroot / s / 'tokenized').glob('*.ds'))}},
                                      indent=2))
        verified[s] = json.loads(rec.read_text())
        if verified[s]['datatrove_total_tokens'] != man['settings'][s]['expected_total_tokens'] or verified[s]['shards'] != N_FILES:
            sys.exit(f'{s}: tokenized total mismatch {verified[s]}')
        print(f'{s}: tokenized with the consumer script: {verified[s]["datatrove_total_tokens"]:,} tokens in 16 shards == expected')
    for s in settings:
        man['settings'][s]['tokenization_verified'] = {
            'script': 'tools/kys_raw/tokenize_raw_text.sh (datatrove 0.5.0, 16 tasks, </s> per document)',
            'datatrove_total_tokens': verified[s]['datatrove_total_tokens'], 'shards': verified[s]['shards'],
            'tokenized_ds_bytes_total': sum(verified[s]['ds_bytes'].values())}
    mpath.write_text(json.dumps(man, indent=2) + '\n')
    if args.stage == 'verify':
        return

    # --- upload ----------------------------------------------------------------------------------------
    for s in settings:
        api.upload_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(stage / 'raw_text' / s), path_in_repo=f'raw_text/{s}',
                          allow_patterns=['part-*.parquet'], commit_message=f'raw_text/{s}: 16 parquet files (global Top-10B, no anchor)')
        sd = args.selection / s
        api.upload_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(sd), path_in_repo=f'selection/{s}',
                          allow_patterns=['*.npy'], commit_message=f'selection/{s}: selected doc ids')
    api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=str(args.selection / 'selection_manifest.json'),
                    path_in_repo='selection/selection_manifest.json', commit_message='selection/selection_manifest.json')
    if args.reports:
        api.upload_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(args.reports), path_in_repo='reports',
                          commit_message='reports: provenance and selection reports')
    api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=str(Path(__file__).with_name('KYS-Pre-Rewritten.README.md')),
                    path_in_repo='README.md', commit_message='README.md: seven settings')
    info = api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=str(mpath), path_in_repo='manifest.json',
                           commit_message=f'manifest.json: add {", ".join(settings)} (additive)')

    # --- round trip ------------------------------------------------------------------------------------
    rev = api.repo_info(REPO_ID, repo_type='dataset').sha
    want = {f'raw_text/{s}/{n}': (r['sha256'], r['bytes']) for s in settings for n, r in recs[s]['files'].items()}
    for s in settings:
        for p in sorted((args.selection / s).glob('*.npy')):
            want[f'selection/{s}/{p.name}'] = (sha256(p), p.stat().st_size)
    remote = {f.path: f for f in api.get_paths_info(REPO_ID, list(want), repo_type='dataset', revision=rev)}
    bad = [p for p, (h, size) in want.items()
           if (f := remote.get(p)) is None or f.lfs is None or f.lfs.sha256 != h or f.size != size]
    got = json.loads(Path(hf_hub_download(REPO_ID, 'manifest.json', repo_type='dataset', revision=rev,
                                          local_dir=args.out / 'hub_after')).read_text())
    ok = not bad and got == man
    (args.out / 'published.json').write_text(json.dumps({'repo': REPO_ID, 'revision': rev, 'manifest_commit': str(info),
                                                         'files_checked': len(want),
                                                         'lfs_sha256_mismatches': bad, 'manifest_roundtrip_equal': got == man,
                                                         'ok': ok, 'code_commit': args.code_commit}, indent=2))
    print(f'revision {rev}: {len(want)} files checked, mismatches {bad}, manifest equal {got == man}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
