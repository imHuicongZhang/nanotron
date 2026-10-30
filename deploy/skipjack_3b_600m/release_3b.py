#!/usr/bin/env python
"""Publish the four-setting 3B handoff: code branch on GitHub, data on the private HF dataset. CPU only; never trains.

    python deploy/skipjack_3b_600m/release_3b.py run [--through <stage>]    # what release_3b.sbatch runs
    python deploy/skipjack_3b_600m/release_3b.py status

Stages run in order; each writes <reports>/release_3b/<stage>.json and is skipped on a rerun once that record says ok,
so a failed or timed-out job is resumed by resubmitting it (upload_large_folder resumes file by file).

  gate            tokenization verified for all four settings (epoch_stats.json), both assembly validations without
                  problems, and from this checkout: assert_invariants_3b --grid --corpus, test_quality_base_3b,
                  test_lr_resume_3b, test_portable_3b
  code            this checkout must be clean and on BRANCH; secret scan of the branch diff; push BRANCH to origin as a
                  new branch or a fast-forward (never force); fresh clone of the pushed branch, HEAD == pushed commit,
                  and the config/invariant/LR-resume/portable tests rerun from that clone
  stage           hardlink tree (contents byte-identical, nothing rewritten) of the four shuffled corpora, tokenized
                  shards, tokenizer and provenance reports; sha256 of every file; cross-checks against the validation
                  reports, the pinned tokenizer, rows/parts/tokens/documents; RELEASE_MANIFEST, HANDOFF_RELEASE, README
  remote_check    dataset repo is private; every remote file is either one of ours with the same size and sha256
                  (reused) or the job stops (nothing is overwritten)
  upload          upload_large_folder (private), then RELEASE_MANIFEST.json + HANDOFF_RELEASE.json + README.md in one commit
  verify_remote   exhaustive: every expected path exists remotely with the same size and sha256 (LFS sha256, or the file
                  downloaded and hashed), no unexpected files, still private; records the content revision
  fresh_download  sampled, at that revision into an empty directory and cache: tokenizer files (sha256, encode check),
                  first and last parquet part of each setting (sha256, rows, text), one tokenized shard triple of each
                  setting (sha256, tokens, documents)
  receipt         UPLOAD_RECEIPT.json (local) and receipts/UPLOAD_RECEIPT.json on the Hub describing the content revision;
                  the final revision must differ from it by that one file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry_3b as R  # noqa: E402

REPO_ID = 'blab-jhu/KYS-3B-pretraining-data'
MODEL_REPO_ID = 'blab-jhu/KYS-3B-pretraining-models'      # trained models later; nothing is uploaded there here
CODE_REMOTE = 'git@github.com:imHuicongZhang/nanotron.git'
CODE_HTTPS = 'https://github.com/imHuicongZhang/nanotron.git'
CODE_URL = 'https://github.com/imHuicongZhang/nanotron'
BRANCH = 'release/kys-3b-600m'
BASE_BRANCH = 'huicong-dev'
CHECKOUT = R.REPO
PRETRAIN_PY = '/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python'
PYTEST_PATH = str(R.TRAIN_ROOT / 'tools' / 'pytest_pkgs')
WORK = R.TRAIN_ROOT / 'release_3b'
STAGE = WORK / 'stage'
REC = R.REPORTS / 'release_3b'
STAGES = ['gate', 'code', 'stage', 'remote_check', 'upload', 'verify_remote', 'fresh_download', 'receipt']
TOP = ['RELEASE_MANIFEST.json', 'HANDOFF_RELEASE.json', 'README.md']      # written by `stage`, committed last
DATA = R.DATA_ROOT_600M / 'dataset'
SELECTION_QB = DATA / '01_selection' / 'quality-base' / '_manifest.json'
REPORT_FILES = {   # published path under reports/ -> local file (provenance and validation only)
    'stage04_assembly/validate_assembly_scope.json': DATA / 'reports/stage04_assembly_20261016/validate_assembly_scope.json',
    'stage04_assembly/validate_rewritten_scope_post_stage04.json':
        DATA / 'reports/stage04_assembly_20261016/validate_rewritten_scope_post_stage04.json',
    'stage04_assembly/STATUS.md': DATA / 'reports/stage04_assembly_20261016/STATUS.md',
    'stage04_assembly/PROVENANCE_SELECTION.md': DATA / 'reports/stage04_assembly_20261016/PROVENANCE_SELECTION.md',
    'stage04_assembly/REVERIFY_20261017.md': DATA / 'reports/stage04_assembly_20261016/REVERIFY_20261017.md',
    'stage05/final_corpora.json': DATA / 'reports/final_corpora.json',
    'stage05/final_corpora_quality-base.json': DATA / 'reports/quality_base_20261030/final_corpora_quality-base.json',
    'quality_base/validate_quality_base.json': DATA / 'reports/quality_base_20261030/validate_quality_base.json',
    'quality_base/STATUS.md': DATA / 'reports/quality_base_20261030/STATUS.md',
    'quality_base/selection_manifest.json': SELECTION_QB,
    'quality_base/config_comparison.json': R.REPORTS / 'quality_base_20261030/config_comparison.json',
    'quality_base/lr_resume_quality_base.txt': R.REPORTS / 'quality_base_20261030/lr_resume_quality_base.txt',
    'quality_base/lr_resume_upstream_vs_worktree_quality_base.json':
        R.REPORTS / 'quality_base_20261030/lr_resume_upstream_vs_worktree_quality_base.json',
    'tokenize/empty_docs_scan.py': R.REPORTS / 'tokenize/empty_docs_scan.py',
    'tokenize/empty_docs_scan.1126434.log': R.REPORTS / 'tokenize/empty_docs_scan.1126434.log',
    **{f'lr_resume_fix_20261017/{n}': R.REPORTS / 'lr_resume_fix_20261017' / n
       for n in ('lr_resume_boundaries.txt', 'lr_resume_report.json', 'lr_resume_report_before_fix.json',
                 'tests_after_fix.txt', 'tests_before_fix.txt')},
    **{f'tokenize/{s}/{n}': R.REPORTS / 'tokenize' / s / p
       for s in R.GRID_SETTINGS for n, p in (('epoch_stats.json', 'epoch_stats.json'),
                                             ('datatrove_stats.json', 'datatrove/stats.json'),
                                             ('datatrove_executor.json', 'datatrove/executor.json'))},
}
SECRET_RE = re.compile(r'gh[pousr]_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{34}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
                       r'|https?://[^/\s:@]+:[^/\s@]+@')


# ---- helpers ----------------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f'[release3b] {time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}', flush=True)


def sh(argv, cwd=None, env=None, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run([str(a) for a in argv], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f'command failed ({r.returncode}): {" ".join(map(str, argv))}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}')
    return r


def record(stage: str) -> dict | None:
    p = REC / f'{stage}.json'
    return json.loads(p.read_text()) if p.is_file() else None


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + '\n')
    tmp.replace(p)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def clean_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('KYS3B_')}
    env.update(PYTHONPATH=PYTEST_PATH, **extra)
    return env


def api():
    from huggingface_hub import HfApi
    return HfApi()


def run_checks(repo: Path) -> dict:
    """The grid checks, run from `repo` with the Skipjack defaults (no KYS3B_* overrides)."""
    out = {}
    cmds = {'invariants_grid_corpus': [PRETRAIN_PY, 'deploy/skipjack_3b_600m/assert_invariants_3b.py', '--grid', '--corpus'],
            'invariants_all': [PRETRAIN_PY, 'deploy/skipjack_3b_600m/assert_invariants_3b.py', '--all'],
            **{t: [PRETRAIN_PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'deploy/skipjack_3b_600m/{t}.py']
               for t in ('test_quality_base_3b', 'test_lr_resume_3b', 'test_portable_3b', 'test_capped_submit_3b',
                         'test_train8_3b')}}
    for name, argv in cmds.items():
        r = sh(argv, cwd=repo, env=clean_env(), check=False)
        out[name] = {'rc': r.returncode, 'tail': (r.stdout + r.stderr).strip().splitlines()[-3:]}
    out['ok'] = all(v['rc'] == 0 for v in out.values() if isinstance(v, dict))
    return out


# ---- stages -----------------------------------------------------------------------------------------------------
def st_gate() -> dict:
    errs = []
    for s in R.GRID_SETTINGS:
        if not (R.REPORTS / 'tokenize' / s / 'epoch_stats.json').is_file():
            errs.append(f'{s}: tokenization not verified (no epoch_stats.json)')
    for s, p in R.ASSEMBLY_VALIDATION.items():
        rep = json.loads(p.read_text()) if p.is_file() else None
        if rep is None or rep.get('problems'):
            errs.append(f'{s}: {p} missing or reports problems')
    checks = run_checks(CHECKOUT)
    if not checks['ok']:
        errs.append(f'grid checks failed: { {k: v for k, v in checks.items() if isinstance(v, dict) and v["rc"]} }')
    return {'ok': not errs, 'errors': errs, 'checks': checks}


def st_code() -> dict:
    git = ['git', '-C', CHECKOUT]
    if sh([*git, 'status', '--porcelain']).stdout.strip():
        raise SystemExit(f'{CHECKOUT} has uncommitted changes; commit them on {BRANCH} first')
    if sh([*git, 'rev-parse', '--abbrev-ref', 'HEAD']).stdout.strip() != BRANCH:
        raise SystemExit(f'{CHECKOUT} is not on {BRANCH}')
    head = sh([*git, 'rev-parse', 'HEAD']).stdout.strip()
    tracked = sh([*git, 'ls-files']).stdout.splitlines()
    need = [f'configs/3B-rewrite-600m/{R.cfg_dir(s).name}/{R.seg_name(s, k)}.{x}'
            for s in R.GRID_SETTINGS for k in R.KINDS for x in ('yaml', 'env')]
    missing = [p for p in need if p not in tracked]
    if missing:
        raise SystemExit(f'configs not tracked in git: {missing[:5]} ... ({len(missing)})')

    remote_env = None
    r = sh(['git', 'ls-remote', CODE_REMOTE, f'refs/heads/{BRANCH}'], check=False)
    remote = CODE_REMOTE
    if r.returncode:
        log(f'ssh remote unavailable ({r.stderr.strip()[:200]}); using HTTPS with the gh credential helper')
        remote = CODE_HTTPS
        remote_env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
        r = sh(['git', '-c', 'credential.helper=', '-c', f'credential.helper=!{shutil.which("gh") or "gh"} auth git-credential',
                'ls-remote', remote, f'refs/heads/{BRANCH}'], env=remote_env)
    cred = [] if remote == CODE_REMOTE else ['-c', 'credential.helper=', '-c',
                                              f'credential.helper=!{shutil.which("gh") or "gh"} auth git-credential']
    sh(['git', *cred, '-C', CHECKOUT, 'fetch', '--quiet', remote, BASE_BRANCH], env=remote_env)
    diff = sh([*git, 'diff', 'FETCH_HEAD...HEAD']).stdout
    hits = sorted({m.group(0)[:12] + '...' for m in SECRET_RE.finditer(diff)})
    if hits:
        raise SystemExit(f'secret-like strings in the branch diff, refusing to push: {hits}')
    have = r.stdout.split()[0] if r.stdout.strip() else None
    if have == head:
        log(f'{BRANCH} already at {head} on origin')
    else:
        if have and sh([*git, 'merge-base', '--is-ancestor', have, head], check=False).returncode:
            raise SystemExit(f'origin {BRANCH} is at {have}, not an ancestor of {head}; refusing to force-push')
        sh(['git', *cred, '-C', CHECKOUT, 'push', remote, f'{head}:refs/heads/{BRANCH}'], env=remote_env)
    pushed = sh(['git', *cred, 'ls-remote', remote, f'refs/heads/{BRANCH}'], env=remote_env).stdout.split()[0]
    if pushed != head:
        raise SystemExit(f'origin {BRANCH} is {pushed} after push, expected {head}')

    clone = WORK / f'code_clone_{head[:12]}'
    if clone.exists():
        shutil.rmtree(clone)
    clone.parent.mkdir(parents=True, exist_ok=True)
    sh(['git', *cred, 'clone', '--quiet', '--branch', BRANCH, '--single-branch', remote, clone], env=remote_env)
    got = sh(['git', '-C', clone, 'rev-parse', 'HEAD']).stdout.strip()
    checks = run_checks(clone) if got == head else {'ok': False}
    rec = {'ok': got == head and checks['ok'], 'commit': head, 'branch': BRANCH, 'repo': CODE_URL,
           'branch_url': f'{CODE_URL}/tree/{BRANCH}', 'commit_url': f'{CODE_URL}/commit/{head}', 'pushed_via': remote,
           'fresh_clone': str(clone), 'fresh_clone_head': got, 'fresh_clone_checks': checks}
    if rec['ok']:
        shutil.rmtree(clone)
    return rec


def _hash_job(args):
    rel, src = args
    st = os.stat(src)
    return rel, {'bytes': st.st_size, 'sha256': sha256(src)}


def _link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if os.path.samefile(src, dst):
            return 'existing'
        raise SystemExit(f'{dst} exists and is not {src}; inspect the staging tree')
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        os.symlink(src, dst)
        return 'symlink'


def _sources() -> dict:
    """published path -> local source, every data file of the release (not the three top files)."""
    src = {}
    for s in R.GRID_SETTINGS:
        parts = sorted((R.FINAL / s / 'shuffled').glob('part_*.parquet'))
        if len(parts) != R.CORPUS_PARTS[s]:
            raise SystemExit(f'{s}: {len(parts)} shuffled parts, expected {R.CORPUS_PARTS[s]}')
        for p in parts:
            src[f'corpora/{s}/{p.name}'] = p
        for n in ('_manifest.json', '_pretrain_manifest.json'):
            src[f'corpora/{s}/{n}'] = R.FINAL / s / n
        tok = sorted(p for p in (R.TOKENIZED / s).iterdir() if p.is_file())
        if len(tok) != 3 * R.CORPUS_PARTS[s]:
            raise SystemExit(f'{R.TOKENIZED / s}: {len(tok)} files, expected {3 * R.CORPUS_PARTS[s]} (.ds/.index/.metadata per part)')
        for p in tok:
            src[f'tokenized/{s}/{p.name}'] = p
    for n in R.TOKENIZER_SHA256:
        src[f'tokenizer/{n}'] = R.TOKENIZER / n
    for rel, p in REPORT_FILES.items():
        src[f'reports/{rel}'] = p
    for rel, p in src.items():
        if not p.is_file():
            raise SystemExit(f'missing release input {p} (for {rel})')
    return src


def grid_manifest() -> dict:
    import yaml
    segs = []
    for s in R.GRID_SETTINGS:
        for k in R.KINDS:
            y = R.cfg_dir(s) / f'{R.seg_name(s, k)}.yaml'
            c = yaml.safe_load(y.read_text())
            segs.append({'setting': s, 'kind': k, 'run': c['general']['run'], 'first_step': R.FIRST_STEP[k],
                         'end_step': R.END_STEP[k], 'depends_on': R.DEPENDS[k], 'final': k in R.BRANCH_KINDS,
                         'config': str(y.relative_to(R.REPO)), 'config_sha256': sha256(y),
                         'env_sha256': sha256(y.with_suffix('.env'))})
    return {'grid': '3B rewrite-600m, four settings, seed 42', 'settings': R.GRID_SETTINGS,
            'live_settings_on_skipjack': R.SETTINGS, 'prepared_settings': R.PREPARED_SETTINGS, 'seed': R.SEED,
            'n_configs': len(segs), 'n_finals': sum(x['final'] for x in segs), 'horizons_steps': R.HORIZONS,
            'branch_points': R.BRANCH_POINTS, 'trunk_ends': R.TRUNK_ENDS, 'tokens_per_step': R.TOKENS_PER_STEP,
            'peak_lr': R.PEAK_LR, 'warmup_steps': R.WARMUP_STEPS, 'n_layers': R.N_LAYERS, 'n_params': R.N_PARAMS,
            'segments': segs}


def st_stage() -> dict:
    import pyarrow.parquet as pq
    src = _sources()
    STAGE.mkdir(parents=True, exist_ok=True)
    gm = grid_manifest()                                      # deterministic: a rerun writes the same bytes
    write_json(STAGE / 'reports' / 'GRID_MANIFEST.json', gm)
    write_json(REC / 'GRID_MANIFEST.json', gm)
    src['reports/GRID_MANIFEST.json'] = STAGE / 'reports' / 'GRID_MANIFEST.json'
    kinds = {}
    for rel, p in src.items():
        k = _link(p, STAGE / rel)
        kinds[k] = kinds.get(k, 0) + 1
    extra = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and str(p.relative_to(STAGE)) not in src and str(p.relative_to(STAGE)) not in TOP)
    if extra:
        raise SystemExit(f'unexpected files in {STAGE}: {extra[:5]}')

    cache_p = REC / 'hash_cache.json'
    cache = json.loads(cache_p.read_text()) if cache_p.is_file() else {}
    todo = []
    for rel, p in src.items():
        st = os.stat(p)
        c = cache.get(rel)
        if not (c and c['bytes'] == st.st_size and c.get('mtime_ns') == st.st_mtime_ns and c.get('src') == str(p)):
            todo.append((rel, str(p)))
    log(f'hashing {len(todo)} of {len(src)} files')
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as ex:
        for i, (rel, h) in enumerate(ex.map(_hash_job, todo, chunksize=1)):
            p = src[rel]
            cache[rel] = {**h, 'mtime_ns': os.stat(p).st_mtime_ns, 'src': str(p)}
            if i % 100 == 99:
                write_json(cache_p, cache)
    write_json(cache_p, cache)
    files = {rel: {'bytes': cache[rel]['bytes'], 'sha256': cache[rel]['sha256']} for rel in sorted(src)}

    errs, settings = [], {}
    for n, want in R.TOKENIZER_SHA256.items():
        if files[f'tokenizer/{n}']['sha256'] != want:
            errs.append(f'tokenizer/{n}: sha256 differs from the pinned 1.5B tokenizer')
    for s in R.GRID_SETTINGS:
        val = json.loads(R.ASSEMBLY_VALIDATION[s].read_text())['results'][s]['shuffled']['part_sha256']
        parts = sorted(r for r in files if r.startswith(f'corpora/{s}/part_'))
        if sorted(val) != [r.rsplit('/', 1)[1] for r in parts]:
            errs.append(f'{s}: part names differ from the validation report')
        rows, plist = 0, []
        for r in parts:
            if files[r]['sha256'] != val.get(r.rsplit('/', 1)[1]):
                errs.append(f'{r}: sha256 differs from {R.ASSEMBLY_VALIDATION[s].name}')
            n = pq.ParquetFile(src[r]).metadata.num_rows
            rows += n
            plist.append({'path': r, 'rows': n, **files[r]})
        pm = json.loads((R.FINAL / s / '_pretrain_manifest.json').read_text())
        tokens, docs, shards = 0, 0, []
        for i, pr in enumerate(parts):
            stem = f'tokenized/{s}/{i:05d}_unshuffled.ds'
            meta = (STAGE / f'{stem}.metadata').read_text().splitlines()
            n_tok = int(meta[1])
            n_doc = files[f'{stem}.index']['bytes'] // 8
            if files[stem]['bytes'] != 2 * n_tok:
                errs.append(f'{stem}: {files[stem]["bytes"]} bytes != 2 x {n_tok} tokens')
            tokens += n_tok
            docs += n_doc
            shards.append({'part': pr, 'ds': stem, 'index': f'{stem}.index', 'metadata': f'{stem}.metadata',
                           'tokens': n_tok, 'documents': n_doc, 'metadata_tokenizer_field': meta[0]})
        want = {'rows': R.CORPUS_ROWS[s], 'parts': R.CORPUS_PARTS[s], 'train_tokens': R.CORPUS_TRAIN_TOKENS[s],
                'tokenized_tokens': R.TOKENIZED_TOKENS[s], 'tokenized_documents': R.CORPUS_ROWS[s] - R.CORPUS_EMPTY_DOCS[s]}
        have = {'rows': rows, 'parts': len(parts), 'train_tokens': pm['corpus_train_tokens'], 'tokenized_tokens': tokens,
                'tokenized_documents': docs}
        if have != want:
            errs.append(f'{s}: totals {have} != registry {want}')
        settings[s] = {'assembled_rows': rows, 'assembled_train_tokens': pm['corpus_train_tokens'],
                       'empty_text_documents': R.CORPUS_EMPTY_DOCS[s], 'tokenized_tokens': tokens,
                       'tokenized_documents': docs, 'n_parts': len(parts),
                       'shortfall_vs_20B': max(0, R.EPOCH_TOKENS - pm['corpus_train_tokens']),
                       'bytes_parquet': sum(x['bytes'] for x in plist),
                       'bytes_tokenized': sum(files[r]['bytes'] for r in files if r.startswith(f'tokenized/{s}/')),
                       'parts': plist, 'tokenized_shards': shards}
    if errs:
        return {'ok': False, 'errors': errs}

    manifest = {'dataset': REPO_ID, 'description': 'KYS 3B four-setting pretraining release (seed 42)',
                'settings_order': R.GRID_SETTINGS, 'tokenizer_sha256': R.TOKENIZER_SHA256,
                'n_files': len(files), 'total_bytes': sum(f['bytes'] for f in files.values()),
                'files': files, 'settings': settings}
    write_json(STAGE / 'RELEASE_MANIFEST.json', manifest)
    code = record('code')
    handoff = handoff_release(manifest, code, sha256(STAGE / 'RELEASE_MANIFEST.json'))
    write_json(STAGE / 'HANDOFF_RELEASE.json', handoff)
    (STAGE / 'README.md').write_text(card(manifest, code))
    top = {n: {'bytes': (STAGE / n).stat().st_size, 'sha256': sha256(STAGE / n)} for n in TOP}
    return {'ok': True, 'links': kinds, 'n_files': len(files) + len(TOP), 'total_bytes': manifest['total_bytes'],
            'top_files': top, 'totals': {s: {k: v for k, v in d.items() if k not in ('parts', 'tokenized_shards')}
                                         for s, d in settings.items()}}


def handoff_release(manifest: dict, code: dict, manifest_sha: str) -> dict:
    return {
        'release': 'KYS 3B rewrite-600m, four settings, seed 42',
        'dataset_repo': REPO_ID, 'dataset_revision': 'see receipts/UPLOAD_RECEIPT.json (written after this file is published)',
        'release_manifest_sha256': manifest_sha,
        'code': {'repo': CODE_URL, 'branch': BRANCH, 'commit': code['commit']},
        'model_output_repo': MODEL_REPO_ID,
        'settings': R.GRID_SETTINGS,
        'grid': {'configs': len(R.GRID_SETTINGS) * len(R.KINDS), 'annealed_finals': len(R.GRID_SETTINGS) * len(R.BRANCH_KINDS),
                 'segments': R.KINDS, 'depends_on': R.DEPENDS, 'horizons_steps': R.HORIZONS, 'branch_points': R.BRANCH_POINTS,
                 'tokens_per_step': R.TOKENS_PER_STEP, 'seq_len': R.SEQ_LEN, 'global_batch_seq': R.GLOBAL_BATCH_SEQ,
                 'peak_lr': R.PEAK_LR, 'warmup_steps': R.WARMUP_STEPS, 'lr_schedule': 'WSD: linear warmup, constant, linear decay to 0',
                 'config_dirs': {s: str(R.cfg_dir(s).relative_to(R.REPO)) for s in R.GRID_SETTINGS}},
        'model': {**R.MODEL, 'n_params': R.N_PARAMS, 'n_params_non_embedding': R.N_PARAMS_NON_EMBEDDING, 'dtype': 'bfloat16'},
        'tokenizer': {'files': {n: f'tokenizer/{n}' for n in R.TOKENIZER_SHA256}, 'sha256': R.TOKENIZER_SHA256,
                      'source': 'the 1.5B KYS tokenizer (llama-2, unsloth/llama-2-7b), byte-identical; one </s> per document, no BOS'},
        'initialization': {'available': False,
                           'reason': 'not yet created; it needs one GPU job and is created by the first trunk1 run',
                           'procedure': 'deploy/skipjack_3b_600m/init_3b.sh + startup_3b.py derive-init (RELEASE.md, "Initialization")',
                           'shared_by': R.GRID_SETTINGS, 'seed': R.SEED, 'init_std': R.INIT_STD},
        'totals': {s: {k: d[k] for k in ('assembled_rows', 'assembled_train_tokens', 'empty_text_documents',
                                          'tokenized_tokens', 'tokenized_documents', 'n_parts', 'shortfall_vs_20B')}
                   for s, d in manifest['settings'].items()},
        'ordered_files': {s: {'corpus_parts': [x['path'] for x in d['parts']],
                              'tokenized_shards': [x['ds'] for x in d['tokenized_shards']]}
                          for s, d in manifest['settings'].items()},
        'notes': ['tokenized shard i is shuffled part i, in order; nanotron reads every *.ds of the folder',
                  'run tools/fix_ds_metadata.py on each tokenized/<setting>/ after download (line 1 of .ds.metadata '
                  'records the Skipjack tokenizer path); binaries are unchanged',
                  'tokenized = assembled train tokens - empty-text documents (datatrove skips a document without text)',
                  'diversity-oriented: accepted shortfall of 165,685,567 tokens vs 20B, unchanged'],
    }


def card(manifest: dict, code: dict) -> str:
    rows = '\n'.join(f'| {s} | {d["n_parts"]} | {d["assembled_rows"]:,} | {d["assembled_train_tokens"]:,} | '
                     f'{d["empty_text_documents"]} | {d["tokenized_tokens"]:,} | {d["tokenized_documents"]:,} |'
                     for s, d in manifest['settings'].items())
    return f"""---
license: other
pretty_name: KYS 3B pretraining data (four settings)
---

# KYS 3B pretraining data — four settings, seed 42

Private. Training data for the KYS 3B grid: a 57-layer, 2,994,444,288-parameter llama model (the 1.5B architecture with
only the depth changed), trained for 20B / 40B / 60B tokens (WSD; finals at steps 9537 / 19073 / 28610). Code:
[{CODE_URL}/tree/{BRANCH}]({CODE_URL}/tree/{BRANCH}) at `{code['commit']}`. Trained models will be published to
`{MODEL_REPO_ID}`.

| setting | parts | rows | train tokens (assembled) | empty-text docs | tokens (tokenized) | documents (tokenized) |
|---|---|---|---|---|---|---|
{rows}

* **quality-first / diversity-oriented / disagreement-aware**: the shared raw base (10,527,887 docs,
  10,000,000,186 train tokens: top DCLM fastText) plus a rewritten 10B strategy block.
* **quality-base**: the same shared base plus the NEXT 7,414,631 documents (10,000,000,113 train tokens) in the same
  fastText order, raw, so the corpus is the global fastText top ~20B prefix with no overlap or duplicate.
* Train tokens are llama-2 tokens + 1 EOS per document. Datatrove skips documents with empty text, so the tokenized
  stream is the assembled count minus the empty-text documents (rows left empty by the rewrite strip). diversity-oriented
  is 165,685,567 tokens short of 20B (accepted; its epochs are re-read accordingly, `reports/tokenize/`).

## Layout

* `corpora/<setting>/part_*.parquet`: the final shuffled corpus (seed 42), in training order, unchanged.
* `tokenized/<setting>/NNNNN_unshuffled.ds{{,.index,.metadata}}`: nanotron/datatrove shards, shard i = part i. After
  downloading run `tools/fix_ds_metadata.py --output-folder tokenized/<setting> --tokenizer-dir tokenizer` (line 1 of
  each `.metadata` records the path the shards were written under).
* `tokenizer/`: the exact 1.5B llama-2 tokenizer (sha256 in `HANDOFF_RELEASE.json`).
* `reports/`: assembly/selection validation, tokenization statistics, LR-resume regression, `GRID_MANIFEST.json`.
* `RELEASE_MANIFEST.json`: size and sha256 of every file, ordered parts and shards with rows/tokens/documents.
* `HANDOFF_RELEASE.json`: code commit, grid, model, tokenizer and initialization provenance, totals.
* `receipts/UPLOAD_RECEIPT.json`: the verified content revision and the verification results.

No initialization checkpoint is included: none exists yet. It is created on GPU by the first trunk1 run
(`deploy/skipjack_3b_600m/init_3b.sh`; see `RELEASE.md` in the code branch) and shared by all four settings.
"""


def _remote_files(a, revision=None) -> tuple[dict, str, bool]:
    info = a.repo_info(REPO_ID, repo_type='dataset', revision=revision, files_metadata=True)
    files = {s.rfilename: {'bytes': s.size, 'lfs_sha256': (s.lfs.sha256 if s.lfs else None)} for s in info.siblings}
    return files, info.sha, bool(info.private)


def _remote_sha(a, rel: str, meta: dict, revision: str, tmp: Path) -> str:
    if meta.get('lfs_sha256'):
        return meta['lfs_sha256']
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, rel, repo_type='dataset', revision=revision, cache_dir=str(tmp))
    return sha256(p)


def _expected() -> dict:
    man = json.loads((STAGE / 'RELEASE_MANIFEST.json').read_text())
    top = record('stage')['top_files']
    return {**man['files'], **top}


def st_remote_check() -> dict:
    a = api()
    remote, rev, private = _remote_files(a)
    if not private:
        raise SystemExit(f'{REPO_ID} is not private; refusing to upload')
    exp = _expected()
    tmp = WORK / 'remote_check_tmp'
    reuse, conflict, unexpected = [], [], []
    for rel, m in remote.items():
        if rel == '.gitattributes' or rel.startswith('receipts/'):
            continue
        if rel not in exp:
            unexpected.append(rel)
        elif m['bytes'] != exp[rel]['bytes'] or _remote_sha(a, rel, m, rev, tmp) != exp[rel]['sha256']:
            conflict.append(rel)
        else:
            reuse.append(rel)
    shutil.rmtree(tmp, ignore_errors=True)
    return {'ok': not conflict and not unexpected, 'revision_before': rev, 'private': private, 'reused': len(reuse),
            'conflicts': conflict, 'unexpected': unexpected}


def st_upload() -> dict:
    from huggingface_hub import CommitOperationAdd
    a = api()
    if not _remote_files(a)[2]:
        raise SystemExit(f'{REPO_ID} is not private; refusing to upload')
    a.upload_large_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(STAGE), private=True,
                          ignore_patterns=TOP, num_workers=int(os.environ.get('KYS3B_UPLOAD_WORKERS', 8)),
                          print_report=False)
    remote, _, _ = _remote_files(a)
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(STAGE / n)) for n in TOP]
    exp = _expected()
    if all(n in remote and remote[n]['bytes'] == exp[n]['bytes'] for n in TOP):
        commit = None
    else:
        commit = a.create_commit(REPO_ID, operations=ops, repo_type='dataset',
                                 commit_message='KYS 3B four-setting release: manifest, handoff, dataset card').oid
    remote, rev, private = _remote_files(a)
    return {'ok': private, 'top_commit': commit, 'revision': rev, 'private': private}


def st_verify_remote() -> dict:
    a = api()
    remote, rev, private = _remote_files(a)
    exp = _expected()
    tmp = WORK / 'verify_tmp'
    missing = [r for r in exp if r not in remote]
    bad, downloaded = [], 0
    for rel in exp:
        if rel in remote:
            m = remote[rel]
            if not m.get('lfs_sha256'):
                downloaded += 1
            if m['bytes'] != exp[rel]['bytes'] or _remote_sha(a, rel, m, rev, tmp) != exp[rel]['sha256']:
                bad.append(rel)
    shutil.rmtree(tmp, ignore_errors=True)
    unexpected = sorted(set(remote) - set(exp) - {'.gitattributes'} - {r for r in remote if r.startswith('receipts/')})
    return {'ok': private and not missing and not bad and not unexpected, 'content_revision': rev, 'private': private,
            'checked': len(exp), 'checked_by_lfs_sha256': len(exp) - len(missing) - downloaded,
            'checked_by_download': downloaded, 'missing': missing, 'mismatched': bad, 'unexpected': unexpected,
            'remote_total_bytes': sum(remote[r]['bytes'] for r in exp if r in remote)}


def st_fresh_download() -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    rev = record('verify_remote')['content_revision']
    man = json.loads((STAGE / 'RELEASE_MANIFEST.json').read_text())
    root = WORK / f'fresh_{rev[:12]}'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    picks = [*(f'tokenizer/{n}' for n in R.TOKENIZER_SHA256)]
    for s, d in man['settings'].items():
        picks += [d['parts'][0]['path'], d['parts'][-1]['path']]
        sh_ = d['tokenized_shards'][len(d['tokenized_shards']) // 2]
        picks += [sh_['ds'], sh_['index'], sh_['metadata']]
    got, errs = {}, []
    for rel in picks:
        p = Path(hf_hub_download(REPO_ID, rel, repo_type='dataset', revision=rev, cache_dir=str(root / 'cache'),
                                 local_dir=str(root / 'files')))
        got[rel] = p
        if sha256(p) != man['files'][rel]['sha256']:
            errs.append(f'{rel}: downloaded sha256 differs from the manifest')
    rep = json.loads(got['tokenizer/tokenizer_report.json'].read_text())
    test = rep['sanity_check']['test_text']
    ids = Tokenizer.from_file(str(got['tokenizer/tokenizer.json'])).encode(test, add_special_tokens=False).ids
    if ids != rep['sanity_check']['ids_add_special_tokens_false']:
        errs.append('downloaded tokenizer encodes the sanity text differently from tokenizer_report.json')
    for s, d in man['settings'].items():
        for x in (d['parts'][0], d['parts'][-1]):
            pf = pq.ParquetFile(got[x['path']])
            if pf.metadata.num_rows != x['rows'] or 'text' not in pf.schema_arrow.names:
                errs.append(f'{x["path"]}: rows/schema differ')
            t = pf.read_row_group(0, columns=['text']).column('text').to_pylist()[:3]
            if not all(isinstance(v, str) for v in t):
                errs.append(f'{x["path"]}: text column unreadable')
        sh_ = d['tokenized_shards'][len(d['tokenized_shards']) // 2]
        meta = got[sh_['metadata']].read_text().splitlines()
        if int(meta[1]) != sh_['tokens'] or got[sh_['ds']].stat().st_size != 2 * sh_['tokens'] \
                or got[sh_['index']].stat().st_size // 8 != sh_['documents']:
            errs.append(f'{sh_["ds"]}: tokens/documents differ from the manifest')
    shutil.rmtree(root)
    return {'ok': not errs, 'revision': rev, 'sampled_files': picks, 'errors': errs,
            'checks': ['sha256 of each downloaded file', 'tokenizer encodes the sanity text to the recorded ids',
                       'parquet rows and text column', 'shard tokens (= .ds bytes / 2) and documents (= .index bytes / 8)']}


def st_receipt() -> dict:
    a = api()
    recs = {s: record(s) for s in STAGES[:-1]}
    man = json.loads((STAGE / 'RELEASE_MANIFEST.json').read_text())
    vr, code = recs['verify_remote'], recs['code']
    ready = all(r and r.get('ok') for r in recs.values())
    receipt = {
        'ready': ready,
        'dataset': {'repo': REPO_ID, 'url': f'https://huggingface.co/datasets/{REPO_ID}', 'private': vr['private'],
                    'content_revision': vr['content_revision'], 'n_files': vr['checked'], 'total_bytes': vr['remote_total_bytes'],
                    'release_manifest_sha256': record('stage')['top_files']['RELEASE_MANIFEST.json']['sha256']},
        'code': {k: code[k] for k in ('repo', 'branch', 'commit', 'branch_url', 'commit_url')},
        'model_output_repo': MODEL_REPO_ID,
        'totals': {s: {k: v for k, v in d.items() if k not in ('parts', 'tokenized_shards')} for s, d in man['settings'].items()},
        'verification': {
            'exhaustive': {'local': 'sha256 of every file; parquet sha256 = assembly validation; tokenizer sha256 pinned; '
                                    'rows/parts/train tokens/tokenized tokens/documents = registry (stage)',
                           'remote': f'all {vr["checked"]} files: path, size and sha256 ({vr["checked_by_lfs_sha256"]} by '
                                     f'LFS sha256, {vr["checked_by_download"]} downloaded and hashed); no unexpected files; private',
                           'code': 'grid invariants (36 configs + four corpora exact), QB/LR-resume/portable/submitter tests '
                                   'from this checkout and from a fresh clone of the pushed branch'},
            'sampled': {'fresh_download': recs['fresh_download']['sampled_files'], 'checks': recs['fresh_download']['checks']},
        },
        'stages': {s: {'ok': bool(r and r.get('ok'))} for s, r in recs.items()},
    }
    write_json(REC / 'UPLOAD_RECEIPT.json', receipt)
    if not ready:
        return {'ok': False, 'receipt': str(REC / 'UPLOAD_RECEIPT.json')}
    info = a.upload_file(path_or_fileobj=str(REC / 'UPLOAD_RECEIPT.json'), path_in_repo='receipts/UPLOAD_RECEIPT.json',
                         repo_id=REPO_ID, repo_type='dataset',
                         commit_message=f'Upload receipt for content revision {vr["content_revision"]}')
    remote, final, private = _remote_files(a)
    before, _, _ = _remote_files(a, revision=vr['content_revision'])
    delta = sorted(set(remote) ^ set(before)) + sorted(r for r in before if r in remote and remote[r] != before[r])
    ok = private and delta == ['receipts/UPLOAD_RECEIPT.json'] and \
        remote['receipts/UPLOAD_RECEIPT.json']['bytes'] == (REC / 'UPLOAD_RECEIPT.json').stat().st_size
    return {'ok': ok, 'final_revision': final, 'content_revision': vr['content_revision'], 'commit_url': info.commit_url,
            'delta_vs_content_revision': delta, 'private': private, 'receipt_sha256': sha256(REC / 'UPLOAD_RECEIPT.json')}


FUNCS = {'gate': st_gate, 'code': st_code, 'stage': st_stage, 'remote_check': st_remote_check, 'upload': st_upload,
         'verify_remote': st_verify_remote, 'fresh_download': st_fresh_download, 'receipt': st_receipt}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=['run', 'status'])
    ap.add_argument('--through', choices=STAGES, default=STAGES[-1])
    args = ap.parse_args(argv)
    REC.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'status':
        for s in STAGES:
            r = record(s)
            print(f'{s:15s} {"-" if r is None else ("ok" if r.get("ok") else "FAILED")}')
        return 0
    for s in STAGES[:STAGES.index(args.through) + 1]:
        r = record(s)
        if r and r.get('ok'):
            log(f'{s}: done earlier, skipping')
            continue
        log(f'{s}: start')
        t0 = time.time()
        r = FUNCS[s]()
        r = {**r, 'stage': s, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'), 'seconds': round(time.time() - t0),
             'job': os.environ.get('SLURM_JOB_ID')}
        write_json(REC / f'{s}.json', r)
        if not r['ok']:
            log(f'{s}: FAILED, see {REC / f"{s}.json"}')
            return 1
        log(f'{s}: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
