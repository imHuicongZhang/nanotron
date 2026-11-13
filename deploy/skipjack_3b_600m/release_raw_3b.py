#!/usr/bin/env python
"""Publish the two 3B raw-baseline corpora (raw_diversity_oriented, raw_disagreement_aware) as an ADDITION to the
four-setting release: new paths only, their own manifest; every file of the existing release stays byte-identical.
CPU only; never trains; nothing is submitted.

    python deploy/skipjack_3b_600m/release_raw_3b.py run [--through <stage>]
    python deploy/skipjack_3b_600m/release_raw_3b.py status

Stages run in order; each writes <reports>/release_raw_3b/<stage>.json and is skipped on a rerun once it says ok.
  gate            stage5 validation passed and tokenization reconciled (reports/tokenize_<s>.json) for both settings, the
                  shuffle state of every part present; portable tests pass from this checkout
  code            this checkout clean and on BRANCH; secret scan of the diff to the four-setting release commit; push BRANCH
                  (new branch or fast-forward, never force); fresh clone at the pushed commit reruns the portable tests
  stage           hardlink tree (byte-identical) of the new files only:
                    corpora/<s>/part_NNNNN.parquet, tokenized/<s>/NNNNN_unshuffled.ds{,.index,.metadata},
                    raw_baselines/ids/*.npy (explicit base / strategy membership lists), raw_baselines/mapping/*,
                    raw_baselines/reports/*, raw_baselines/{RAW_RELEASE_MANIFEST.json, HANDOFF_RAW.json, README.md}
  remote_check    repo visibility unchanged; none of the new paths exists remotely with different content (identical
                  ones are reused); records the revision and every file (size, sha256/blob) of the existing release
  upload          upload_large_folder of the stage tree (resumable, only new paths), circuit breaker on 401/403/long 429;
                  the three raw_baselines/ top files are committed last
  verify_remote   every new path remotely with the same size and sha256; every file of the pre-upload revision unchanged
                  (same size and LFS sha256 / git blob id); no other new files; records the content revision
  fresh_download  sampled at that revision into an empty directory: manifest, first/last part of each corpus (sha256, rows),
                  one tokenized shard triple of each (sha256, tokens), the id lists (sha256)
  receipt         raw_baselines/RAW_UPLOAD_RECEIPT.json (local + Hub) describing the verified content revision
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry_3b as R  # noqa: E402
import release_3b as RL  # noqa: E402   (helpers and the four-setting release constants; never modified here)
import science_3b as S  # noqa: E402

REPO_ID = RL.REPO_ID
PRIVATE, VISIBILITY = RL.PRIVATE, RL.VISIBILITY
BRANCH = 'release/kys-3b-600m-raw-baselines'
BASE_COMMIT = 'cd4ba52ab83807bc7a9502046a846829249849c0'      # release/kys-3b-600m, the four-setting code release
CHECKOUT = HERE.parents[1]
PREP = Path('/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/raw_baselines_prep')
PREFLIGHT = R.REPORTS / 'raw_baselines_preflight'
WORK = R.TRAIN_ROOT / 'release_raw_3b'
STAGE = WORK / 'stage'
REC = R.REPORTS / 'release_raw_3b'
STAGES = ['gate', 'code', 'stage', 'remote_check', 'upload', 'verify_remote', 'fresh_download', 'receipt']
NS = 'raw_baselines'
TOP = (f'{NS}/RAW_RELEASE_MANIFEST.json', f'{NS}/HANDOFF_RAW.json', f'{NS}/README.md')
RECEIPT = f'{NS}/RAW_UPLOAD_RECEIPT.json'
PORTABLE_TESTS = ['deploy/portable_3b/test_portable_3b_operator.py', 'deploy/portable_3b/test_portable_3b_render.py',
                  'deploy/portable_3b/test_portable_3b_init.py', 'deploy/portable_3b/test_portable_3b_lr.py',
                  'deploy/portable_3b/test_portable_3b_raw.py']
EXTRA = {   # published path -> local file (small provenance / validation records)
    f'{NS}/ids/base_ids_sorted.npy': PREP / 'stage1_ids/base_ids_sorted.npy',
    **{f'{NS}/ids/{s}_selected_perm_order.npy': PREP / f'stage2_select/{s}_selected_perm_order.npy' for s in S.RAW_SETTINGS},
    f'{NS}/mapping/expected_raw_sets.json': PREFLIGHT / 'evidence/out/expected_raw_sets.json',
    f'{NS}/mapping/shard_manifest.parquet': PREFLIGHT / 'evidence/local/shard_manifest.copy.parquet',
    f'{NS}/reports/prep_config.json': PREP / 'prep_config.json',
    f'{NS}/reports/stage12.json': PREP / 'state/stage12.json',
    f'{NS}/reports/test_resumable_shuffle_real100k.json': PREP / 'reports/test_resumable_shuffle_real100k.json',
    f'{NS}/reports/pool_sha256_sample.json': PREP / 'reports/pool_sha256_sample.json',
    **{f'{NS}/reports/{n}_{s}.json': PREP / ('state' if n == 'stage4_shuffle' else 'reports') / f'{n}_{s}.json'
       for s in S.RAW_SETTINGS for n in ('stage4_shuffle', 'stage5', 'tokenize', 'stage7')},
    **{f'{NS}/code/{n}': PREP / 'code' / n for n in ('stage12_ids_select.py', 'stage3_extract.py', 'stage4_assemble.py',
                                                     'stage4b_shuffle.py', 'resumable_shuffle.py', 'stage5_validate_corpus.py',
                                                     'stage7_final_validate.py', 'make_config.py', 'test_resumable_shuffle.py')},
}
log = lambda m: print(f'[release_raw3b] {time.strftime("%Y-%m-%dT%H:%M:%S")} {m}', flush=True)  # noqa: E731


def record(stage):
    p = REC / f'{stage}.json'
    return json.loads(p.read_text()) if p.is_file() else None


def shuffled(s) -> Path:
    return PREP / 'stage4_assemble' / s / 'shuffled'


# ---- stages -------------------------------------------------------------------------------------------------------
def st_gate() -> dict:
    errs = []
    for s in S.RAW_SETTINGS:
        v = json.loads((PREP / f'reports/stage5_{s}.json').read_text())
        t = json.loads((PREP / f'reports/tokenize_{s}.json').read_text())
        sh = json.loads((PREP / f'state/stage4_shuffle_{s}.json').read_text())
        if not v.get('pass'):
            errs.append(f'{s}: stage5 did not pass')
        if not t.get('pass') or t['tokens'] != S.TOKENIZED_TOKENS[s] or t['shards'] != S.CORPUS_PARTS[s]:
            errs.append(f'{s}: tokenization not reconciled')
        f7 = PREP / f'reports/stage7_{s}.json'   # per-document final validation (lengths, EOS, components by id list)
        v7 = json.loads(f7.read_text()) if f7.is_file() else {}
        if v7.get('pass') is not True or v7.get('partial') or v7['totals']['tokens'] != S.TOKENIZED_TOKENS[s] \
                or v7['totals']['rows'] != S.CORPUS_ROWS[s] or v7['totals']['base_tokens'] != S.SHARED_BASE_TRAIN_TOKENS \
                or v7['totals']['block_tokens'] != S.RAW_STRATEGY_BLOCK[s]['train_tokens']:
            errs.append(f'{s}: stage7 final validation missing or not passed')
        if v['rows'] != S.CORPUS_ROWS[s] or v['train_tokens'] != S.CORPUS_TRAIN_TOKENS[s] or v['empty_texts'] != S.CORPUS_EMPTY_DOCS[s]:
            errs.append(f'{s}: stage5 rows/tokens/empty differ from science_3b')
        if sorted(sh['files']) != sorted(p.name for p in shuffled(s).glob('part_*.parquet')) or len(sh['files']) != S.CORPUS_PARTS[s]:
            errs.append(f'{s}: shuffled parts differ from stage4_shuffle state')
    missing = [k for k, p in EXTRA.items() if not p.is_file()]
    if missing:
        errs.append(f'missing provenance files: {missing}')
    r = RL.sh([RL.PRETRAIN_PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *PORTABLE_TESTS], cwd=CHECKOUT,
              env=RL.clean_env(), check=False)
    tests = r.stdout.strip().splitlines()[-1:] or ['?']
    if r.returncode:
        errs.append(f'portable tests failed: {tests}')
    return {'ok': not errs, 'problems': errs, 'tests': tests}


def st_code() -> dict:
    git = ['git', '-C', CHECKOUT]
    if RL.sh([*git, 'status', '--porcelain']).stdout.strip():
        raise SystemExit(f'{CHECKOUT} has uncommitted changes; commit them on {BRANCH} first')
    if RL.sh([*git, 'rev-parse', '--abbrev-ref', 'HEAD']).stdout.strip() != BRANCH:
        raise SystemExit(f'{CHECKOUT} is not on {BRANCH}')
    head = RL.sh([*git, 'rev-parse', 'HEAD']).stdout.strip()
    if RL.sh([*git, 'merge-base', '--is-ancestor', BASE_COMMIT, head], check=False).returncode:
        raise SystemExit(f'{head} does not descend from the four-setting release {BASE_COMMIT}')
    diff = RL.sh([*git, 'diff', f'{BASE_COMMIT}...HEAD']).stdout
    hits = sorted({m.group(0)[:12] + '...' for m in RL.SECRET_RE.finditer(diff)})
    if hits:
        raise SystemExit(f'secret-like strings in the branch diff, refusing to push: {hits}')
    cred = ['-c', 'credential.helper=', '-c', f'credential.helper=!{shutil.which("gh") or "gh"} auth git-credential']
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    remote = RL.CODE_HTTPS
    have = RL.sh(['git', *cred, 'ls-remote', remote, f'refs/heads/{BRANCH}'], env=env).stdout.split()
    have = have[0] if have else None
    if have != head:
        if have and RL.sh([*git, 'merge-base', '--is-ancestor', have, head], check=False).returncode:
            raise SystemExit(f'origin {BRANCH} is at {have}, not an ancestor of {head}; refusing to force-push')
        RL.sh(['git', *cred, '-C', CHECKOUT, 'push', remote, f'{head}:refs/heads/{BRANCH}'], env=env)
    pushed = RL.sh(['git', *cred, 'ls-remote', remote, f'refs/heads/{BRANCH}'], env=env).stdout.split()[0]
    base_remote = RL.sh(['git', *cred, 'ls-remote', remote, f'refs/heads/{RL.BRANCH}'], env=env).stdout.split()[0]
    clone = WORK / f'code_clone_{head[:12]}'
    shutil.rmtree(clone, ignore_errors=True)
    clone.parent.mkdir(parents=True, exist_ok=True)
    RL.sh(['git', *cred, 'clone', '--quiet', '--branch', BRANCH, '--single-branch', remote, clone], env=env)
    got = RL.sh(['git', '-C', clone, 'rev-parse', 'HEAD']).stdout.strip()
    r = RL.sh([RL.PRETRAIN_PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *PORTABLE_TESTS], cwd=clone,
              env=RL.clean_env(), check=False)
    ok = pushed == head == got and not r.returncode and base_remote == BASE_COMMIT
    rec = {'ok': ok, 'commit': head, 'branch': BRANCH, 'repo': RL.CODE_URL, 'branch_url': f'{RL.CODE_URL}/tree/{BRANCH}',
           'commit_url': f'{RL.CODE_URL}/commit/{head}', 'four_setting_branch_unchanged': base_remote == BASE_COMMIT,
           'fresh_clone_head': got, 'fresh_clone_tests': r.stdout.strip().splitlines()[-1:]}
    if ok:
        shutil.rmtree(clone)
    return rec


def _sources() -> dict:
    src = {}
    for s in S.RAW_SETTINGS:
        for p in sorted(shuffled(s).glob('part_*.parquet')):
            src[f'corpora/{s}/{p.name}'] = p
        for p in sorted((R.TOKENIZED / s).iterdir()):
            if p.is_file() and p.name.split('.', 1)[-1] in ('ds', 'ds.index', 'ds.metadata'):
                src[f'tokenized/{s}/{p.name}'] = p
    src.update(EXTRA)
    return src


def st_stage() -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    if STAGE.exists():
        shutil.rmtree(STAGE)
    src = _sources()
    files = {}
    for rel, p in src.items():
        d = STAGE / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, d)
        files[rel] = {'bytes': d.stat().st_size, 'sha256': RL.sha256(d)}
    errs, settings = [], {}
    for s in S.RAW_SETTINGS:
        shs = json.loads((PREP / f'state/stage4_shuffle_{s}.json').read_text())['files']
        v = json.loads((PREP / f'reports/stage5_{s}.json').read_text())
        tk = json.loads((PREP / f'reports/tokenize_{s}.json').read_text())
        parts = []
        for name, m in sorted(shs.items()):
            rel = f'corpora/{s}/{name}'
            if files[rel]['sha256'] != m['sha256'] or pq.ParquetFile(STAGE / rel).metadata.num_rows != m['rows']:
                errs.append(f'{rel}: differs from the shuffle state')
            parts.append({'path': rel, 'rows': m['rows'], 'sha256': m['sha256']})
        shards = []
        for ds in sorted((STAGE / 'tokenized' / s).glob('*.ds')):
            n = int(Path(f'{ds}.metadata').read_text().splitlines()[1])
            rel = f'tokenized/{s}/{ds.name}'
            shards.append({'ds': rel, 'index': rel + '.index', 'metadata': rel + '.metadata', 'tokens': n})
        if sum(x['tokens'] for x in shards) != S.TOKENIZED_TOKENS[s] or len(shards) != S.CORPUS_PARTS[s]:
            errs.append(f'{s}: staged shards do not reconcile')
        sel = np.load(STAGE / NS / 'ids' / f'{s}_selected_perm_order.npy')
        settings[s] = {'counterpart': S.RAW_COUNTERPART[s], 'rows': v['rows'], 'train_tokens': v['train_tokens'],
                       'tokenized_tokens': tk['tokens'], 'empty_docs': v['empty_texts'], 'parts': parts,
                       'tokenized_shards': shards, 'order_digest_sha256': v['order_digest_sha256'],
                       'membership_sorted_sha256': v['membership_sorted_sha256'],
                       'components': {'shared_base': {'ids': f'{NS}/ids/base_ids_sorted.npy', 'docs': 10_527_887,
                                                      'train_tokens': S.SHARED_BASE_TRAIN_TOKENS},
                                      'raw_strategy_block': {'ids': f'{NS}/ids/{s}_selected_perm_order.npy',
                                                             'docs': int(sel.size), **S.RAW_STRATEGY_BLOCK[s]}},
                       'source_prompt': 'original (both components; membership is defined by the id lists)'}
        if sel.size != S.RAW_STRATEGY_BLOCK[s]['docs']:
            errs.append(f'{s}: selected id count')
    base = np.load(STAGE / NS / 'ids' / 'base_ids_sorted.npy')
    if base.size != 10_527_887:
        errs.append('base id count')
    manifest = {'dataset': REPO_ID, 'description': 'KYS 3B raw baselines (seed 42): raw_diversity_oriented, '
                'raw_disagreement_aware. Addition to the four-setting release; that release (RELEASE_MANIFEST.json) is unchanged.',
                'settings': settings, 'tokenizer': {f'tokenizer/{n}': h for n, h in S.TOKENIZER_SHA256.items()},
                'files': files}
    RL.write_json(STAGE / TOP[0], manifest)
    code = record('code') or {}
    handoff = {
        'what': 'two raw-baseline settings of the KYS 3B rewrite-600m grid, ready to train',
        'dataset_repo': REPO_ID, 'dataset_revision': f'see {RECEIPT} (written after this file is published)',
        'manifest': TOP[0], 'manifest_sha256': RL.sha256(STAGE / TOP[0]),
        'code': {'repo': RL.CODE_URL, 'branch': BRANCH, 'commit': code.get('commit')},
        'settings': S.RAW_SETTINGS, 'segments': len(S.RAW_SETTINGS) * len(S.KINDS),
        'annealed_finals': len(S.RAW_SETTINGS) * len(S.BRANCH_KINDS),
        'science': {'seed': S.SEED, 'horizons': S.HORIZONS, 'branch_points': S.BRANCH_POINTS, 'trunk_ends': S.TRUNK_ENDS,
                    'tokens_per_step': S.TOKENS_PER_STEP, 'peak_lr': S.PEAK_LR, 'n_params': S.N_PARAMS},
        'init': 'REUSE your existing <INIT_ROOT>/_init_3B_seed42 (the init of the four-setting grid). Put its '
                'init_manifest.json weights_sha256 in the operator config grid.init_weights_sha256; never create a new init.',
        'operator': {'grid': {'settings': S.RAW_SETTINGS, 'init_weights_sha256': '<weights_sha256 of your existing init>'},
                     'paths': 'same DATA_ROOT, INIT_ROOT and CHECKPOINT_ROOT as the four-setting grid; a NEW CONFIG_ROOT'},
        'commands': ['huggingface-cli download blab-jhu/KYS-3B-pretraining-data --repo-type dataset --revision <rev> '
                     '--include "tokenized/raw_*" "raw_baselines/*" --local-dir <DATA_ROOT>',
                     'python deploy/portable_3b/kys3b.py validate --operator <operator.raw.yaml>',
                     'python deploy/portable_3b/kys3b.py render --operator <operator.raw.yaml>',
                     'python deploy/portable_3b/kys3b.py relocate-metadata --operator <operator.raw.yaml>',
                     'python deploy/portable_3b/kys3b.py check --operator <operator.raw.yaml> --corpus --hash',
                     'python deploy/portable_3b/kys3b.py verify-init --operator <operator.raw.yaml>',
                     'bash <CONFIG_ROOT>/slurm/submit_grid.sh'],
    }
    RL.write_json(STAGE / TOP[1], handoff)
    (STAGE / TOP[2]).write_text(card(manifest, handoff))
    top = {n: {'bytes': (STAGE / n).stat().st_size, 'sha256': RL.sha256(STAGE / n)} for n in TOP}
    return {'ok': not errs, 'problems': errs, 'n_files': len(files), 'bytes': sum(f['bytes'] for f in files.values()),
            'top_files': top}


def card(man: dict, h: dict) -> str:
    rows = '\n'.join(f'| `{s}` | `{d["counterpart"]}` | {d["rows"]:,} | {d["train_tokens"]:,} | {d["tokenized_tokens"]:,} | '
                     f'{len(d["parts"])} | {d["components"]["raw_strategy_block"]["docs"]:,} |' for s, d in man['settings'].items())
    return f"""# KYS 3B raw baselines

Two raw (un-rewritten) baselines for the KYS 3B rewrite-600m grid. They are an addition to this dataset: the
four-setting release (`RELEASE_MANIFEST.json`, `corpora/<setting>/`, `tokenized/<setting>/`, `tokenizer/`) is unchanged.

| setting | counterpart | rows | train tokens (+1 EOS/doc) | tokenized tokens | parts | raw strategy docs |
|---|---|---|---|---|---|---|
{rows}

Each corpus is the shared raw base (10,527,887 docs, 10,000,000,186 train tokens; `ids/base_ids_sorted.npy`) once,
plus a seed-42 whole-document sample targeting 10B raw tokens of the unique source documents of the final
retained rewritten blocks of its counterpart (`ids/<setting>_selected_perm_order.npy`, in sampling order). All rows have
`source_prompt="original"`; base and strategy membership is defined by the two id lists. Text comes from the public
pool `blab-jhu/dclm-refinedweb-600m-sample` (id → file/row: `mapping/shard_manifest.parquet`). Expected counts and
digests: `mapping/expected_raw_sets.json`. Shuffle: seed 42, the same two-pass bucketed shuffle as the four-setting
corpora (`reports/test_resumable_shuffle_real100k.json`). Validation: `reports/stage5_<setting>.json` (text vs
source, ids, membership), `reports/tokenize_<setting>.json` (shard totals), `reports/stage7_<setting>.json` (every
document: index length == tokens_llama2 + 1, ends in `</s>`, no BOS; base and strategy docs/tokens by the id lists; sha256
of every shard file).

Training: code `{h['code']['repo']}` branch `{h['code']['branch']}` (commit `{h['code']['commit']}`).
Same model, seed 42, optimizer, LR schedule, global batch and branch points as the four-setting grid: 2 settings x 9
segments = 18 segments, 6 annealed finals. **Reuse the existing 3B seed-42 init**; see `HANDOFF_RAW.json`.

Files: `RAW_RELEASE_MANIFEST.json` (size and sha256 of every new file), `HANDOFF_RAW.json`, `RAW_UPLOAD_RECEIPT.json`
(the verified content revision).

Recovery: download only the raw paths at the receipt's revision
(`--include "corpora/raw_*" "tokenized/raw_*" "raw_baselines/*"`), check every file against `RAW_RELEASE_MANIFEST.json`
(`kys3b.py check --corpus --hash`), and re-download any file whose sha256 differs. The corpora can be rebuilt from the public
pool with `code/` (stage12 → stage3 → stage4 → stage4b → stage5 → stage7, each resumable); the id lists and
`mapping/expected_raw_sets.json` pin the exact membership, and `reports/stage4_shuffle_<setting>.json` the part sha256s.
"""


def _expected() -> dict:
    return {**json.loads((STAGE / TOP[0]).read_text())['files'], **record('stage')['top_files']}


def st_remote_check() -> dict:
    a = RL.api()
    remote, rev, private = RL._remote_files(a)
    if private != PRIVATE:
        raise SystemExit(f'{REPO_ID} visibility changed; refusing')
    exp = _expected()
    tmp = WORK / 'remote_check_tmp'
    reuse, conflict = [], []
    for rel in exp:
        if rel in remote:
            m = remote[rel]
            (reuse if m['bytes'] == exp[rel]['bytes'] and RL._remote_sha(a, rel, m, rev, tmp) == exp[rel]['sha256']
             else conflict).append(rel)
    shutil.rmtree(tmp, ignore_errors=True)
    before = {k: v for k, v in remote.items() if k not in exp and k != RECEIPT}
    RL.write_json(REC / 'remote_before.json', {'revision': rev, 'files': before})
    return {'ok': not conflict, 'revision_before': rev, 'existing_files': len(before), 'reused': len(reuse),
            'conflicts': conflict}


class _Breaker(RL._Breaker):
    def stop(self, why, msg):
        RL.write_json(REC / 'upload.json', {'ok': False, 'error': why, 'message': msg.splitlines()[:8]})
        log(f'upload: STOPPED ({why})')
        os._exit(3)


def st_upload() -> dict:
    from huggingface_hub import CommitOperationAdd
    a = RL.api()
    b = _Breaker()
    hl = logging.getLogger('huggingface_hub')
    hl.addHandler(b)
    try:
        a.upload_large_folder(repo_id=REPO_ID, repo_type='dataset', folder_path=str(STAGE), private=PRIVATE,
                              ignore_patterns=list(TOP), num_workers=int(os.environ.get('KYS3B_UPLOAD_WORKERS', 8)),
                              print_report=False)
    finally:
        hl.removeHandler(b)
    remote, _, _ = RL._remote_files(a)
    commit = None
    if not all(n in remote and remote[n]['blob_id'] == RL.git_blob_sha1(STAGE / n) for n in TOP):
        commit = a.create_commit(REPO_ID, repo_type='dataset',
                                 operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(STAGE / n)) for n in TOP],
                                 commit_message=f'KYS 3B raw baselines: manifest, handoff, card (code {(record("code") or {}).get("commit", "?")[:12]})').oid
    _, rev, private = RL._remote_files(a)
    return {'ok': private == PRIVATE, 'top_commit': commit, 'revision': rev, 'rate_limited_429': b.n_rate}


def st_verify_remote() -> dict:
    a = RL.api()
    remote, rev, private = RL._remote_files(a)
    exp = _expected()
    before = json.loads((REC / 'remote_before.json').read_text())['files']
    tmp = WORK / 'verify_tmp'
    missing = [r for r in exp if r not in remote]
    bad = [r for r in exp if r in remote and (remote[r]['bytes'] != exp[r]['bytes']
                                              or RL._remote_sha(a, r, remote[r], rev, tmp) != exp[r]['sha256'])]
    changed = [r for r, m in before.items() if remote.get(r) != m]
    unexpected = sorted(set(remote) - set(exp) - set(before) - {RECEIPT})
    shutil.rmtree(tmp, ignore_errors=True)
    return {'ok': private == PRIVATE and not (missing or bad or changed or unexpected), 'content_revision': rev,
            'checked_new': len(exp), 'existing_unchanged': len(before) - len(changed), 'missing': missing,
            'mismatched': bad, 'existing_changed': changed, 'unexpected': unexpected}


def st_fresh_download() -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rev = record('verify_remote')['content_revision']
    man = json.loads((STAGE / TOP[0]).read_text())
    root = WORK / f'fresh_{rev[:12]}'
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    picks = [TOP[0], f'{NS}/ids/base_ids_sorted.npy', *(f'{NS}/ids/{s}_selected_perm_order.npy' for s in S.RAW_SETTINGS)]
    for s, d in man['settings'].items():
        picks += [d['parts'][0]['path'], d['parts'][-1]['path']]
        x = d['tokenized_shards'][len(d['tokenized_shards']) // 2]
        picks += [x['ds'], x['index'], x['metadata']]
    errs, got = [], {}
    for rel in picks:
        p = Path(hf_hub_download(REPO_ID, rel, repo_type='dataset', revision=rev, cache_dir=str(root / 'cache'),
                                 local_dir=str(root / 'files')))
        got[rel] = p
        want = man['files'][rel]['sha256'] if rel in man['files'] else record('stage')['top_files'][rel]['sha256']
        if RL.sha256(p) != want:
            errs.append(f'{rel}: sha256 differs')
    for s, d in man['settings'].items():
        for x in (d['parts'][0], d['parts'][-1]):
            if pq.ParquetFile(got[x['path']]).metadata.num_rows != x['rows']:
                errs.append(f'{x["path"]}: rows differ')
        if np.load(got[f'{NS}/ids/{s}_selected_perm_order.npy']).size != d['components']['raw_strategy_block']['docs']:
            errs.append(f'{s}: id list size')
    shutil.rmtree(root, ignore_errors=True)
    return {'ok': not errs, 'revision': rev, 'files': picks, 'problems': errs}


def st_receipt() -> dict:
    from huggingface_hub import CommitOperationAdd
    rec = {'dataset': REPO_ID, 'content_revision': record('verify_remote')['content_revision'],
           'manifest': TOP[0], 'manifest_sha256': record('stage')['top_files'][TOP[0]]['sha256'],
           'code': {k: (record('code') or {}).get(k) for k in ('repo', 'branch', 'commit')},
           'verify_remote': {k: v for k, v in record('verify_remote').items() if k != 'ok'},
           'fresh_download': record('fresh_download'), 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    p = WORK / 'RAW_UPLOAD_RECEIPT.json'
    RL.write_json(p, rec)
    a = RL.api()
    oid = a.create_commit(REPO_ID, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=RECEIPT, path_or_fileobj=str(p))],
                          commit_message='KYS 3B raw baselines: upload receipt').oid
    return {'ok': True, 'receipt_commit': oid, 'content_revision': rec['content_revision']}


FUN = {'gate': st_gate, 'code': st_code, 'stage': st_stage, 'remote_check': st_remote_check, 'upload': st_upload,
       'verify_remote': st_verify_remote, 'fresh_download': st_fresh_download, 'receipt': st_receipt}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['run', 'status'])
    ap.add_argument('--through', choices=STAGES, default=STAGES[-1])
    a = ap.parse_args(argv)
    if a.command == 'status':
        for s in STAGES:
            r = record(s)
            print(f'{s:15s} {"-" if r is None else ("ok" if r.get("ok") else "FAILED")}')
        return 0
    tok = Path.home() / '.cache' / 'huggingface' / 'token'      # ~/.bashrc points HF_HOME at another project's cache
    if not os.environ.get('HF_TOKEN') and not os.environ.get('HF_TOKEN_PATH') and tok.is_file():
        os.environ['HF_TOKEN_PATH'] = str(tok)
    REC.mkdir(parents=True, exist_ok=True)
    lock = open(REC / '.uploader.lock', 'w')
    import fcntl
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another release_raw_3b.py holds the lock; single uploader only')
    for s in STAGES[:STAGES.index(a.through) + 1]:
        r = record(s)
        if r and r.get('ok'):
            log(f'{s}: ok (recorded)')
            continue
        log(f'{s}: running')
        t0 = time.time()
        r = FUN[s]()
        r.update(stage=s, seconds=round(time.time() - t0, 1), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 job=os.environ.get('SLURM_JOB_ID'))
        RL.write_json(REC / f'{s}.json', r)
        log(f'{s}: {"ok" if r["ok"] else "FAILED"}')
        if not r['ok']:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
