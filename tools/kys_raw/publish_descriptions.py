#!/usr/bin/env python
"""Update ONLY the descriptive metadata of blab-jhu/KYS-Pre-Rewritten after the strategy-linked audit.

One Hub commit containing:
  README.md                                   tools/kys_raw/KYS-Pre-Rewritten.README.md
  reports/RAW_SELECTED_BASELINES_PROVENANCE.md configs/1.5B-baseline/reports/strategy_linked_audit/ (byte copy)
  manifest.json                               the current Hub manifest with (a) settings.<s>.role replaced by
                                              publish_raw_text.ROLE for the four strategy-linked settings and
                                              (b) a new top-level `strategy_linked_audit` block. Nothing else may
                                              change; the script refuses otherwise.
No data file, selection array, tokenizer file or count is touched. After the commit it checks that every
pre-existing path other than these three is unchanged (same LFS sha256 / blob id) and that manifest.json
round-trips.

    python tools/kys_raw/publish_descriptions.py --base-revision <sha> --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_raw_text import ROLE  # noqa: E402

REPO_ID = 'blab-jhu/KYS-Pre-Rewritten'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'configs/1.5B-baseline/reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md'
README = REPO / 'tools/kys_raw/KYS-Pre-Rewritten.README.md'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-revision', required=True, help='the Hub revision the update is based on (must be the current head)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--code-commit', required=True, help='pushed nanotron commit holding the report mirror and this script')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    head = api.repo_info(REPO_ID, repo_type='dataset').sha
    if head != a.base_revision:
        sys.exit(f'Hub head is {head}, not --base-revision {a.base_revision}; re-read before editing')
    a.out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(hf_hub_download(REPO_ID, 'manifest.json', repo_type='dataset', revision=head,
                                          local_dir=a.out / 'before')).read_text())
    new = json.loads(json.dumps(old))
    for s, role in ROLE.items():
        new['settings'][s]['role'] = role
    new['strategy_linked_audit'] = {
        'date': '2026-09-28',
        'report': 'reports/RAW_SELECTED_BASELINES_PROVENANCE.md',
        'evidence': f'https://github.com/imHuicongZhang/nanotron/tree/{a.code_commit}/configs/1.5B-baseline/reports/strategy_linked_audit',
        'rewritten_source': 'wytro/Know-Your-Sources@9e5ff24149c2957c30f0c8fdd051a8eb3b75baad (pinned by sha256 of all files read)',
        'summary': ('No implementation defect. Each strategy-linked raw corpus = shared 5B anchor (identical to the counterpart '
                    "arm's anchor) + a seed-42 uniform random 5B (whole documents) of the unique source documents of the "
                    "counterpart's final rewritten half; rebuilt independently and equal to the published files. They are "
                    'equal-token-budget controls, not identical-document controls; raw_rewire_inspired is conditional on '
                    "REWIRE's post-rewrite filter. Token stream: text + </s> per document, no <s>, as in the rewritten arms."),
        'data_changed': False,
    }
    # nothing but the four roles and the new block may differ
    for k in set(old) | set(new):
        if k in ('settings', 'strategy_linked_audit'):
            continue
        if old.get(k) != new.get(k):
            sys.exit(f'refusing: top-level key {k} would change')
    for s in old['settings']:
        o, n = dict(old['settings'][s]), dict(new['settings'][s])
        if s in ROLE:
            o.pop('role'), n.pop('role')
        if o != n:
            sys.exit(f'refusing: settings.{s} would change beyond its role')
    mpath = a.out / 'manifest.json'
    mpath.write_text(json.dumps(new, indent=2) + '\n')
    changed = [s for s in ROLE if old['settings'][s].get('role') != new['settings'][s]['role']]
    print(f'roles changed: {changed}; new block: strategy_linked_audit')
    if a.dry_run:
        return

    before = {f.path: f for f in api.list_repo_tree(REPO_ID, repo_type='dataset', revision=head, recursive=True) if hasattr(f, 'size')}
    info = api.create_commit(
        repo_id=REPO_ID, repo_type='dataset', revision='main', parent_commit=head,
        commit_message='Documentation only: strategy-linked audit report; README and manifest role descriptions (no data change)',
        operations=[CommitOperationAdd('README.md', str(README)),
                    CommitOperationAdd('reports/RAW_SELECTED_BASELINES_PROVENANCE.md', str(REPORT)),
                    CommitOperationAdd('manifest.json', str(mpath))])
    rev = info.oid
    after = {f.path: f for f in api.list_repo_tree(REPO_ID, repo_type='dataset', revision=rev, recursive=True) if hasattr(f, 'size')}
    touched = {'README.md', 'manifest.json', 'reports/RAW_SELECTED_BASELINES_PROVENANCE.md'}

    def ident(f):
        return f.lfs.sha256 if f.lfs else f.blob_id
    changed_paths = sorted(p for p in before if p not in touched and (p not in after or ident(after[p]) != ident(before[p])))
    added = sorted(set(after) - set(before))
    got = json.loads(Path(hf_hub_download(REPO_ID, 'manifest.json', repo_type='dataset', revision=rev,
                                          local_dir=a.out / 'after')).read_text())
    rec = {'repo': REPO_ID, 'base_revision': head, 'revision': rev, 'paths_before': len(before), 'paths_after': len(after),
           'other_paths_changed': changed_paths, 'paths_added': added, 'manifest_roundtrip_equal': got == new,
           'ok': not changed_paths and added == ['reports/RAW_SELECTED_BASELINES_PROVENANCE.md'] and got == new}
    (a.out / 'published_descriptions.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec, indent=2))
    if not rec['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()
