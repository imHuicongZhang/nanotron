#!/usr/bin/env python
"""Download and verify the pinned inputs of the skipjack raw-baseline grid (CPU only).

    python deploy/skipjack_1p5b/download_inputs.py --data-root <data root>/raw --init-root <ckpt root>/init \
        --report <data root>/reports/inputs_verification.json

  1. snapshot_download of blab-jhu/KYS-Pre-Rewritten and wytro/Know-Your-Sources-init at the exact
     revisions below, with local_dir = the root (files already present are re-used by the hub client
     only when their hash matches its ledger; every file is re-hashed below anyway);
  2. every file listed in manifest.json (16 parquet files x 7 settings + tokenizer) is sha256-checked;
  3. tools/hash_init_checkpoint.py --check runs for seeds 42/43/44 and each init holds exactly 180 files.
Writes a JSON report and exits non-zero on any mismatch. Does not touch the scored pool or the
rewritten corpora (allow_patterns restricts the dataset download to what training needs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[2]
DATA = ('blab-jhu/KYS-Pre-Rewritten', '4ba8df66b51c210bb2bc51d2f5f93c48e850171d')
INIT = ('wytro/Know-Your-Sources-init', '87e1335647a3105471928e0fd261d92846021d1a')
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--init-root', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    rep = {'data': {'repo': DATA[0], 'revision': DATA[1], 'local_dir': str(args.data_root)},
           'init': {'repo': INIT[0], 'revision': INIT[1], 'local_dir': str(args.init_root)}, 'failed': []}

    for root, need in ((args.data_root, 125e9), (args.init_root, 30e9)):
        free = shutil.disk_usage(root).free
        print(f'[dl] {root}: {free / 1e12:.2f} TB free')
        if free < need:
            sys.exit(f'{root}: only {free / 1e9:.0f} GB free, need ~{need / 1e9:.0f} GB')

    snapshot_download(DATA[0], repo_type='dataset', revision=DATA[1], local_dir=args.data_root,
                      allow_patterns=['manifest.json', 'raw_text/*', 'tokenizer/*', 'selection/*', 'reports/*',
                                      '*.md', '.gitattributes'],
                      max_workers=args.workers)
    snapshot_download(INIT[0], revision=INIT[1], local_dir=args.init_root, max_workers=args.workers)
    print('[dl] downloads finished; verifying')

    man = json.loads((args.data_root / 'manifest.json').read_text())
    tok = args.data_root / 'tokenizer'
    rep['tokenizer'] = {}
    for name, want in sorted(man['tokenizer']['sha256'].items()):
        ok = (tok / name).is_file() and sha256(tok / name) == want
        rep['tokenizer'][name] = ok
        if not ok:
            rep['failed'].append(f'tokenizer/{name}')
    rep['settings'] = {}
    for setting, s in man['settings'].items():
        d = args.data_root / 'raw_text' / setting
        bad = [n for n, rec in sorted(s['files'].items()) if not (d / n).is_file() or sha256(d / n) != rec['sha256']]
        extra = sorted({p.name for p in d.glob('*.parquet')} - set(s['files']))
        rep['settings'][setting] = {'files': len(s['files']), 'sha256_mismatch': bad, 'extra': extra,
                                    'expected_total_tokens': s.get('expected_total_tokens')}
        rep['failed'] += [f'raw_text/{setting}/{n}' for n in bad + extra]
        print(f'[verify] {setting}: {len(s["files"])} files, {len(bad)} mismatched, {len(extra)} extra')

    rep['init_seeds'] = {}
    for seed in SEEDS:
        d = args.init_root / f'_init_1.5B_seed{seed}' / '0'
        r = subprocess.run([sys.executable, REPO / 'tools/hash_init_checkpoint.py', d, '--check',
                            args.init_root / f'init_1.5B_seed{seed}.hash.json'], capture_output=True, text=True)
        n = sum(1 for p in d.rglob('*') if p.is_file())
        rep['init_seeds'][seed] = {'hash_check_rc': r.returncode, 'files': n, 'tail': (r.stdout + r.stderr)[-400:]}
        if r.returncode or n != 180:
            rep['failed'].append(f'init seed {seed}: rc={r.returncode} files={n}')
        print(f'[verify] init seed {seed}: hash rc={r.returncode}, {n} files')

    args.report.write_text(json.dumps(rep, indent=1))
    print(f'[verify] report -> {args.report}; failed: {rep["failed"]}')
    sys.exit(1 if rep['failed'] else 0)


if __name__ == '__main__':
    main()
