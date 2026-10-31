"""The shared seed-42 initialization: digest, finalize (create once, never overwrite), verify, seed a trunk.

Layout under the operator's INIT_ROOT:
    <INIT_ROOT>/_init_3B_seed42/0/                 nanotron step-0 checkpoint written by the 1-step init run
    <INIT_ROOT>/_init_3B_seed42/init_manifest.json files + weights_sha256 (+ provenance)
Every setting's trunk1 starts from a byte-identical copy of that step 0 (trunk/0 + latest.txt = 0), so all four
settings share one initialization. Used by kys3b.py and, unchanged, by the historical Skipjack tools.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import science_3b as S

MANIFEST = 'init_manifest.json'


class InitError(SystemExit):
    pass


def init_parent(init_root) -> Path:
    return Path(init_root) / S.INIT_DIRNAME


def init_digest(step_dir: Path) -> dict:
    """Digest of a step directory's weights + config (optimizer and lr_scheduler state excluded)."""
    step_dir = Path(step_dir)
    files = sorted(p for p in step_dir.rglob('*') if p.is_file())
    h = hashlib.sha256()
    for p in files:
        if p.parts[-2:-1] != ('lr_scheduler',) and 'optimizer' not in p.parts:   # weights + config only
            h.update(p.relative_to(step_dir).as_posix().encode())
            with open(p, 'rb') as f:
                while chunk := f.read(16 << 20):
                    h.update(chunk)
    return {'files': len(files), 'weights_sha256': h.hexdigest()}


def read_manifest(init_root) -> dict:
    m = init_parent(init_root) / MANIFEST
    if not m.is_file():
        raise InitError(f'no {m}: create the shared init first (deploy/portable_3b/create_init.sh)')
    return json.loads(m.read_text())


def verify_init(init_root) -> dict:
    """The init exists and its step 0 still matches its manifest."""
    want = read_manifest(init_root)
    got = init_digest(init_parent(init_root) / '0')
    if got != {'files': want['files'], 'weights_sha256': want['weights_sha256']}:
        raise InitError(f'{init_parent(init_root)}/0 does not match its {MANIFEST} ({got} vs {want["weights_sha256"]})')
    return want


def finalize_init(scratch_ckpt: Path, init_root: Path, provenance: dict) -> dict:
    """Move the init run's step 0 into <INIT_ROOT>/_init_3B_seed42 with its manifest. Refuses to touch an existing
    init (complete or not); the move is a rename of a fully written, re-digested copy."""
    step0 = Path(scratch_ckpt) / '0'
    if not (step0 / 'model_config.json').is_file():
        raise InitError(f'{step0} has no model_config.json: the init run did not write its step-0 checkpoint')
    parent = init_parent(init_root)
    if (parent / MANIFEST).is_file():
        raise InitError(f'{parent} already holds a finished init; it is reused as is (never overwritten)')
    if parent.exists():
        raise InitError(f'{parent} exists without {MANIFEST}: a previous finalize did not complete. Inspect it; '
                        f'remove it yourself only if no trunk was seeded from it')
    tmp = parent.with_name(parent.name + '.tmp')
    if tmp.exists():
        raise InitError(f'{tmp} exists (an interrupted finalize); inspect and remove it, then rerun')
    tmp.mkdir(parents=True)
    shutil.copytree(step0, tmp / '0', symlinks=True)
    src, copy = init_digest(step0), init_digest(tmp / '0')
    if src != copy:
        raise InitError(f'copy of {step0} differs from the source ({copy} vs {src}); {tmp} left for inspection')
    m = {**copy, 'seed': S.SEED, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'source': str(step0),
         'n_layers': S.N_LAYERS, 'n_params': S.N_PARAMS, 'init_std': S.INIT_STD, **provenance}
    (tmp / MANIFEST).write_text(json.dumps(m, indent=1) + '\n')
    os.rename(tmp, parent)
    return m


def seed_trunk(trunk: Path, init_step0: Path, manifest: Path) -> bool:
    """Shared init -> <trunk>/0 + latest.txt, only if the trunk has no latest.txt; identical weights for every setting.
    Never seeds over progress: a trunk with step directories but no latest.txt is refused."""
    trunk = Path(trunk)
    if (trunk / 'latest.txt').is_file():
        return False
    if trunk.exists() and any(p.is_dir() and p.name.isdigit() for p in trunk.iterdir()):
        raise InitError(f'{trunk} holds step directories but no latest.txt; refusing to seed over a progressed trunk')
    want = json.loads(Path(manifest).read_text())
    if init_digest(init_step0) != {'files': want['files'], 'weights_sha256': want['weights_sha256']}:
        raise InitError(f'{init_step0} does not match {manifest}; refusing to seed')
    trunk.mkdir(parents=True, exist_ok=True)
    tmp = trunk / '0.seeding'
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(init_step0, tmp, symlinks=True)
    if init_digest(tmp)['weights_sha256'] != want['weights_sha256']:
        raise InitError(f'{tmp} differs from the init after copying')
    os.rename(tmp, trunk / '0')
    (trunk / 'latest.txt').write_text('0\n')
    return True
