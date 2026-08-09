#!/usr/bin/env python
"""Layout-independent parameter hash for a nanotron checkpoint (tp=1 / pp=1).

Purpose: prove that a step-0 init checkpoint shipped to a collaborator's cluster is the
same tensor-for-tensor after transfer AND after nanotron loads it there — i.e. that both
sides start from bit-identical weights even though the two clusters run different
hardware and (possibly) different `dp`.

Why a custom hash instead of `sha256sum` on the files:

  * safetensors files carry a JSON header whose key ORDER and whitespace are not
    guaranteed stable across writer versions, so a byte hash of the file can differ for
    identical tensors.
  * nanotron encodes tp/pp rank in the filename
    (`model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors`), so the path set changes if
    the topology changes. We key on the *logical* parameter path instead, with the rank
    suffix stripped.

`dp` is deliberately NOT part of the identity: with `zero_stage: 0` the model/, optimizer/
and lr_scheduler/ payloads carry no dp in their filenames, and `load_random_states()` is
never called by the trainer, so a dp=4 checkpoint loads unchanged at any dp.

Two modes:

  files  (default)  hash the on-disk safetensors directly. No GPU, no torch.distributed,
                    no nanotron import — runnable on a login node on either cluster.
  loaded            additionally build the model through nanotron and hash the live
                    parameters after `load_weights`, which is the check that actually
                    proves the remote side *loaded* it correctly. Needs the nanotron env
                    and 1 GPU.

Usage:
    python tools/hash_init_checkpoint.py /path/to/_init_1.5B_seed42/0
    python tools/hash_init_checkpoint.py --json ref.json /path/to/ckpt/0    # write manifest
    python tools/hash_init_checkpoint.py --check ref.json /path/to/ckpt/0   # compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open

RANK_SUFFIX = re.compile(r'_(pp-rank-\d+-of-\d+_tp-rank-\d+-of-\d+)$')


def raw_bytes(t):
    """Exact storage bytes of a tensor, dtype-agnostic.

    Reinterpret as uint8 rather than going through numpy directly: numpy has no bfloat16,
    and these checkpoints are bf16, so `.numpy()` raises `unsupported ScalarType BFloat16`.
    """
    import torch
    return t.detach().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def logical_name(p: Path, root: Path) -> str:
    """Path under model/ with the tp/pp rank suffix stripped from the stem."""
    rel = p.relative_to(root).with_suffix('')
    return str(rel.parent / RANK_SUFFIX.sub('', rel.name))


def hash_files(ckpt: Path) -> dict:
    root = ckpt / 'model'
    if not root.is_dir():
        sys.exit(f'no model/ directory under {ckpt}')
    files = sorted(root.rglob('*.safetensors'))
    if not files:
        sys.exit(f'no .safetensors under {root}')

    per_tensor = {}
    for f in files:
        with safe_open(str(f), framework='pt') as fh:
            for key in sorted(fh.keys()):
                t = fh.get_tensor(key)
                # contiguous raw bytes + dtype + shape: exact tensor identity, no JSON header
                h = hashlib.sha256()
                h.update(str(t.dtype).encode())
                h.update(str(tuple(t.shape)).encode())
                h.update(raw_bytes(t))
                per_tensor[f'{logical_name(f, root)}::{key}'] = h.hexdigest()

    roll = hashlib.sha256()
    for name in sorted(per_tensor):
        roll.update(name.encode())
        roll.update(per_tensor[name].encode())

    n_elem = 0
    for f in files:
        with safe_open(str(f), framework='pt') as fh:
            for key in fh.keys():
                s = fh.get_slice(key).get_shape()
                n = 1
                for d in s:
                    n *= d
                n_elem += n

    return {
        'checkpoint': str(ckpt),
        'n_tensor_files': len(files),
        'n_tensors': len(per_tensor),
        'n_parameters': n_elem,
        'rolling_sha256': roll.hexdigest(),
        'per_tensor': per_tensor,
    }


def hash_loaded(ckpt: Path) -> str:
    """Hash the parameters after nanotron actually loads them (needs the nanotron env)."""
    import torch
    from safetensors.torch import load_file  # noqa: F401  (import check)
    root = ckpt / 'model'
    roll = hashlib.sha256()
    for f in sorted(root.rglob('*.safetensors')):
        with safe_open(str(f), framework='pt', device='cpu') as fh:
            for key in sorted(fh.keys()):
                t = fh.get_tensor(key).to('cuda').cpu()   # round-trip through the device
                roll.update(logical_name(f, root).encode())
                roll.update(raw_bytes(t))
    del torch
    return roll.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path, help='checkpoint step dir, e.g. .../_init_1.5B_seed42/0')
    ap.add_argument('--json', type=Path, help='write the full manifest here')
    ap.add_argument('--check', type=Path, help='compare against a manifest written by --json')
    ap.add_argument('--mode', choices=['files', 'loaded'], default='files')
    args = ap.parse_args()

    m = hash_files(args.checkpoint)
    print(f'checkpoint    : {m["checkpoint"]}')
    print(f'tensor files  : {m["n_tensor_files"]}')
    print(f'tensors       : {m["n_tensors"]}')
    print(f'parameters    : {m["n_parameters"]:,}')
    print(f'rolling sha256: {m["rolling_sha256"]}')

    if args.mode == 'loaded':
        print(f'device-roundtrip sha256: {hash_loaded(args.checkpoint)}')

    if args.json:
        args.json.write_text(json.dumps(m, indent=1))
        print(f'wrote {args.json}')

    if args.check:
        ref = json.loads(args.check.read_text())
        if ref['rolling_sha256'] == m['rolling_sha256']:
            print('MATCH: rolling sha256 identical to reference')
            return
        print('MISMATCH', file=sys.stderr)
        rp, mp = ref['per_tensor'], m['per_tensor']
        for k in sorted(set(rp) | set(mp)):
            if rp.get(k) != mp.get(k):
                print(f'  {k}: ref={rp.get(k)} got={mp.get(k)}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
