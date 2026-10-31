#!/usr/bin/env python
"""Check a KYS 3B training environment against the pinned stack (run with the env's own python; CPU only).

    <env>/bin/python deploy/portable_3b/check_env.py [--json <out>]

required (exit 1 if any fails):
  * every pin of requirements/{1-torch-cu128,2-stack}.txt is installed at exactly that version (URL wheels: their
    package version, and grouped_gemm_backend's sha256);
  * torch is 2.8.0+cu128 and was built for CUDA 12.8;
  * imports that need no GPU: torch, numpy, datatrove, transformers, tokenizers, safetensors, wandb, yaml, and from
    nanotron: config, serialize, the TokenizedBytes data path, the compiled C++ dataset helper;
  * the C++ dataset helper was built for this interpreter (extension suffix).
reported only (they need a GPU host: CUDA driver / devices):
  * flash_attn, grouped_gemm and nanotron.models.llama imports, torch.cuda.is_available(), device names.
A pass here is NOT CUDA runtime validation; the first real check of that is the startup validation on the GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import re
import sys
import sysconfig
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
REQ = HERE / 'requirements'
REQ_FILES = ['1-torch-cu128.txt', '2-stack.txt']
REQUIRED_IMPORTS = ['torch', 'numpy', 'datatrove', 'transformers', 'tokenizers', 'safetensors', 'wandb', 'yaml',
                    'nanotron.config', 'nanotron.serialize.optimizer', 'nanotron.data.tokenized_bytes',
                    'nanotron.data.nemo_dataset.helpers']
GPU_IMPORTS = ['flash_attn', 'grouped_gemm', 'nanotron.models.llama']
GROUPED_GEMM_SO_SHA256 = '66f79446cf71fcc204db20ba0aea77458e669d02fb9008b86a9815a21954ad21'   # = the Skipjack env's


def pins() -> dict[str, str]:
    """name -> exact version ('@url' requirements: the version is checked through the installed dist only)."""
    out = {}
    for f in REQ_FILES:
        for ln in (REQ / f).read_text().splitlines():
            ln = ln.strip() if ' @ ' in ln else ln.split('#', 1)[0].strip()
            if not ln or ln.startswith('-'):
                continue
            if ' @ ' in ln:
                out[ln.split(' @ ')[0].strip()] = '@' + ln.split(' @ ')[1].split('#')[0]
            else:
                n, v = ln.split('==')
                out[n.strip()] = v.strip()
    return out


def _norm(n: str) -> str:
    return re.sub(r'[-_.]+', '-', n).lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CODE / 'src'))
    errs, report = [], {'python': sys.version.split()[0], 'executable': sys.executable, 'versions': {}, 'gpu_only': {}}
    installed = {_norm(d.metadata['Name']): d.version for d in md.distributions()}
    for name, want in pins().items():
        have = installed.get(_norm(name))
        report['versions'][name] = have
        if have is None:
            errs.append(f'{name}: not installed')
        elif not want.startswith('@') and have != want:
            errs.append(f'{name}: {have} installed, pinned {want}')
    for name, want in (('flash-attn', '2.8.3'), ('nv-grouped-gemm', '1.1.4.post8')):
        if installed.get(name) != want:
            errs.append(f'{name}: {installed.get(name)}, pinned {want} (prebuilt wheel)')
    gg = [p for f in md.files('nv-grouped-gemm') or [] if (p := Path(f.locate())).name.startswith('grouped_gemm_backend')] \
        if installed.get('nv-grouped-gemm') else []
    if not gg or hashlib.sha256(gg[0].read_bytes()).hexdigest() != GROUPED_GEMM_SO_SHA256:
        errs.append(f'grouped_gemm_backend: not the pinned binary {GROUPED_GEMM_SO_SHA256[:12]}... ({gg})')
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - report every failure
            errs.append(f'import {mod}: {type(e).__name__}: {e}')
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda}
        if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8':
            errs.append(f'torch {torch.__version__} built for CUDA {torch.version.cuda}; need 2.8.0+cu128 / 12.8')
        report['gpu_only']['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report['gpu_only']['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    helper = list((CODE / 'src/nanotron/data/nemo_dataset').glob('helpers*.so'))
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not any(p.name == f'helpers{suffix}' for p in helper):
        errs.append(f'no helpers{suffix} in src/nanotron/data/nemo_dataset (found {[p.name for p in helper]}); '
                    f'run: make -C src/nanotron/data/nemo_dataset PYTHON={sys.executable}')
    for mod in GPU_IMPORTS:
        try:
            importlib.import_module(mod)
            report['gpu_only'][mod] = 'imported'
        except Exception as e:  # noqa: BLE001
            report['gpu_only'][mod] = f'not importable here ({type(e).__name__}: {str(e).splitlines()[0][:200]})'
    report['errors'] = errs
    report['ok'] = not errs
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    for k, v in report['gpu_only'].items():
        print(f'[check_env] gpu-host only (not required here): {k}: {v}')
    for e in errs:
        print(f'[check_env] FAIL {e}')
    print(f'[check_env] {"OK" if not errs else "FAILED"}: {len(report["versions"])} pins, '
          f'{len(REQUIRED_IMPORTS)} required imports' + ('' if not errs else f', {len(errs)} problem(s)'))
    return 0 if not errs else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
