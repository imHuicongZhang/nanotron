"""Install instructions and pins (CPU only, no network).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_install_3b.py

Asserts:
  * requirements/1-torch-cu128.txt installs exactly torch==2.8.0+cu128 from the official PyTorch CUDA 12.8 index;
  * 1-torch-cu128.txt + 2-stack.txt pin exactly the frozen Skipjack env (deploy/skipjack_3b_600m/requirements-3b.lock.txt):
    every package, every version; the two binary wheels by URL + sha256 (nv-grouped-gemm: the release wheel of the
    lock's git commit, version 1.1.4.post8);
  * every requirement is an exact pin (== or URL#sha256); nothing floats;
  * nanotron's C++ helper is built with the interpreter that imports it (Makefile PYTHON, compile_helper passes
    sys.executable); with Python headers available the helper is compiled and imported from a temporary copy.
The network-dependent part (a real clean install with install_env.sh) is a separate run, reported in the release notes.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from conftest import CODE, HERE

REQ = HERE / 'requirements'
LOCK = CODE / 'deploy' / 'skipjack_3b_600m' / 'requirements-3b.lock.txt'


def _reqs(path: Path) -> dict[str, str]:
    out = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#') or ln.startswith('-'):
            continue
        if ' @ ' in ln:
            n, u = ln.split(' @ ', 1)
            out[re.sub(r'[-_.]+', '-', n.strip()).lower()] = '@' + u.strip()
        else:
            n, v = ln.split('==')
            out[re.sub(r'[-_.]+', '-', n.strip()).lower()] = v.strip()
    return out


def test_torch_from_the_official_cu128_index():
    lines = [ln.strip() for ln in (REQ / '1-torch-cu128.txt').read_text().splitlines() if ln.strip() and not ln.startswith('#')]
    assert lines == ['--index-url https://download.pytorch.org/whl/cu128', 'torch==2.8.0+cu128']


def test_portable_pins_equal_the_frozen_lock():
    lock = _reqs(LOCK)
    mine = {**_reqs(REQ / '1-torch-cu128.txt'), **_reqs(REQ / '2-stack.txt')}
    assert set(mine) == set(lock)
    diff = {n: (lock[n], mine[n]) for n in lock if lock[n] != mine[n]}
    assert set(diff) == {'nv-grouped-gemm'}, diff
    assert lock['nv-grouped-gemm'] == '@git+https://github.com/fanshiqing/grouped_gemm@efe8c40eaf4c8ef57191e0ea9aa4117aa5b1a8f2'
    assert mine['nv-grouped-gemm'].startswith('@https://github.com/fanshiqing/grouped_gemm/releases/download/v1.1.4.post8/'
                                              'nv_grouped_gemm-1.1.4.post8%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-')
    for n in ('nv-grouped-gemm', 'flash-attn'):
        assert re.search(r'#sha256=[0-9a-f]{64}$', mine[n]), n


def test_every_requirement_is_exact():
    for f in ('1-torch-cu128.txt', '2-stack.txt', 'test.txt'):
        for ln in (REQ / f).read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#') or ln.startswith('--index-url'):
                continue
            assert re.match(r'^[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+$', ln) or re.search(r' @ https://\S+#sha256=[0-9a-f]{64}$', ln), (f, ln)


def test_install_script_order_and_no_hardware_defaults():
    s = (HERE / 'install_env.sh').read_text()
    assert subprocess.run(['bash', '-n', str(HERE / 'install_env.sh')]).returncode == 0
    steps = [s.index(x) for x in ('requirements/1-torch-cu128.txt', '$REQ/1-torch-cu128.txt', '$REQ/2-stack.txt',
                                   '--no-deps -e "$CODE"', 'make -C "$CODE/src/nanotron/data/nemo_dataset" PYTHON="$P"',
                                   '"$P" -m pip check', 'check_env.py"')]
    assert steps[1:] == sorted(steps[1:])
    assert 'TORCH_CUDA_ARCH_LIST' not in s and 'nvcc' not in s.split('set -euo pipefail', 1)[1]
    assert 'Python.h' in s and 'sys.version_info[:2] == (3, 11)' in s


def test_helper_build_uses_the_importing_interpreter(tmp_path):
    mk = (CODE / 'src/nanotron/data/nemo_dataset/Makefile').read_text()
    assert 'PYTHON ?= python3' in mk and '$(PYTHON) -m pybind11 --includes' in mk and "get_config_var('EXT_SUFFIX')" in mk
    assert 'python3-config' not in mk.split('# PYTHON is', 1)[1].split('\n', 3)[-1]
    assert 'f"PYTHON={sys.executable}"' in (CODE / 'src/nanotron/data/nemo_dataset/dataset_utils.py').read_text()
    inc = sysconfig.get_paths()['include']
    if not os.path.isfile(os.path.join(inc, 'Python.h')) or importlib.util.find_spec('pybind11') is None:
        pytest.skip(f'this interpreter has no Python.h in {inc} (or no pybind11): the helper cannot be compiled for it; '
                    f'install_env.sh refuses such an interpreter')
    d = tmp_path / 'nemo_dataset'
    d.mkdir()
    for f in ('Makefile', 'helpers.cpp'):
        shutil.copy(CODE / 'src/nanotron/data/nemo_dataset' / f, d / f)
    r = subprocess.run(['make', '-C', str(d), f'PYTHON={sys.executable}'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    so = d / f'helpers{sysconfig.get_config_var("EXT_SUFFIX")}'
    spec = importlib.util.spec_from_file_location('helpers', so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, 'build_blending_indices')


def test_check_env_reads_every_pin():
    import check_env
    pins = check_env.pins()
    assert len(pins) == len(_reqs(REQ / '1-torch-cu128.txt')) + len(_reqs(REQ / '2-stack.txt'))
    assert pins['torch'] == '2.8.0+cu128' and pins['nv-grouped-gemm'].startswith('@https://')
