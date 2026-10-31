#!/usr/bin/env bash
# Build the KYS 3B training environment from the pinned stack. Portable: it reads nothing Skipjack-specific.
#
#   bash deploy/portable_3b/install_env.sh <CPython 3.11 interpreter that ships Python.h> <new env directory>
#
# Build host: network access to pypi.org, download.pytorch.org and github.com; a C++ compiler (g++ >= 9, or $CXX) and
# make. No GPU and no CUDA toolkit are needed: torch, flash-attn and nv-grouped-gemm are prebuilt wheels.
# GPU (training) hosts additionally need: an NVIDIA driver that supports CUDA 12.8, and at run time a C compiler plus
# the same interpreter's headers (Triton JIT-compiles small launcher stubs against Python.h on first use).
#
# Order (each step is pinned; README.md "Install" explains why):
#   0  venv from the given CPython 3.11 (the flash-attn and grouped-gemm wheels are cp311; nanotron and Triton compile
#      against the interpreter's headers: conda / uv / pyenv Pythons ship them; distro Pythons need python3.11-devel)
#   1  torch==2.8.0+cu128 from the official PyTorch CUDA 12.8 index        (requirements/1-torch-cu128.txt)
#   2  the rest of the stack from PyPI + two sha256-pinned wheels by URL     (requirements/2-stack.txt)
#   3  this nanotron checkout, editable, without touching dependencies
#   4  nanotron's C++ dataset helper, compiled now with this env's interpreter (otherwise rank 0 compiles it at the
#      first training step, inside the source tree)
#   5  pip check, version checks and CPU import checks; GPU-dependent imports are reported, not required
#
# KYS3B_PIP_EXTRA (optional) is appended to every pip command, e.g. "--cache-dir /big/disk/pip-cache".
set -euo pipefail
PY_BASE="${1:?usage: install_env.sh <python3.11 with headers> <new env dir>}"
ENV="${2:?usage: install_env.sh <python3.11 with headers> <new env dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"
REQ="$HERE/requirements"
read -r -a PIP_EXTRA <<< "${KYS3B_PIP_EXTRA:-}"
say() { echo "[install3b] $(date +%H:%M:%S) $*"; }
die() { echo "[install3b] ERROR: $*" >&2; exit 1; }

[[ ! -e "$ENV" ]] || die "$ENV exists; give a new directory (nothing is installed over an existing env)"
"$PY_BASE" - <<'PY' || die "$PY_BASE is not a CPython 3.11 with headers (see the comment at the top of this script)"
import os, sys, sysconfig
assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11), \
    f'{sys.implementation.name} {sys.version.split()[0]}: need CPython 3.11 (the binary wheels are cp311)'
inc = sysconfig.get_paths()['include']
assert os.path.isfile(os.path.join(inc, 'Python.h')), f'no Python.h in {inc}'
PY
command -v "${CXX:-g++}" >/dev/null && command -v make >/dev/null || die "need a C++ compiler (\$CXX or g++) and make"

say "venv $ENV from $PY_BASE"
"$PY_BASE" -m venv "$ENV"
P="$ENV/bin/python"
"$P" -m pip install "${PIP_EXTRA[@]}" --quiet pip==26.2.1
say "step 1: torch from the PyTorch cu128 index"
"$P" -m pip install "${PIP_EXTRA[@]}" -r "$REQ/1-torch-cu128.txt"
say "step 2: pinned stack from PyPI + pinned wheels"
"$P" -m pip install "${PIP_EXTRA[@]}" -r "$REQ/2-stack.txt"
say "step 3: nanotron (editable, --no-deps)"
"$P" -m pip install "${PIP_EXTRA[@]}" --no-deps -e "$CODE"
say "step 4: C++ dataset helper"
make -C "$CODE/src/nanotron/data/nemo_dataset" PYTHON="$P"
say "step 5: checks"
"$P" -m pip check
"$P" "$HERE/check_env.py"
say "done: $ENV  (set environment.PYTHON in your operator config to $P)"
