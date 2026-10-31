#!/usr/bin/env bash
# Run one segment of the grid end to end: prepare -> torchrun on every node -> finish. Requeue-safe: a rerun of a
# finished segment exits 0 without training; trunk1 past step 0 continues with optimizer + LR state.
#
#   bash deploy/portable_3b/run_segment.sh <CONFIG_ROOT>/site.env <segment run name, e.g. 3b_quality_base_seed42_trunk1>
#
# One node:              runs torchrun here.
# Several nodes, Slurm:  run it inside the allocation (the rendered slurm/<run>.sbatch does); torchrun is started on
#                        every node with srun.
# Several nodes, other:  do the three steps yourself: `kys3b.py prepare` once, launch_torchrun.sh on every node,
#                        `kys3b.py finish` once (README.md, "Multi-node without Slurm").
set -euo pipefail
SITE="${1:?site.env}"; RUN="${2:?segment run name}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "[kys3b-run] ERROR: $*" >&2; exit 1; }
set -a; source "$SITE"; set +a
K="$KYS3B_PYTHON $HERE/kys3b.py"
TAG="${SLURM_JOB_ID:-local}.$(date +%Y%m%d-%H%M%S)"
LOG="$KYS3B_LOG_ROOT/$RUN.$TAG.log"
mkdir -p "$KYS3B_LOG_ROOT"

set +e
EFF="$($K prepare --operator "$KYS3B_OPERATOR" --run "$RUN")"; rc=$?
set -e
if [[ $rc -eq 3 ]]; then echo "[kys3b-run] $EFF"; exit 0; fi
[[ $rc -eq 0 ]] || die "prepare failed for $RUN (rc $rc)"
WENV="$KYS3B_CONFIG_ROOT/grid/$RUN.env"

if [[ "$KYS3B_NNODES" -eq 1 ]]; then
  bash "$HERE/launch_torchrun.sh" "$SITE" "$WENV" "$EFF" "$LOG"
elif [[ "$KYS3B_SCHEDULER" == slurm && -n "${SLURM_JOB_ID:-}" ]]; then
  srun --nodes="$KYS3B_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
      bash "$HERE/launch_torchrun.sh" "$SITE" "$WENV" "$EFF" "$LOG"
else
  die "nnodes=$KYS3B_NNODES outside a Slurm allocation: start launch_torchrun.sh on every node (README.md)"
fi
$K finish --operator "$KYS3B_OPERATOR" --run "$RUN"
