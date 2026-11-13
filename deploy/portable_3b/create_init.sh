#!/usr/bin/env bash
# Create the shared seed-42 initialization ONCE (needs the GPUs of one segment): the 1-step init run of
# <CONFIG_ROOT>/init/init.yaml (derived from the rendered quality-first trunk1 config, so it uses your tokenizer, data,
# deployment and W&B project), then `kys3b.py finalize-init` hashes step 0 and moves it to
# <INIT_ROOT>/_init_3B_seed42/{0,init_manifest.json}. All four settings' trunk1 start from that one directory.
#
#   bash deploy/portable_3b/create_init.sh <CONFIG_ROOT>/site.env
#
# An existing finished init is reused, never overwritten; a half-finished one stops with instructions. Run it from
# one place at a time (the lock below is advisory and only holds on filesystems with working flock).
set -euo pipefail
SITE="${1:?site.env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "[kys3b-init] ERROR: $*" >&2; exit 1; }
set -a; source "$SITE"; set +a
K="$KYS3B_PYTHON $HERE/kys3b.py"
PARENT="$KYS3B_INIT_ROOT/_init_3B_seed42"
SCRATCH="$KYS3B_INIT_ROOT/_init_3B_seed42.run"
if [[ "${KYS3B_INIT_MODE:-create}" == reuse ]]; then   # raw baselines: never create an init, only verify the pinned one
  $K verify-init --operator "$KYS3B_OPERATOR" || die "raw-baseline settings reuse the existing shared init (weights_sha256 ${KYS3B_INIT_WEIGHTS_SHA256:-?}); point INIT_ROOT at it. A new init is never created for them."
  exit 0
fi
mkdir -p "$KYS3B_INIT_ROOT" "$KYS3B_LOG_ROOT"
exec 9>"$KYS3B_INIT_ROOT/.init.lock"
flock -n 9 || die "another create_init.sh holds $KYS3B_INIT_ROOT/.init.lock"
if [[ -f "$PARENT/init_manifest.json" ]]; then
  echo "[kys3b-init] using the existing init $PARENT"; exit 0
fi
[[ ! -e "$PARENT" ]] || die "$PARENT exists without init_manifest.json; inspect it (never overwritten)"
[[ ! -e "$SCRATCH/0" ]] || die "$SCRATCH/0 exists from an earlier init run; if that run finished, run: $K finalize-init --operator $KYS3B_OPERATOR ; otherwise remove $SCRATCH"
$K check --operator "$KYS3B_OPERATOR" --corpus || die "check failed; fix it before creating the init"
$K wandb-check || die "no W&B login for the operator entity on this host"
LOG="$KYS3B_LOG_ROOT/init.${SLURM_JOB_ID:-local}.$(date +%Y%m%d-%H%M%S).log"
if [[ "$KYS3B_NNODES" -eq 1 ]]; then
  bash "$HERE/launch_torchrun.sh" "$SITE" "$KYS3B_CONFIG_ROOT/init/init.env" "$KYS3B_CONFIG_ROOT/init/init.yaml" "$LOG"
elif [[ "$KYS3B_SCHEDULER" == slurm && -n "${SLURM_JOB_ID:-}" ]]; then
  srun --nodes="$KYS3B_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
      bash "$HERE/launch_torchrun.sh" "$SITE" "$KYS3B_CONFIG_ROOT/init/init.env" "$KYS3B_CONFIG_ROOT/init/init.yaml" "$LOG"
else
  die "nnodes=$KYS3B_NNODES outside a Slurm allocation: start launch_torchrun.sh with init/init.env + init/init.yaml on every node, then run finalize-init"
fi
[[ -f "$SCRATCH/0/model_config.json" ]] || die "the init run wrote no $SCRATCH/0"
$K finalize-init --operator "$KYS3B_OPERATOR"
rm -rf "$SCRATCH"
echo "[kys3b-init] $(date -Is) created $PARENT"
