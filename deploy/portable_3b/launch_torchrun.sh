#!/usr/bin/env bash
# torchrun one prepared config with the operator's launch settings. Run it ONCE PER NODE (run_segment.sh / the rendered
# Slurm files do; with scheduler.type none and nnodes > 1, start it on every node yourself).
#
#   bash deploy/portable_3b/launch_torchrun.sh <CONFIG_ROOT>/site.env <W&B env file> <effective config.yaml> <log file>
#
# Everything comes from site.env (rendered from the operator config): Python, code, nnodes, nproc_per_node and the
# rendezvous. ${VAR} references in the rendezvous (e.g. ${MASTER_ADDR}) resolve when site.env is sourced here.
set -euo pipefail
SITE="${1:?site.env}"; WENV="${2:?W&B env file}"; CFG="${3:?config}"; LOG="${4:?log file}"
die() { echo "[kys3b-launch] ERROR: $*" >&2; exit 1; }
set -a; source "$SITE"; source "$WENV"; set +a
for v in KYS3B_PYTHON KYS3B_CODE_ROOT KYS3B_NNODES KYS3B_NPROC_PER_NODE KYS3B_RDZV_BACKEND KYS3B_RDZV_ENDPOINT KYS3B_RDZV_ID KYS3B_DP; do
  [[ -n "${!v:-}" ]] || die "$v is empty after sourcing $SITE (an unset \${VAR} in the operator launch settings?)"
done
[[ "$KYS3B_RDZV_ENDPOINT" == *:* && "$KYS3B_RDZV_ENDPOINT" != :* ]] || die "rendezvous endpoint '$KYS3B_RDZV_ENDPOINT' is not host:port"
(( KYS3B_NNODES * KYS3B_NPROC_PER_NODE == KYS3B_DP )) || die "nnodes x nproc_per_node != data-parallel size $KYS3B_DP"
"$KYS3B_PYTHON" - "$CFG" "$KYS3B_DP" <<'PY' || die "$CFG does not match this launch (data-parallel size)"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
t, p = c['tokens'], c['parallelism']
assert p['dp'] == int(sys.argv[2]) and p['tp'] == 1 and p['pp'] == 1
assert t['micro_batch_size'] * t['batch_accumulation_per_replica'] * p['dp'] == 1024 and t['sequence_length'] == 2048
PY
[[ "${WANDB_MODE:-}" == online && -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]] || die "$WENV must set WANDB_MODE=online, WANDB_ENTITY, WANDB_PROJECT"
mkdir -p "$(dirname "$LOG")" "$WANDB_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
cd "$KYS3B_CODE_ROOT"
echo "[kys3b-launch] $(date -Is) host=$(hostname) nnodes=$KYS3B_NNODES nproc=$KYS3B_NPROC_PER_NODE rdzv=$KYS3B_RDZV_ENDPOINT id=$KYS3B_RDZV_ID cfg=$CFG" | tee -a "$LOG"
set +e
"$KYS3B_PYTHON" -u -m torch.distributed.run --nnodes "$KYS3B_NNODES" --nproc_per_node "$KYS3B_NPROC_PER_NODE" \
    --rdzv_backend "$KYS3B_RDZV_BACKEND" --rdzv_endpoint "$KYS3B_RDZV_ENDPOINT" --rdzv_id "$KYS3B_RDZV_ID" --max_restarts 0 \
    run_train.py --config-file "$CFG" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
exit "$rc"
