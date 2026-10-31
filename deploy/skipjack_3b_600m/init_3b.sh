#!/usr/bin/env bash
# SKIPJACK-ONLY (hard-coded Skipjack worktree; other sites: deploy/portable_3b/create_init.sh).
# Create the shared 3B init (<INIT_ROOT>/_init_3B_seed42/{0,init_manifest.json}) exactly once, on a 4 x H100 node inside an
# rw3b_gpu_* job (startup_3b.sbatch or segment_3b.sbatch for trunk1). Serialized by flock, so whichever job starts first
# creates it and the other reuses it; an existing init is never overwritten (a half-written one stops the job).
#
#   init_3b.sh <scratch dir for the 1-step init run>
set -euo pipefail
SCRATCH="${1:?scratch dir}"
REPO=/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-3b600m
cd "$REPO"
read -r INIT_ROOT INIT_PARENT < <(python -c "import sys; sys.path.insert(0,'deploy/skipjack_3b_600m'); import registry_3b as R; print(R.INIT_ROOT, R.INIT_ROOT / f'_init_3B_seed{R.SEED}')")
mkdir -p "$INIT_ROOT"
exec 9>"$INIT_ROOT/.init.lock"
flock 9
if [[ -f "$INIT_PARENT/init_manifest.json" ]]; then
  echo "[init3b] using existing init $INIT_PARENT (manifest present)"
  exit 0
fi
[[ -e "$INIT_PARENT" ]] && { echo "[init3b] $INIT_PARENT exists without init_manifest.json; inspect it, never overwrite"; exit 1; }
mkdir -p "$SCRATCH/logs"
python deploy/skipjack_3b_600m/startup_3b.py derive-init --config configs/3B-rewrite-600m/seed42/3b_quality_first_seed42_trunk1.yaml \
    --out "$SCRATCH/init.yaml" --ckpt "$SCRATCH/ckpt"
set -a; source configs/3B-rewrite-600m/seed42/3b_quality_first_seed42_trunk1.env; set +a
export WANDB_RUN_GROUP=init WANDB_JOB_TYPE=init WANDB_MODE=offline
python -u -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
    run_train.py --config-file "$SCRATCH/init.yaml" 2>&1 | tee "$SCRATCH/logs/init.log"
[[ ${PIPESTATUS[0]} -eq 0 && -f "$SCRATCH/ckpt/0/model_config.json" ]] || { echo "[init3b] init run wrote no step-0 checkpoint"; exit 1; }
mkdir -p "$INIT_PARENT.tmp" && cp -a "$SCRATCH/ckpt/0" "$INIT_PARENT.tmp/0"
python - "$INIT_PARENT.tmp" "${SLURM_JOB_ID:-}" <<'PY'
import json, sys, time
from pathlib import Path
sys.path.insert(0, 'deploy/skipjack_3b_600m'); import capped_submit_3b as C
d = Path(sys.argv[1]); m = C.init_digest(d / '0')
m.update(created=time.strftime('%Y-%m-%dT%H:%M:%S'), job=sys.argv[2], seed=42, source='init_3b.sh (1 step, save_initial_state)')
(d / 'init_manifest.json').write_text(json.dumps(m, indent=1)); print('[init3b] init', m)
PY
mv "$INIT_PARENT.tmp" "$INIT_PARENT"
rm -rf "$SCRATCH/ckpt"
echo "[init3b] $(date -Is) created $INIT_PARENT"
