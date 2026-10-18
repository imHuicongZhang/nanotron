#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
# Bounded 3B smoke test on one 4 x H100 node (~2 h, limit 4 h). Submit ONLY through
#   python deploy/skipjack_3b_600m/capped_submit_3b.py smoke
# which checks the baseline handoff, the owner's LAUNCH_APPROVED and the 20-GPU cap, and names the job kys_3b_smoke.
#
#   smoke_3b.sbatch <fresh workdir under <TRAIN_ROOT>/reports/smoke/>
#
# Phases and pass criteria: deploy/skipjack_3b_600m/smoke_3b.py. Creates the shared init only if it does not exist
# (never overwrites it). W&B group "smoke" in the 3B project; no production checkpoint folder is written.
set -euo pipefail
WORK="${1:?fresh workdir}"
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys-3b600m
[[ "${KYS_REPO:-}" == "$REPO" ]] || { echo "[smoke3b] KYS_REPO must be $REPO"; exit 1; }
source $W/envs/pretrain/bin/activate
cd "$REPO"
export C_INCLUDE_PATH=$W/basic/miniconda3/envs/agents/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
[[ -e "$WORK" ]] && { echo "[smoke3b] $WORK exists; use a fresh workdir"; exit 1; }
mkdir -p "$WORK/logs"
echo "[smoke3b] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} head=$(git rev-parse HEAD)"

python - <<'PY'
import torch, flash_attn
print("[smoke3b] device", torch.cuda.get_device_name(), "count", torch.cuda.device_count(), "| torch", torch.__version__, "| flash_attn", flash_attn.__version__)
assert torch.cuda.get_device_capability() == (9, 0) and torch.cuda.device_count() == 4
PY
python deploy/skipjack_3b_600m/assert_invariants_3b.py --all --corpus
python deploy/skipjack_3b_600m/smoke_3b.py derive --workdir "$WORK"
INIT_PARENT=$(python -c "import sys; sys.path.insert(0,'deploy/skipjack_3b_600m'); import registry_3b as R; print(R.INIT_ROOT / f'_init_3B_seed{R.SEED}')")

set -a; source configs/3B-rewrite-600m/seed42/3b_quality_first_seed42_trunk1.env; set +a
export WANDB_RUN_GROUP=smoke WANDB_JOB_TYPE=smoke WANDB_TAGS="${WANDB_TAGS},smoke"

run() {   # run <phase>: torchrun one smoke config with an nvidia-smi sampler
  local ph=$1
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -lms 2000 > "$WORK/logs/$ph.nvsmi.csv" &
  local nv=$!
  python -u -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --rdzv_backend c10d \
      --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
      run_train.py --config-file "$WORK/cfg/$ph.yaml" 2>&1 | tee "$WORK/logs/$ph.log"
  local rc=${PIPESTATUS[0]}
  kill $nv 2>/dev/null || true
  return $rc
}
seed() {  # seed <trunk folder>: init -> <trunk>/0 + latest.txt
  mkdir -p "$WORK/ckpt/$1" && cp -a "$INIT_PARENT/0" "$WORK/ckpt/$1/0" && echo 0 > "$WORK/ckpt/$1/latest.txt"
}

if [[ ! -f "$INIT_PARENT/init_manifest.json" ]]; then
  [[ -e "$INIT_PARENT" ]] && { echo "[smoke3b] $INIT_PARENT exists without init_manifest.json; inspect it, never overwrite"; exit 1; }
  run init
  [[ -f "$WORK/ckpt/init/0/model_config.json" ]] || { echo "[smoke3b] init run wrote no step-0 checkpoint"; exit 1; }
  mkdir -p "$INIT_PARENT" && cp -a "$WORK/ckpt/init/0" "$INIT_PARENT/0"
  python - "$INIT_PARENT" "${SLURM_JOB_ID:-}" <<'PY'
import json, sys, time
from pathlib import Path
sys.path.insert(0, 'deploy/skipjack_3b_600m'); import capped_submit_3b as C
d = Path(sys.argv[1]); m = C.init_digest(d / '0')
m.update(created=time.strftime('%Y-%m-%dT%H:%M:%S'), smoke_job=sys.argv[2], seed=42, source='smoke phase init (save_initial_state)')
(d / 'init_manifest.json').write_text(json.dumps(m, indent=1)); print('[smoke3b] init', m)
PY
else
  echo "[smoke3b] using existing init $INIT_PARENT (manifest present)"
fi
for t in a straight div da b c; do seed $t; done
run a_trunk && run a_resume && run a_branch && run a_straight && run a_div && run a_da && run b_trunk || echo "[smoke3b] a phase failed; summarizing what ran"
run c_trunk || echo "[smoke3b] optional layout C (mbs 32) failed (likely OOM); recorded, not a smoke failure"
python deploy/skipjack_3b_600m/smoke_3b.py summarize --workdir "$WORK"
echo "[smoke3b] $(date -Is) done; review $WORK/smoke_report.json, then the owner writes SMOKE_ACCEPTED"
