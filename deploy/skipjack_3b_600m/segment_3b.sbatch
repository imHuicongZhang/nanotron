#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
# One 3B rewrite-600m segment (trunk1-6 or ep1-3) on one 4 x H100 node. Same requeue-safe structure as
# deploy/slurm/kys_segment.sbatch (unchanged, still used by the 1.5B grid); only the invariant check and the
# repository differ. Partition, account, QoS, gres, time, name, output and dependencies come from
# deploy/skipjack_3b_600m/capped_submit_3b.py on the sbatch command line.
#
#   sbatch [...] deploy/skipjack_3b_600m/segment_3b.sbatch <rendered 3B config.yaml>
#
#   * trunk segments share one checkpoints_path and resume through latest.txt (<= 1000 steps lost);
#   * branches resume from the trunk STEP directory, so a restart re-runs the branch from its branch point;
#   * a segment whose final checkpoint exists exits 0 before torchrun (nanotron asserts train_steps > last step).
set -euo pipefail
CFG="${1:?rendered config}"
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO="${KYS_REPO:?KYS_REPO must name the 3B worktree}"
[[ "$REPO" == "$W/projects/nanotron-kys-3b600m" ]] || { echo "[3b] refusing to run from $REPO (not the 3B worktree)"; exit 1; }
source "${KYS_ENV_ACTIVATE:-$W/envs/pretrain/bin/activate}"

read -r CKPT STEPS RUN < <(python - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["checkpoints"]["checkpoints_path"], c["tokens"]["train_steps"], c["general"]["run"])
PY
)
echo "[3b] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} run=$RUN target=$STEPS head=$(git -C "$REPO" rev-parse HEAD)"
if [[ -f "$CKPT/$STEPS/model_config.json" ]]; then
  echo "[3b] $CKPT/$STEPS already written — segment complete, nothing to do"
  exit 0
fi

set -a; source "${CFG%.yaml}.env"; set +a
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
if [[ -n "${KYS_PYTHON_INCLUDE:-}" ]]; then
  export C_INCLUDE_PATH=$KYS_PYTHON_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
fi
cd "$REPO"
python deploy/skipjack_3b_600m/assert_invariants_3b.py --config "$CFG" --corpus --check-resume

python -u -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
    run_train.py --config-file "$CFG"

[[ -f "$CKPT/$STEPS/model_config.json" ]] || { echo "[3b] finished without $CKPT/$STEPS — failing so dependents do not start"; exit 1; }
echo "[3b] $(date -Is) segment complete: $CKPT/$STEPS"
