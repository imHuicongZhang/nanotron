#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
# SKIPJACK-ONLY (Skipjack worktree, partition, paths). Other sites: deploy/portable_3b/run_segment.sh / README.md.
# One 3B rewrite-600m segment (trunk1-6 or ep1-3) on one 4 x H100 node. Same requeue-safe structure as
# deploy/slurm/kys_segment.sbatch (unchanged, still used by the 1.5B grid); only the invariant check and the
# repository differ. Partition, account, QoS, gres, time, name, output and dependencies come from
# deploy/skipjack_3b_600m/train8_3b.py (dedicated 8-GPU budget, jobs rw3b_gpu_*) on the sbatch command line.
#   * trunk1 with no latest.txt: creates the shared init if absent (init_3b.sh, locked) and seeds trunk/0 from it;
#   * trunk1 past step 0 (requeue, or after the startup validation's steps 0 -> 30) runs a copy of its config with
#     load_optimizer / load_lr_scheduler true: the rendered trunk1 is weights-only, which is right only from the init.
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
RUNCFG="$CFG"
TRAIN8=/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/reports/train8
if [[ "$RUN" == *_trunk1 ]]; then
  if [[ ! -f "$CKPT/latest.txt" ]]; then
    bash deploy/skipjack_3b_600m/init_3b.sh "$TRAIN8/init_scratch/${SLURM_JOB_ID:-nojob}"
    python deploy/skipjack_3b_600m/startup_3b.py seed-trunk --config "$CFG"
  fi
  if [[ "$(cat "$CKPT/latest.txt")" != 0 ]]; then
    mkdir -p "$TRAIN8/derived"
    RUNCFG="$TRAIN8/derived/$RUN.${SLURM_JOB_ID:-nojob}.r${SLURM_RESTART_COUNT:-0}.yaml"
    python deploy/skipjack_3b_600m/startup_3b.py resume-config --config "$CFG" --out "$RUNCFG"
    echo "[3b] trunk1 resumes $CKPT/$(cat "$CKPT/latest.txt") with optimizer + LR state: $RUNCFG"
  fi
fi
python deploy/skipjack_3b_600m/assert_invariants_3b.py --config "$CFG" --corpus --check-resume

mkdir -p "$TRAIN8/nvsmi"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 60 \
    > "$TRAIN8/nvsmi/$RUN.${SLURM_JOB_ID:-nojob}.r${SLURM_RESTART_COUNT:-0}.csv" &
NV=$!
rc=0
python -u -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
    run_train.py --config-file "$RUNCFG" || rc=$?
kill $NV 2>/dev/null || true
[[ $rc -eq 0 ]] || { echo "[3b] torchrun exited $rc"; exit $rc; }

[[ -f "$CKPT/$STEPS/model_config.json" ]] || { echo "[3b] finished without $CKPT/$STEPS — failing so dependents do not start"; exit 1; }
echo "[3b] $(date -Is) segment complete: $CKPT/$STEPS"
