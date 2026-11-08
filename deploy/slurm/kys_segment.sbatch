#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
# One KYS segment (trunk1-3 or ep1-3) on one node. Partition, account, QoS, gres, cpus, time,
# job name, output path and dependencies are passed on the sbatch command line by
# tools/kys_raw/plan_submit.py, which reads them from deploy/clusters.yaml.
#
#   sbatch [...] deploy/slurm/kys_segment.sbatch <rendered config.yaml>
#
# Requeue safety. The partition runs PreemptMode=REQUEUE with GraceTime 0, and a failed node
# requeues too, so this script can start any number of times for the same segment:
#   * trunk segments share one checkpoints_path and resume through latest.txt, so a restart
#     continues from the newest checkpoint (checkpoint_interval 1500 is the most work lost).
#     trunk1 is rendered weights-only (load_optimizer / load_lr_scheduler false) because step 0
#     is the init, which has no optimizer state. If its latest.txt is already > 0 (a restart after
#     its step-1500 or 3000 save), tools/kys_raw/trunk_restart_config.py writes a copy that loads
#     optimizer, LR scheduler and metadata, and that copy is what runs. Without it nanotron would
#     restart the schedule and the data stream from step 0 on trained weights and still exit 0
#     (prelaunch audit 2026-11-07, B1); assert_invariants.py --check-resume refuses that case;
#   * branches resume from the trunk STEP directory, so a restart re-runs the branch from its
#     branch point (at most ep3's 1430 steps lost);
#   * a segment that already reached train_steps exits 0 before torchrun — nanotron asserts
#     train_steps > last_train_step on resume, so without this guard a requeue after the final
#     save would fail and break the dependency chain.
set -euo pipefail
CFG="${1:?rendered config}"
# Site paths come from the cluster entry via plan_submit.py (--export); the defaults are skipjack's.
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO="${KYS_REPO:-$W/projects/nanotron-kys}"
source "${KYS_ENV_ACTIVATE:-$W/envs/pretrain/bin/activate}"

read -r CKPT STEPS RUN < <(python - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["checkpoints"]["checkpoints_path"], c["tokens"]["train_steps"], c["general"]["run"])
PY
)
echo "[kys] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} run=$RUN target=$STEPS"
if [[ -f "$CKPT/$STEPS/model_config.json" ]]; then
  echo "[kys] $CKPT/$STEPS already written — segment complete, nothing to do"
  exit 0
fi

set -a; source "${CFG%.yaml}.env"; set +a
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
# Nodes without Python dev headers: Triton JIT-builds its CUDA driver stub with gcc and needs Python.h
# (INSTALL.md, "Python headers on compute nodes"). KYS_PYTHON_INCLUDE = the cluster entry's python_include.
if [[ -n "${KYS_PYTHON_INCLUDE:-}" ]]; then
  export C_INCLUDE_PATH=$KYS_PYTHON_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
elif [[ -z "${KYS_REPO:-}" && -d $W/basic/miniconda3/envs/agents/include/python3.11 ]]; then
  export C_INCLUDE_PATH=$W/basic/miniconda3/envs/agents/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
fi
cd "$REPO"
# The rendered config, or (trunk restart with latest.txt > 0) its state-loading copy.
RUNCFG=$(python tools/kys_raw/trunk_restart_config.py "$CFG")
echo "[kys] running config: $RUNCFG"
python tools/assert_invariants.py --config "$RUNCFG" --check-resume

NPROC=$(python -c "import yaml,sys; p=yaml.safe_load(open(sys.argv[1]))['parallelism']; print(p['dp']*p['tp']*p['pp'])" "$RUNCFG")
python -u -m torch.distributed.run --nproc_per_node "$NPROC" --nnodes 1 --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
    run_train.py --config-file "$RUNCFG"

[[ -f "$CKPT/$STEPS/model_config.json" ]] || { echo "[kys] finished without $CKPT/$STEPS — failing so dependents do not start"; exit 1; }
echo "[kys] $(date -Is) segment complete: $CKPT/$STEPS"
