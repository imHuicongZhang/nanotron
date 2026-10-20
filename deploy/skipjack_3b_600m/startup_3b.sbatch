#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
# Bounded startup validation of the 3B grid, run on quality-first trunk1 itself (steps 0 -> 30, kept as real progress).
# Submitted ONLY by deploy/skipjack_3b_600m/train8_3b.py as rw3b_gpu_3b_seed42_startup (lane 0 head, 4 x H100). The
# production quality-first trunk1 job depends on it (afterok), so a failed check stops that lane; train8_monitor stops
# the rest. Phases and pass criteria: startup_3b.py.
#
#   startup_3b.sbatch <workdir under <TRAIN_ROOT>/reports/train8/startup/>
set -euo pipefail
WORK="${1:?workdir}"
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys-3b600m
[[ "${KYS_REPO:-}" == "$REPO" ]] || { echo "[startup3b] KYS_REPO must be $REPO"; exit 1; }
source $W/envs/pretrain/bin/activate
cd "$REPO"
export C_INCLUDE_PATH=$W/basic/miniconda3/envs/agents/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
echo "[startup3b] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} head=$(git rev-parse HEAD)"
TRUNK=$(python -c "import sys; sys.path.insert(0,'deploy/skipjack_3b_600m'); import registry_3b as R; print(R.ckpt_dir('quality-first','trunk1'))")
if [[ -f "$WORK/startup_report.json" ]] && python -c "import json,sys; sys.exit(0 if json.load(open('$WORK/startup_report.json'))['ok'] else 1)"; then
  echo "[startup3b] $WORK/startup_report.json already OK; nothing to do"; exit 0
fi
[[ -e "$WORK/cfg" ]] && { echo "[startup3b] $WORK has an earlier, unfinished attempt; inspect it (and $TRUNK) before a retry"; exit 1; }
if [[ -f "$TRUNK/latest.txt" && "$(cat "$TRUNK/latest.txt")" != 0 ]]; then
  echo "[startup3b] $TRUNK/latest.txt is $(cat "$TRUNK/latest.txt"), not 0: trunk already progressed; refusing"; exit 1
fi
mkdir -p "$WORK/logs"

python - <<'PY'
import torch, flash_attn
print("[startup3b] device", torch.cuda.get_device_name(), "count", torch.cuda.device_count(), "| torch", torch.__version__, "| flash_attn", flash_attn.__version__)
assert torch.cuda.get_device_capability() == (9, 0) and torch.cuda.device_count() == 4
PY
python deploy/skipjack_3b_600m/assert_invariants_3b.py --all --corpus
PYTHONPATH=/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/tools/pytest_pkgs \
  python -m pytest -q -p no:cacheprovider -W ignore deploy/skipjack_3b_600m/test_lr_resume_3b.py
bash deploy/skipjack_3b_600m/init_3b.sh "$WORK/init"
python deploy/skipjack_3b_600m/startup_3b.py derive --workdir "$WORK"
python deploy/skipjack_3b_600m/startup_3b.py seed-trunk --config configs/3B-rewrite-600m/seed42/3b_quality_first_seed42_trunk1.yaml
python deploy/skipjack_3b_600m/startup_3b.py seed-trunk --config "$WORK/cfg/s_straight.yaml"

set -a; source configs/3B-rewrite-600m/seed42/3b_quality_first_seed42_trunk1.env; set +a
export WANDB_JOB_TYPE=startup WANDB_TAGS="${WANDB_TAGS},startup"

run() {   # run <phase>: torchrun one startup config with an nvidia-smi sampler
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
rc=0
run s_trunk && run s_resume && run s_straight || { rc=$?; echo "[startup3b] a phase failed (rc $rc); summarizing what ran"; }
rm -rf "$WORK/ckpt/straight"     # the reference's seeded step-0 copy; nothing else is saved there
python deploy/skipjack_3b_600m/startup_3b.py summarize --workdir "$WORK" || rc=1
if [[ $rc -eq 0 ]]; then   # the DONE marker train8_3b.py reconciles on (only a passing validation writes it)
  PASSED="$(dirname "$WORK")/PASSED"
  mkdir -p "$PASSED" && cp "$WORK/startup_report.json" "$PASSED/" && cp "$TRUNK/30/model_config.json" "$PASSED/model_config.json"
fi
echo "[startup3b] $(date -Is) done rc=$rc; report $WORK/startup_report.json"
exit $rc
