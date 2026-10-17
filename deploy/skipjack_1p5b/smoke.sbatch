#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
# 20-step real-data smoke test on one 4 x H100 node (INSTALL.md §3.1-3.4), inside the project GPU budget.
#
#   python deploy/skipjack_1p5b/capped_submit.py guard --need-gpus 4 && \
#   sbatch --job-name=kys_smoke20_seed42 -p h100 -A bvandur1_blabscify -q jhu2 --gres=gpu:h100:4 \
#       --cpus-per-task=32 --time=01:30:00 --output=<data root>/logs/slurm/%x.%j.log \
#       deploy/skipjack_1p5b/smoke.sbatch <filled trunk1 config> <fresh workdir>
#
# Same environment, C_INCLUDE_PATH and .env as deploy/slurm/kys_segment.sbatch; the run is grouped as
# "smoke" in W&B so it never joins a production chain. Writes no checkpoint (smoke_20steps.py), and the
# workdir must be fresh. Samples nvidia-smi every 2 s for peak memory.
set -euo pipefail
CFG="${1:?filled trunk1 config}"; WORK="${2:?fresh workdir}"
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys
INIT_ROOT=/weka/projects/bvandur1/zhuicon1/checkpoints/1p5_baseline/init
source $W/envs/pretrain/bin/activate
cd "$REPO"
[[ -e "$WORK" ]] && { echo "[smoke] $WORK exists; use a fresh workdir"; exit 1; }
mkdir -p "$WORK"
echo "[smoke] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} head=$(git rev-parse HEAD) cfg=$CFG"

set -a; source "${CFG%.yaml}.env"; set +a
export WANDB_RUN_GROUP=smoke WANDB_JOB_TYPE=smoke WANDB_TAGS="${WANDB_TAGS},smoke"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
export C_INCLUDE_PATH=$W/basic/miniconda3/envs/agents/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}

# INSTALL.md §3.1
python - <<'PY'
import torch, flash_attn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
print("[smoke] device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "count", torch.cuda.device_count())
print("[smoke] torch", torch.__version__, "| flash_attn", flash_attn.__version__, "| sm_90 in arch_list", "sm_90" in torch.cuda.get_arch_list())
q=k=v=torch.randn(256,16,128,device="cuda",dtype=torch.bfloat16)
cu=torch.tensor([0,256],dtype=torch.int32,device="cuda")
o=flash_attn_varlen_func(q=q,k=k,v=v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=256,max_seqlen_k=256,dropout_p=0.0,softmax_scale=None,causal=True)
torch.cuda.synchronize(); print("[smoke] flash-attn OK", tuple(o.shape))
assert torch.cuda.get_device_capability() == (9, 0) and torch.cuda.device_count() == 4
PY
# full invariant check of the production config, including corpus totals and environment
python tools/assert_invariants.py --config "$CFG"

SEED=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['general']['seed'])" "$CFG")
python tools/kys_raw/smoke_20steps.py --config "$CFG" --init "$INIT_ROOT/_init_1.5B_seed$SEED/0" --workdir "$WORK"
SCFG=$(ls "$WORK"/*_smoke20.yaml)

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -lms 2000 > "$WORK/nvsmi.csv" &
NV=$!
python -u -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$((29500 + ${SLURM_JOB_ID:-0} % 1000))" --max_restarts 0 \
    run_train.py --config-file "$SCFG"
kill $NV || true
python - "$WORK/nvsmi.csv" <<'PY'
import sys
rows = [l.split(',') for l in open(sys.argv[1]) if l.strip()]
peak = max(int(r[1]) for r in rows)
print(f"[smoke] nvidia-smi peak memory.used {peak} MiB = {peak / 1024:.1f} GiB of {int(rows[0][2]) / 1024:.1f} GiB")
PY
echo "[smoke] $(date -Is) done"
