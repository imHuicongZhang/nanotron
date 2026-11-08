# ---- wave 2, after wave 1 frees GPUs, at dp 8 with the filled-dp8 configs ----
cd "$CODE_ROOT"
# step 1: plan and inspect
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8"
# step 2: validate with SLURM, nothing queued
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8" --test-only
