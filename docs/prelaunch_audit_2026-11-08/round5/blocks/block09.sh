# ---- wave 1 ----
cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
# step 1: plan and inspect
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1"
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1"
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware
# step 2: validate with SLURM, nothing queued
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1" --test-only
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1" --test-only
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware --test-only
