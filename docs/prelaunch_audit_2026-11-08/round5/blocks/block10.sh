# step 3, wave 1: only after steps 1 and 2 succeeded for all three lines
cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1" --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1" --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware --submit
