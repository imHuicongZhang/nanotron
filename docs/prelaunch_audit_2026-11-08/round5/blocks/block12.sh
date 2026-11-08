# step 3, wave 2: only after steps 1 and 2 succeeded
cd "$CODE_ROOT"
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8" --submit
