cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42 --settings "$W1"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 43 --settings "$W1"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings raw_diversity_oriented,raw_disagreement_aware
# wave 2, λ = 0 seed 44, ONE of the two:
#   at dp 8, into a separate folder so seed 44's dp-4 configs stay as they are
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster-8gpu --seed 44 \
    --settings signal-disagreement-lambda0 --out-dir configs/1.5B-baseline-seed44/filled-dp8
#   or at dp 4: python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings signal-disagreement-lambda0
