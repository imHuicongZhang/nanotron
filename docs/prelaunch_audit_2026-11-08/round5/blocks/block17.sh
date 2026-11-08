cd "$EVAL_ROOT" && source .venv/bin/activate
for S in 42 43 44; do
  hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "results/raw_selected/seed$S" \
      "rewrite-1p5b/seed$S/eval/results/raw_selected/seed$S"
  hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines reports "rewrite-1p5b/seed$S/eval/reports"
done
