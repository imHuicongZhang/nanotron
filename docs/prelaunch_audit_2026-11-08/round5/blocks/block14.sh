cd "$CODE_ROOT"
for S in 42 43 44; do
  for setting in raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0; do
    for N in 1 2 3; do
      step=$(case $N in 1) echo 4768;; 2) echo 9537;; 3) echo 14305;; esac)
      final="$CKPT_ROOT/seed$S/$setting/ep$N/$setting/seed$S/$step"
      hfdir="$EXPORT_ROOT/rewrite-1p5b/seed$S/$setting/ep$N/hf"
      [ -f "$final/model_config.json" ] || { echo "missing final $final"; continue; }
      python tools/kys_eval/convert_to_hf.py --checkpoint_path "$final" --save_path "$hfdir" \
          --tokenizer "$DATA_ROOT/tokenizer" --dtype bfloat16
      hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "$hfdir" "rewrite-1p5b/seed$S/$setting/ep$N/hf"
      hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "$final" "rewrite-1p5b/seed$S/$setting/ep$N/nanotron"
    done
  done
done
# Hub revision after the uploads: record it, it is the HF_REVISION of §10
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('blab-jhu/KYS-1.5B-Raw-Selected-Baselines').sha)"
