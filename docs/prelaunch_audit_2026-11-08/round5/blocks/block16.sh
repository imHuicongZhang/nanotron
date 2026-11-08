cd "$EVAL_ROOT" && source .venv/bin/activate && export HF_HOME="$EVAL_ROOT/hf_cache"
HF_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('blab-jhu/KYS-1.5B-Raw-Selected-Baselines').sha)")
echo "scoring at $HF_REVISION"                  # must equal the revision recorded in §9; record it
R="--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0 --hf-revision $HF_REVISION"
python -m kys_eval.run_grid $R --list --check-hub      # "27 cells; SLURM: --array=0-26", all "present"
mkdir -p logs && sbatch --array=0-26%8 --export=ALL,KYS_EVAL_ROOT="$EVAL_ROOT",KYS_GRID_ARGS="$R" slurm/eval_grid_array.sbatch
python -m kys_eval.run_grid $R --list                  # all 27 "done"
python -m kys_eval.aggregate                           # reports/raw_selected_report.md, raw_vs_rewritten.md, CSVs
