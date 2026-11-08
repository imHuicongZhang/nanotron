cd "$EVAL_ROOT" && source .venv/bin/activate && export HF_HOME="$EVAL_ROOT/hf_cache"
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203
