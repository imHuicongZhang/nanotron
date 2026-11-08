cd "$CODE_ROOT"
python tools/kys_raw/smoke_20steps.py \
    --config configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_trunk1.yaml \
    --init "$INIT_ROOT/_init_1.5B_seed42/0" --workdir "$SMOKE_DIR"
# then run the torchrun command it prints, on one node with dp GPUs
# (for dp 8: --config configs/1.5B-baseline-seed44/filled-dp8/signal-disagreement-lambda0_seed44_trunk1.yaml)
