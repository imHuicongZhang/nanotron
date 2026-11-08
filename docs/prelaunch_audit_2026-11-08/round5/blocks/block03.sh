cd "$CODE_ROOT"
(cd "$DATA_ROOT/signal-disagreement-lambda0/tokenized" && sha256sum -c SHA256SUMS)     # 78 files OK
python tools/fix_ds_metadata.py --output-folder "$DATA_ROOT/signal-disagreement-lambda0/tokenized" \
    --tokenizer-dir "$DATA_ROOT/tokenizer"
