cd "$CODE_ROOT"
bash tools/kys_raw/tokenize_raw_text.sh "$DATA_ROOT" raw_diversity_oriented
bash tools/kys_raw/tokenize_raw_text.sh "$DATA_ROOT" raw_disagreement_aware
# 16 tasks at once need ~165 GiB RAM; on a smaller node prefix KYS_TOKENIZE_WORKERS=n (n x 10.5 GiB must fit)
