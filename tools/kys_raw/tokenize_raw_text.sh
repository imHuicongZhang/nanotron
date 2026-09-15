#!/usr/bin/env bash
# Tokenize one raw-selected corpus from the Hub's raw_text/ layout into nanotron's format.
#
#   tools/kys_raw/tokenize_raw_text.sh <data_root> <setting> <tokenizer_dir>
#
#   <data_root>       local_dir of snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset");
#                     holds manifest.json and raw_text/<setting>/part-000NN.parquet (16 files)
#   <setting>         raw_diversity_oriented | raw_disagreement_aware | raw_random | raw_rewire_inspired
#   <tokenizer_dir>   the tokenizer DIRECTORY (tokenizer/ of wytro/Know-Your-Sources-tokenized)
#   -> <data_root>/<setting>/tokenized/000NN_unshuffled.ds{,.index,.metadata}   (16 shards)
#
# The parquet files are already the final corpus: anchor merged in, shuffled at the document level
# with seed 42, rows in order. Do NOT merge or shuffle anything. With 16 files and 16 tasks,
# datatrove gives task i exactly file i (files[rank::16]) and shuffle_documents is off, so shards
# 00000..00015 concatenate to the file order. Same recipe as the published arms: datatrove 0.5.0
# DocumentTokenizer, llama-2 tokenizer, one </s> appended per document, no BOS.
#
# Steps: check the 16 files against manifest.json sha256 -> tokenize -> tools/fix_ds_metadata.py ->
# assert 16 shards and the exact token total (manifest settings.<setting>.expected_total_tokens).
set -euo pipefail
ROOT="${1:?data_root}"; SETTING="${2:?setting}"; TOK="${3:?tokenizer dir}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-python}"
IN="$ROOT/raw_text/$SETTING"
OUT="$ROOT/$SETTING/tokenized"
LOG="$ROOT/$SETTING/tokenize_logs"
[[ -f "$TOK/tokenizer.json" ]] || { echo "no tokenizer.json in $TOK"; exit 1; }

"$PY" - "$ROOT/manifest.json" "$SETTING" "$IN" <<'PY'
import hashlib, json, sys
from pathlib import Path
man, setting, d = json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3])
files = man["settings"][setting]["files"]
have = sorted(p.name for p in d.glob("part-*.parquet"))
if have != sorted(files) or len(have) != 16:
    sys.exit(f"{d}: expected the 16 files listed in manifest.json, found {len(have)}")
for name, rec in sorted(files.items()):
    h = hashlib.sha256()
    with open(d / name, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    if h.hexdigest() != rec["sha256"]:
        sys.exit(f"{d / name}: sha256 mismatch with manifest.json")
print(f"[tok] {setting}: 16 parquet files match manifest.json sha256")
PY

if [[ -e "$OUT" ]] && compgen -G "$OUT/*.ds" >/dev/null; then
  echo "$OUT already holds .ds files; refusing to overwrite (remove it to re-tokenize)"; exit 1
fi
mkdir -p "$OUT" "$LOG"
cd "$REPO"
"$PY" tools/preprocess_data_parquet.py \
    --tokenizer-name-or-path "$TOK/tokenizer.json" \
    --eos-token "</s>" \
    --output-folder "$OUT" \
    --logging-dir "$LOG" \
    --n-tasks 16 \
    parquet --dataset "$IN" --column text --glob-pattern "part-*.parquet" &
TP=$!
while kill -0 $TP 2>/dev/null; do
  sleep 60
  newest=$(find "$OUT" "$LOG" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  if [[ -n "$newest" ]] && (( $(date +%s) - ${newest%.*} > 1800 )); then
    echo "[tok] no output for 30 min — a datatrove worker stalled (data_preprocessing_guide.md, Caveat 5); killing"
    kill $TP; exit 1
  fi
done
wait $TP

"$PY" tools/fix_ds_metadata.py --output-folder "$OUT" --tokenizer-dir "$TOK"

"$PY" - "$OUT" "$ROOT/manifest.json" "$SETTING" <<'PY'
import glob, json, sys
out, man, setting = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
ds, meta = sorted(glob.glob(f"{out}/*.ds")), sorted(glob.glob(f"{out}/*.ds.metadata"))
total = sum(int(open(m).read().splitlines()[1]) for m in meta)
want = man["settings"][setting]["expected_total_tokens"]
print(f"[tok] {setting}: {len(ds)} shards, {total:,} tokens (manifest expected_total_tokens {want:,})")
if len(ds) != 16 or len(meta) != 16:
    sys.exit(f"expected 16 .ds and 16 .ds.metadata, got {len(ds)} / {len(meta)}")
if total != want:
    sys.exit(f"token total {total:,} != manifest {want:,} ({total - want:+,}); do not train on this folder")
print(f"[tok] OK — {out} is ready; tools/assert_invariants.py checks the same total from manifest.json")
PY
