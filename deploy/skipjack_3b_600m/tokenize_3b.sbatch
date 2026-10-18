#!/usr/bin/env bash
#SBATCH --job-name=rw3b_tokenize
#SBATCH --partition=med
#SBATCH --account=bvandur1_blabscify
#SBATCH --qos=jhu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --open-mode=append
# CPU only. Tokenize one final 600M corpus (array index -> setting) into nanotron .ds shards, then verify it.
#
#   sbatch --output=<TRAIN_ROOT>/logs/slurm/%x.%A_%a.log deploy/skipjack_3b_600m/tokenize_3b.sbatch
#
# Same recipe as every KYS corpus (tools/kys_raw/tokenize_raw_text.sh): datatrove DocumentTokenizer, llama-2 tokenizer,
# one </s> per document, no BOS, no re-shuffle; n_tasks = number of shuffled parts, so task i tokenizes exactly part i.
# Reads dataset/05_final/<setting>/shuffled/ only (never writes there). Refuses a non-empty output folder.
# Steps: tokenizer copy + sha256 -> part sha256 vs validate_assembly_scope.json -> tokenize (stall watchdog) ->
# fix_ds_metadata -> assert_invariants_3b --corpus (exact corpus_train_tokens - empty-text documents; registry_3b
# CORPUS_EMPTY_DOCS) -> epoch_stats_3b.py (windows/epoch). A complete earlier run is reused (verification only).
set -euo pipefail
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys-3b600m
source $W/envs/pretrain/bin/activate
cd "$REPO"
SETTINGS=(quality-first diversity-oriented disagreement-aware)
S=${SETTINGS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}
eval "$(python - "$S" <<'PY'
import sys; sys.path.insert(0, 'deploy/skipjack_3b_600m'); import registry_3b as R
s = sys.argv[1]
print(f'IN={R.FINAL / s / "shuffled"} OUT={R.TOKENIZED / s} TOK={R.TOKENIZER} TOKSRC={R.TOKENIZER_SOURCE} '
      f'NPARTS={R.CORPUS_PARTS[s]} REPORTS={R.REPORTS} CFG={R.REPO}/configs/3B-rewrite-600m/seed{R.SEED}/{R.seg_name(s, "trunk1")}.yaml')
PY
)"
LOG=$REPORTS/tokenize/$S
echo "[tok3b] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} setting=$S head=$(git rev-parse HEAD)"

# 1. tokenizer: one copy of the exact 1.5B tokenizer directory under TRAIN_ROOT, every file pinned by sha256
mkdir -p "$(dirname "$TOK")"
if [[ ! -f "$TOK/tokenizer.json" ]]; then
  ( flock 9; [[ -f "$TOK/tokenizer.json" ]] || { cp -a "$TOKSRC" "$TOK.tmp.$$" && mv "$TOK.tmp.$$" "$TOK"; } ) 9>"$(dirname "$TOK")/.tokenizer.lock"
fi
python - "$TOK" <<'PY'
import hashlib, sys
from pathlib import Path
sys.path.insert(0, 'deploy/skipjack_3b_600m'); import registry_3b as R
tok = Path(sys.argv[1])
have = sorted(p.name for p in tok.iterdir() if p.is_file())
if have != sorted(R.TOKENIZER_SHA256):
    sys.exit(f'[tok3b] {tok} holds {have}, expected exactly {sorted(R.TOKENIZER_SHA256)}')
for name, want in R.TOKENIZER_SHA256.items():
    if hashlib.sha256((tok / name).read_bytes()).hexdigest() != want:
        sys.exit(f'[tok3b] {tok / name}: sha256 differs from the 1.5B tokenizer')
print(f'[tok3b] tokenizer {tok}: {len(have)} files identical to the 1.5B tokenizer (sha256)')
PY

# 2. inputs are exactly the validated parts
python - "$IN" "$S" "$NPARTS" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, s, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
want = json.load(open('/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/dataset/reports/stage04_assembly_20261016/validate_assembly_scope.json'))['results'][s]['shuffled']['part_sha256']
have = sorted(p.name for p in d.glob('part_*.parquet'))
if have != sorted(want) or len(have) != n:
    sys.exit(f'{d}: {len(have)} parts, validator recorded {len(want)} (expected {n})')
for name in have:
    h = hashlib.sha256()
    with open(d / name, 'rb') as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    if h.hexdigest() != want[name]:
        sys.exit(f'{d / name}: sha256 differs from validate_assembly_scope.json')
print(f'[tok3b] {s}: {n} parts match validate_assembly_scope.json sha256')
PY

# 3. tokenize (or reuse a complete earlier run: every datatrove task completed and one .ds per part)
if [[ -e "$OUT" ]] && compgen -G "$OUT/*.ds" >/dev/null; then
  ndone=$(ls "$LOG/datatrove/completions" 2>/dev/null | wc -l); nds=$(ls "$OUT"/*.ds | wc -l)
  if [[ "$ndone" == "$NPARTS" && "$nds" == "$NPARTS" ]]; then
    echo "[tok3b] reusing complete tokenization in $OUT ($ndone/$NPARTS tasks completed, $nds .ds); verifying only"
    REUSE=1
  else
    echo "[tok3b] $OUT holds $nds .ds with $ndone/$NPARTS completed tasks; refusing (move it aside after inspection)"; exit 1
  fi
fi
mkdir -p "$OUT" "$LOG"
if [[ -z "${REUSE:-}" ]]; then
setsid python tools/preprocess_data_parquet.py \
    --tokenizer-name-or-path "$TOK/tokenizer.json" --eos-token "</s>" \
    --output-folder "$OUT" --logging-dir "$LOG/datatrove" --n-tasks "$NPARTS" --workers "${KYS3B_TOKENIZE_WORKERS:-16}" \
    parquet --dataset "$IN" --column text --glob-pattern "part_*.parquet" &
TP=$!
while kill -0 $TP 2>/dev/null; do
  sleep 60
  newest=$(find "$OUT" "$LOG" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  if [[ -n "$newest" ]] && (( $(date +%s) - ${newest%.*} > ${KYS3B_TOKENIZE_STALL_MIN:-30} * 60 )); then
    echo "[tok3b] no output for ${KYS3B_TOKENIZE_STALL_MIN:-30} min — stalled worker; killing"
    kill -TERM -- -$TP 2>/dev/null; sleep 10; kill -KILL -- -$TP 2>/dev/null || true; exit 1
  fi
done
wait $TP
fi

# 4. metadata -> tokenizer DIRECTORY; 5. exact totals; 6. epoch geometry
python tools/fix_ds_metadata.py --output-folder "$OUT" --tokenizer-dir "$TOK"
python deploy/skipjack_3b_600m/assert_invariants_3b.py --config "$CFG" --corpus
python deploy/skipjack_3b_600m/epoch_stats_3b.py --setting "$S" --json "$REPORTS/tokenize/$S/epoch_stats.json"
echo "[tok3b] $(date -Is) $S ready: $OUT"
