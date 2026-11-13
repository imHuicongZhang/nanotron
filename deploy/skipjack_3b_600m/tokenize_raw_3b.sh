#!/usr/bin/env bash
# SKIPJACK-ONLY, CPU only. Tokenize one raw-baseline corpus (raw_diversity_oriented | raw_disagreement_aware) into nanotron
# .ds shards with the SAME recipe as tokenize_3b.sbatch (datatrove DocumentTokenizer via tools/preprocess_data_parquet.py,
# the pinned llama-2 tokenizer, one </s> per document, no BOS, no re-shuffle, n_tasks = number of shuffled parts, so task
# i tokenizes exactly part i), then relocate metadata and reconcile the exact token total. Resumable for short
# allocations: datatrove skips completed tasks (logging_dir/completions); a task interrupted at STOP_EPOCH is re-run
# from scratch next time (its output files are rewritten). Single writer per setting (flock).
#
#   bash deploy/skipjack_3b_600m/tokenize_raw_3b.sh SETTING WORKERS STOP_EPOCH      # exit 4 = stopped, rerun to resume
#
# Inputs: raw_baselines_prep/stage4_assemble/<s>/shuffled/part_*.parquet, accepted only if reports/stage5_<s>.json has
# "pass": true and every part's sha256 equals state/stage4_shuffle_<s>.json. Memory: up to ~10 GiB per worker.
set -euo pipefail
S=${1:?setting}; WK=${2:?workers}; STOP=${3:?stop epoch}
case "$S" in raw_diversity_oriented|raw_disagreement_aware) ;; *) echo "unknown raw setting $S"; exit 2;; esac
W=/weka/scratch/jhu/bvandur1/zhuicon1
P=/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/raw_baselines_prep
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source $W/envs/pretrain/bin/activate
cd "$REPO"
eval "$(python - "$S" <<'PY'
import sys; sys.path.insert(0, 'deploy/skipjack_3b_600m'); import registry_3b as R
s = sys.argv[1]
print(f'OUT={R.TOKENIZED / s} TOK={R.TOKENIZER} NPARTS={R.CORPUS_PARTS[s]} LOG={R.REPORTS / "tokenize" / s} WANT={R.TOKENIZED_TOKENS[s]}')
PY
)"
IN=$P/stage4_assemble/$S/shuffled
mkdir -p "$LOG" "$OUT"
exec 9>"$LOG/.tokenize_raw.lock"
flock -n 9 || { echo "[tokraw] another tokenizer holds $LOG/.tokenize_raw.lock"; exit 5; }
echo "[tokraw] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} setting=$S head=$(git rev-parse HEAD) workers=$WK stop=$STOP"

# 1. tokenizer = the pinned 1.5B tokenizer (sha256 of every file)
python - "$TOK" <<'PY'
import hashlib, sys
from pathlib import Path
sys.path.insert(0, 'deploy/skipjack_3b_600m'); import registry_3b as R
tok = Path(sys.argv[1]); have = sorted(p.name for p in tok.iterdir() if p.is_file())
assert have == sorted(R.TOKENIZER_SHA256), have
for name, want in R.TOKENIZER_SHA256.items():
    assert hashlib.sha256((tok / name).read_bytes()).hexdigest() == want, name
print(f'[tokraw] tokenizer {tok}: identical to the 1.5B tokenizer (sha256)')
PY

# 2. inputs are exactly the validated parts
python - "$IN" "$S" "$NPARTS" "$P" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, s, n, P = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
v = json.load(open(P / 'reports' / f'stage5_{s}.json'))
assert v.get('pass') is True, f'stage5_{s}.json does not pass; refusing to tokenize'
want = {k: x['sha256'] for k, x in json.load(open(P / 'state' / f'stage4_shuffle_{s}.json'))['files'].items()}
have = sorted(p.name for p in d.glob('part_*.parquet'))
assert have == sorted(want) and len(have) == n, (len(have), len(want), n)
for name in have:
    h = hashlib.sha256()
    with open(d / name, 'rb') as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    assert h.hexdigest() == want[name], f'{d / name}: sha256 differs from stage4_shuffle_{s}.json'
print(f'[tokraw] {s}: {n} parts match stage4_shuffle_{s}.json sha256; stage 5 passed')
PY

# 3. tokenize (resumable)
ndone() { ls "$LOG/datatrove/completions" 2>/dev/null | wc -l; }
if [[ "$(ndone)" != "$NPARTS" ]]; then
  (( STOP - $(date +%s) > 900 )) || { echo "[tokraw] < 15 min before STOP; not starting"; exit 4; }
  echo "[tokraw] $(date -Is) tokenizing: $(ndone)/$NPARTS tasks already complete"
  setsid python tools/preprocess_data_parquet.py \
      --tokenizer-name-or-path "$TOK/tokenizer.json" --eos-token "</s>" \
      --output-folder "$OUT" --logging-dir "$LOG/datatrove" --n-tasks "$NPARTS" --workers "$WK" \
      parquet --dataset "$IN" --column text --glob-pattern "part_*.parquet" &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    sleep 20
    if (( $(date +%s) >= STOP )); then
      echo "[tokraw] $(date -Is) STOP reached with $(ndone)/$NPARTS tasks complete; killing process group $TP"
      kill -TERM -- -$TP 2>/dev/null || true; sleep 15; kill -KILL -- -$TP 2>/dev/null || true
      exit 4
    fi
  done
  wait $TP || { echo "[tokraw] tokenizer exited non-zero ($(ndone)/$NPARTS complete)"; exit 1; }
fi
[[ "$(ndone)" == "$NPARTS" ]] || { echo "[tokraw] only $(ndone)/$NPARTS tasks complete"; exit 1; }

# 4. metadata -> tokenizer DIRECTORY; 5. exact totals
python tools/fix_ds_metadata.py --output-folder "$OUT" --tokenizer-dir "$TOK"
python - "$OUT" "$TOK" "$NPARTS" "$WANT" "$P/reports/tokenize_$S.json" "$S" <<'PY'
import json, sys, time
from pathlib import Path
out, tok, n, want, rep, s = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
shards = sorted(out.glob('*.ds')); errs = []; total = 0; per = {}
for ds in shards:
    ml = Path(f'{ds}.metadata').read_text().splitlines()
    k = int(ml[1]); total += k; per[ds.name] = k
    if ml[0] != f'{tok}|2': errs.append(f'{ds}: metadata line 1 {ml[0]!r}')
    if ds.stat().st_size != 2 * k: errs.append(f'{ds}: size != 2 x tokens')
    if not Path(f'{ds}.index').is_file(): errs.append(f'{ds}: no .index')
if len(shards) != n: errs.append(f'{len(shards)} shards != {n}')
if total != want: errs.append(f'tokens {total} != expected {want}')
r = {'setting': s, 'shards': len(shards), 'tokens': total, 'expected_tokens': want, 'per_shard': per, 'problems': errs,
     'pass': not errs, 'checked': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
json.dump(r, open(rep, 'w'), indent=1)
print(f'[tokraw] {s}: {len(shards)} shards, {total:,} tokens (expected {want:,}):', 'PASS' if not errs else f'FAIL {errs}')
sys.exit(1 if errs else 0)
PY
echo "[tokraw] $(date -Is) $S tokenized and reconciled: $OUT"
