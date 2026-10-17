#!/usr/bin/env bash
# Tokenize the seven raw corpora with the unchanged tools/kys_raw/tokenize_raw_text.sh (16 tasks, CPU only).
#
#   deploy/skipjack_1p5b/tokenize_all.sh <data_root>/raw <reports dir>
#
# Runs only after download_inputs.py wrote a verification report with no failures. A setting whose
# tokenized/ already holds 16 shards with the exact manifest total is skipped (verified again, not
# re-tokenized); partial output is left untouched and stops the run for inspection.
set -euo pipefail
ROOT="${1:?data_root (<data root>/raw)}"; REP="${2:?reports dir}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PY="${PY:-/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python}"
export KYS_TOKENIZE_WORKERS="${KYS_TOKENIZE_WORKERS:-4}"
"$PY" -c "import json,sys; r=json.load(open('$REP/inputs_verification.json')); sys.exit('input verification failed: %s' % r['failed'] if r['failed'] else 0)"

total() {  # prints the summed token count and shard count of a tokenized folder
  "$PY" - "$1" <<'PY'
import glob, sys
meta = sorted(glob.glob(f"{sys.argv[1]}/*.ds.metadata"))
print(sum(int(open(m).read().splitlines()[1]) for m in meta), len(glob.glob(f"{sys.argv[1]}/*.ds")))
PY
}
for s in raw_diversity_oriented raw_disagreement_aware raw_random raw_rewire_inspired \
         raw_top10b_fineweb_edu raw_top10b_modernbert raw_top10b_consensus; do
  out="$ROOT/$s/tokenized"
  want=$("$PY" -c "import json; print(json.load(open('$ROOT/manifest.json'))['settings']['$s']['expected_total_tokens'])")
  if compgen -G "$out/*.ds" >/dev/null; then
    read -r got n < <(total "$out")
    if [[ $got == "$want" && $n == 16 ]]; then echo "[all] $s: already tokenized, $got tokens, 16 shards"; continue; fi
    echo "[all] $s: partial/mismatched output ($n shards, $got vs $want tokens) — stopping for inspection"; exit 1
  fi
  echo "[all] $(date -Is) $s: start"
  "$REPO/tools/kys_raw/tokenize_raw_text.sh" "$ROOT" "$s" "$ROOT/tokenizer"
  echo "[all] $(date -Is) $s: done"
done
"$PY" - "$ROOT" "$REP/tokenized_totals.json" <<'PY'
import glob, json, sys
root, out = sys.argv[1], sys.argv[2]
man = json.load(open(f"{root}/manifest.json"))
res = {}
for s, rec in man["settings"].items():
    meta = sorted(glob.glob(f"{root}/{s}/tokenized/*.ds.metadata"))
    tot = sum(int(open(m).read().splitlines()[1]) for m in meta)
    toks = {open(m).read().splitlines()[0] for m in meta}
    res[s] = {"shards": len(meta), "tokens": tot, "expected": rec["expected_total_tokens"],
              "match": tot == rec["expected_total_tokens"] and len(meta) == 16, "tokenizer_in_metadata": sorted(toks)}
json.dump(res, open(out, "w"), indent=1)
for s, r in res.items():
    print(f"[all] {s:24s} {r['shards']:2d} shards {r['tokens']:>15,} tokens  expected {r['expected']:>15,}  {'OK' if r['match'] else 'MISMATCH'}")
sys.exit(0 if all(r["match"] for r in res.values()) else 1)
PY
