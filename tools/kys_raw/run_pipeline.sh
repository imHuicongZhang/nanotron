#!/usr/bin/env bash
# Assemble, leakage-check and publish the four raw-selected baseline corpora as RAW TEXT.
#
#   tools/kys_raw/run_pipeline.sh <kys_root> [workers]
#
# Per setting, in order:
#   1. assemble   anchor + selected docs, text read only from the 100M pool, seed-42 pp_io.bucketed_shuffle
#   2. verify     all five leakage checks. The first setting checks all 4,120,164 anchor documents against
#                 the pool (and the exact 5,000,002,332 token total); later settings compare their anchor to
#                 it by (orig_doc_id, sha256) digest. The pipeline STOPS at the first failed check.
#   3. publish    tools/kys_raw/publish_raw_text.py uploads raw_text/<setting>/ (16 ordered parquet files),
#                 refreshes manifest.json and README.md on the Hub, and round-trips one file's sha256.
#
# There is no tokenization here: the consumer tokenizes raw_text/<setting>/ with
# tools/kys_raw/tokenize_raw_text.sh (configs/1.5B-baseline/RUNBOOK.md).
#
# Inputs under <kys_root>: raw_sources/ (build_raw_sources.py, all four settings), hf_parquet/,
# nanotron_tokenized/tokenizer, code_commit.txt (the pushed commit recorded in manifest.json).
set -euo pipefail
K="${1:?kys root}"; WORKERS="${2:-4}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python}"
POOL=/weka/scratch/jhu/bvandur1/zhuicon1/datasets/ppl-dsai/dclm-refinedweb-100m-sample
SETTINGS=(raw_diversity_oriented raw_disagreement_aware raw_random raw_rewire_inspired)
cd "$REPO"
log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

"$PY" - "$K/raw_sources/manifest.json" "${SETTINGS[@]}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
missing = [s for s in sys.argv[2:] if s not in m.get("settings", {})]
if missing:
    sys.exit(f"raw_sources/manifest.json lacks {missing}; run build_raw_sources.py first")
arms = {"diversity_oriented", "disagreement_aware", "wrap_inspired", "rewire_inspired"}
if not m["anchor"]["matches_recorded"] or set(m["anchor"]["identical_across"]) != arms:
    sys.exit(f"anchor not confirmed identical across all four arms: {m['anchor']}")
print("[pipeline] sources complete for all four settings; anchor identical across arms")
PY

mkdir -p "$K/verify"
for i in "${!SETTINGS[@]}"; do
  s=${SETTINGS[$i]}

  if [[ -f "$K/raw_corpus/$s/_raw_manifest.json" ]] && grep -q '"shuffled_rows"' "$K/raw_corpus/$s/_raw_manifest.json"; then
    log "$s: already assembled"
  else
    log "$s: assemble"
    "$PY" tools/kys_raw/assemble_raw_corpus.py --setting "$s" --with-anchor --sources "$K/raw_sources" \
        --pool "$POOL" --tokenizer "$K/nanotron_tokenized/tokenizer" --out "$K/raw_corpus" --workers "$WORKERS"
  fi

  V=$K/verify/verify_$s.json
  if [[ -f $V ]] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$V'))['failed'] == [] else 1)"; then
    log "$s: leakage checks already passed (recorded in $V)"
  else
    ref=()
    [[ $i -gt 0 ]] && ref=(--anchor-ref "$K/verify/verify_${SETTINGS[0]}.json")
    log "$s: leakage checks"
    if ! "$PY" tools/kys_raw/verify_raw_corpus.py --setting "$s" --corpus "$K/raw_corpus/$s" --sources "$K/raw_sources" \
          --parquet-root "$K/hf_parquet" --pool "$POOL" --tokenizer "$K/nanotron_tokenized/tokenizer" \
          "${ref[@]}" --out "$K/verify" > "$K/verify/verify_$s.log" 2>&1; then
      log "$s: verify exited non-zero — a check failed, or it wrote no report; see $V and $K/verify/verify_$s.log; stopping (nothing further is published)"
      exit 1
    fi
    log "$s: all five checks passed"
  fi

  until [[ -s "$K/code_commit.txt" ]]; do
    log "$s: waiting for $K/code_commit.txt (pushed commit hash for manifest.json)"; sleep 300
  done
  P=$K/hf_stage/published_$s.json
  if [[ -f $P ]] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$P'))['roundtrip_ok'] else 1)"; then
    log "$s: already published (round-trip recorded in $P)"
  else
    log "$s: publish raw_text/$s"
    "$PY" tools/kys_raw/publish_raw_text.py --kys-root "$K" --setting "$s" --code-commit "$(cat "$K/code_commit.txt")"
    log "$s: published"
  fi
done
log "done: all four settings checked and published"
