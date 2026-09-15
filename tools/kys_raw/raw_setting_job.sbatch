#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --account=bvandur1
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=30
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# One raw-selected setting end to end: sources -> assemble -> five leakage checks -> publish raw_text.
#
#   sbatch --job-name=kysraw_<setting> --output=<log> tools/kys_raw/raw_setting_job.sbatch <setting> [kys_root]
#
# The GPU is only the ticket to a node's CPU cores (1 GPU carries 30 cores on skipjack's h100
# nodes); nothing here touches it. Resumable under scavenger preemption (--requeue): every stage
# checks for its complete output first and is skipped if present.
#   sources   raw_sources/manifest.json has the setting (raw_rewire_inspired: counted here if absent)
#   assemble  raw_corpus/<setting>/_raw_manifest.json records shuffled_rows; otherwise the partial
#             shuffle output is removed and assembly re-runs
#   verify    verify/verify_<setting>.json exists: failed == [] -> skip; failed != [] -> STOP (exit 3)
#   publish   hf_stage/published_<setting>.json has roundtrip_ok true
# The job ends right after its upload and round-trip check, which releases the GPU.
set -euo pipefail
SETTING="${1:?setting}"
K="${2:-/projects/bvandur1/zhuicon1/kys}"
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys
POOL=$W/datasets/ppl-dsai/dclm-refinedweb-100m-sample
source $W/envs/pretrain/bin/activate
cd "$REPO"
NCPU=${SLURM_CPUS_PER_TASK:-8}
log() { echo "[$SETTING $(date '+%F %T') $(hostname) restart=${SLURM_RESTART_COUNT:-0}] $*"; }
log "start: $NCPU cpus, job ${SLURM_JOB_ID:-local}"

have_source() {
  python -c "import json,sys; m=json.load(open('$K/raw_sources/manifest.json')); sys.exit(0 if '$1' in m['settings'] else 1)" 2>/dev/null
}

# --- sources ---------------------------------------------------------------------------------------
if [[ $SETTING == raw_rewire_inspired ]]; then
  # never count concurrently with another writer of raw_sources/: wait for the disagreement+random
  # count to be recorded, and for any other rewire count (log still growing) to finish
  until have_source raw_disagreement_aware && have_source raw_random; do log "waiting for the disagreement+random count"; sleep 60; done
  RL=$K/logs/build_raw_rewire.log
  while [[ -f $RL ]] && ! grep -q 'wrote .*manifest.json' $RL && (( $(date +%s) - $(stat -c %Y $RL) < 600 )); do
    log "another rewire count is writing $RL; waiting"; sleep 60
  done
  if have_source raw_rewire_inspired; then
    log "sources: already counted"
  else
    log "sources: counting raw_rewire_inspired"
    TOKENIZERS_PARALLELISM=false python tools/kys_raw/build_raw_sources.py --parquet-root $K/hf_parquet --pool $POOL \
        --tokenizer $K/nanotron_tokenized/tokenizer --out $K/raw_sources --workers $NCPU --settings raw_rewire_inspired \
        > $RL 2>&1
    grep 'raw tokens' $RL | sed "s/^/[$SETTING] /"
  fi
else
  until have_source "$SETTING"; do log "waiting for the $SETTING count"; sleep 60; done
fi
log "sources: $(python -c "import json; s=json.load(open('$K/raw_sources/manifest.json'))['settings']['$SETTING']; print({k: s[k] for k in ('source_docs_dedup','raw_tokens_before','docs_after','tokens_after')})")"

# --- assemble --------------------------------------------------------------------------------------
C=$K/raw_corpus/$SETTING
if [[ -f $C/_raw_manifest.json ]] && grep -q '"shuffled_rows"' $C/_raw_manifest.json; then
  log "assemble: already complete"
else
  rm -rf "$C/shuffled" "$C/_shuffle_tmp" "$C/_raw_manifest.json"
  log "assemble: start"
  TOKENIZERS_PARALLELISM=false python tools/kys_raw/assemble_raw_corpus.py --setting "$SETTING" --with-anchor \
      --sources $K/raw_sources --pool $POOL --tokenizer $K/nanotron_tokenized/tokenizer --out $K/raw_corpus \
      --workers $(( NCPU > 4 ? NCPU - 2 : NCPU ))
  log "assemble: done $(cat $C/_raw_manifest.json | tr -d '\n ')"
fi

# --- verify (all five checks, full anchor check) ---------------------------------------------------
V=$K/verify/verify_$SETTING.json
mkdir -p $K/verify
if [[ ! -f $V ]]; then
  log "verify: start"
  python tools/kys_raw/verify_raw_corpus.py --setting "$SETTING" --corpus "$C" --sources $K/raw_sources \
      --parquet-root $K/hf_parquet --pool $POOL --tokenizer $K/nanotron_tokenized/tokenizer --out $K/verify \
      > $K/verify/verify_$SETTING.log 2>&1 || true
fi
[[ -f $V ]] || { log "verify: no result written — see $K/verify/verify_$SETTING.log"; exit 3; }
FAILED=$(python -c "import json; print(json.load(open('$V'))['failed'])")
if [[ $FAILED != "[]" ]]; then
  log "verify: CHECKS FAILED $FAILED — stopping, nothing published (see $V)"; exit 3
fi
log "verify: all five checks passed"

# --- publish ---------------------------------------------------------------------------------------
P=$K/hf_stage/published_$SETTING.json
if [[ -f $P ]] && python -c "import json,sys; sys.exit(0 if json.load(open('$P'))['roundtrip_ok'] else 1)"; then
  log "publish: already done"
else
  [[ -s $K/code_commit.txt ]] || { log "no $K/code_commit.txt"; exit 3; }
  log "publish: start (code commit $(cat $K/code_commit.txt))"
  python tools/kys_raw/publish_raw_text.py --kys-root $K --setting "$SETTING" --code-commit "$(cat $K/code_commit.txt)"
fi
log "UPLOAD_ROUNDTRIP_DONE $(tr -d '\n' < $P)"
