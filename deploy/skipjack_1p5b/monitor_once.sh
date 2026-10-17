#!/usr/bin/env bash
# One monitoring snapshot: project GPU allocation from squeue (alarm above 20) and the seed's progress table.
#   deploy/skipjack_1p5b/monitor_once.sh <seed>     -> appends to <data root>/logs/monitor_seed<S>.log
# Read-only: never submits, cancels or modifies jobs.
set -uo pipefail
SEED="${1:?seed}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
P=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python
LOG=/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/logs/monitor_seed$SEED.log
{
  echo "=== $(date -Is)"
  U=$("$P" "$REPO/deploy/skipjack_1p5b/capped_submit.py" usage 2>&1)
  echo "$U" | tail -1
  held=$(echo "$U" | sed -n 's/^project GPUs allocated now: \([0-9]*\).*/\1/p')
  [[ -n "$held" && "$held" -gt 20 ]] && echo "ALARM: project holds $held GPUs > 20"
  "$P" "$REPO/deploy/skipjack_1p5b/capped_submit.py" status --seed "$SEED" 2>&1 | tail -3
} >> "$LOG" 2>&1
