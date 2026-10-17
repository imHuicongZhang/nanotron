#!/usr/bin/env bash
#SBATCH --job-name=rw3b_handoff_waiter
#SBATCH --partition=med
#SBATCH --account=bvandur1_blabscify
#SBATCH --qos=jhu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
# PREPARED, NOT STARTED. Durable waiter for the baseline -> 3B handoff (scrontab/crontab are unavailable here, and a
# nohup loop dies with its interactive allocation). One CPU core for a few seconds per check, every 6 h:
#
#   sbatch --output=<TRAIN_ROOT>/logs/slurm/%x.log deploy/skipjack_3b_600m/handoff_waiter.sbatch
#
#   * HELD (any gate condition unmet, including the owner's LAUNCH_APPROVED): resubmit itself with --begin=now+6hours;
#   * handoff reached: submit the 4-GPU smoke through capped_submit_3b.py smoke (cap-checked), then stop.
# It never submits the training grid: that needs the owner's SMOKE_ACCEPTED and an explicit capped_submit_3b.py submit.
# Stop it with `scancel -n rw3b_handoff_waiter`. A second waiter refuses to run.
set -euo pipefail
W=/weka/scratch/jhu/bvandur1/zhuicon1
REPO=$W/projects/nanotron-kys-3b600m
P=$W/envs/pretrain/bin/python
cd "$REPO"
others=$(squeue -h -u "$USER" -n rw3b_handoff_waiter -o %i | grep -vx "${SLURM_JOB_ID:-x}" || true)
[[ -z "$others" ]] || { echo "[waiter] another waiter is queued ($others); exiting without resubmitting"; exit 0; }
REP=$($P -c "import sys; sys.path.insert(0,'deploy/skipjack_3b_600m'); import registry_3b as R; print(R.REPORTS)")
mkdir -p "$REP/handoff"
echo "[waiter] $(date -Is) job=${SLURM_JOB_ID:-} head=$(git rev-parse HEAD)"
if $P deploy/skipjack_3b_600m/handoff_gate.py check --require-approval --json "$REP/handoff/last_check.json" > "$REP/handoff/last_check.txt"; then
  echo "[waiter] handoff reached; submitting the smoke test"
  $P deploy/skipjack_3b_600m/capped_submit_3b.py smoke | tee -a "$REP/handoff/smoke_submission.txt"
  exit 0
fi
tail -1 "$REP/handoff/last_check.txt"
sbatch --begin=now+6hours --output="$REP/../logs/slurm/rw3b_handoff_waiter.log" deploy/skipjack_3b_600m/handoff_waiter.sbatch
echo "[waiter] resubmitted for +6 h"
