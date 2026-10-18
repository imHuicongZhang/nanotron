#!/usr/bin/env bash
#SBATCH --job-name=rw3b_handoff_waiter
#SBATCH --partition=med
#SBATCH --account=bvandur1_blabscify
#SBATCH --qos=jhu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
# Durable waiter for the baseline -> 3B handoff (scrontab/crontab are unavailable here, and a nohup loop dies with its
# interactive allocation). CPU only, one tick every 2 h; logic and state: deploy/skipjack_3b_600m/handoff_waiter.py.
#
#   sbatch --output=<TRAIN_ROOT>/logs/slurm/rw3b_handoff_waiter.log deploy/skipjack_3b_600m/handoff_waiter.sbatch
#
# It submits at most the bounded 4-GPU smoke (capped_submit_3b.py smoke, fully gated) and then only follows it; it never
# submits the training grid. Stop: touch <TRAIN_ROOT>/reports/handoff/STOP (or scancel -n rw3b_handoff_waiter).
set -uo pipefail
W=/weka/scratch/jhu/bvandur1/zhuicon1
cd "$W/projects/nanotron-kys-3b600m"
echo "[waiter] $(date -Is) job=${SLURM_JOB_ID:-} host=$(hostname)"
"$W/envs/pretrain/bin/python" deploy/skipjack_3b_600m/handoff_waiter.py tick
