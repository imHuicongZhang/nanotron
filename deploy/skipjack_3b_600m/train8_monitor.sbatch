#!/usr/bin/env bash
#SBATCH --job-name=rw3b_train8_monitor
#SBATCH --partition=med
#SBATCH --account=bvandur1_blabscify
#SBATCH --qos=jhu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
# Durable monitor of the 3B grid on its dedicated 8-GPU budget (scrontab/crontab are unavailable here, and a nohup loop
# dies with its interactive allocation). CPU only, one tick every 30 min; logic: deploy/skipjack_3b_600m/train8_monitor.py.
#
#   sbatch --output=<TRAIN_ROOT>/logs/slurm/rw3b_train8_monitor.log deploy/skipjack_3b_600m/train8_monitor.sbatch
#
# It reconciles/submits only through train8_3b.py (2 lanes x 4 GPUs), never retries a FAILED segment, and scancels a
# segment whose log shows a demonstrated failure. Stop: touch <TRAIN_ROOT>/reports/train8/STOP.
set -uo pipefail
W=/weka/scratch/jhu/bvandur1/zhuicon1
cd "$W/projects/nanotron-kys-3b600m"
echo "[monitor] $(date -Is) job=${SLURM_JOB_ID:-} host=$(hostname)"
"$W/envs/pretrain/bin/python" deploy/skipjack_3b_600m/train8_monitor.py tick
