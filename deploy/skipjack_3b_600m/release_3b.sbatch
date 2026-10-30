#!/usr/bin/env bash
#SBATCH --job-name=rw3b_release
#SBATCH --partition=med
#SBATCH --account=bvandur1_blabscify
#SBATCH --qos=jhu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=append
# CPU only. Publish the four-setting 3B handoff (code branch + private HF dataset) and verify it; never trains, never
# submits GPU jobs, never touches running jobs. Stages and resume: deploy/skipjack_3b_600m/release_3b.py.
#
#   sbatch [--dependency=afterok:<tokenize job>] --output=<TRAIN_ROOT>/logs/slurm/%x.%j.log deploy/skipjack_3b_600m/release_3b.sbatch
#
# Resubmit the same command after a failure or timeout: completed stages are skipped.
set -uo pipefail
W=/weka/scratch/jhu/bvandur1/zhuicon1
cd "$W/projects/nanotron-kys-3b600m-release"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token} HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[release3b] $(date -Is) job=${SLURM_JOB_ID:-} host=$(hostname) head=$(git rev-parse HEAD)"
"$W/envs/data/bin/python" -u deploy/skipjack_3b_600m/release_3b.py run
