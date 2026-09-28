#!/usr/bin/env bash
# Consumer-path tokenization of the four strategy-linked corpora (audit only; nothing is published).
D=/projects/bvandur1/zhuicon1/data/kys-1p5b-strategy-linked-audit
cd /home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/projects/nanotron-kys
for s in raw_diversity_oriented raw_disagreement_aware raw_random raw_rewire_inspired; do
  t0=$(date +%s)
  KYS_TOKENIZE_WORKERS=4 PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python bash tools/kys_raw/tokenize_raw_text.sh $D/tok_root $s > $D/logs/tok_$s.log 2>&1
  echo "$s rc=$? seconds=$(( $(date +%s) - t0 ))" >> $D/logs/tok_summary.txt
done
