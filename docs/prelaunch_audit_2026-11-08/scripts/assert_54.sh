#!/usr/bin/env bash
# §6: render the 54 rebuttal configs at the current huicong-dev HEAD against local data and run the FULL
# tools/assert_invariants.py (corpus totals, tokenizer metadata, batch, environment, placeholders) on each.
# Uses profile.yaml here: a copy of the 2026-11-07 b1_fix profile with data_root / ckpt_root / wandb.dir moved into
# ../work/ (shared storage). Usage: bash assert_54.sh <python with nanotron installed>
set -uo pipefail
E=$1; A=$(cd "$(dirname "$0")/.." && pwd); REPO=$(cd "$A/../.." && pwd); W=$A/work/assert54
rm -rf "$W"; mkdir -p "$W/data_root" "$W/ckpt" "$W/wandb" "$A/logs/assert54"
R=/home/jhu/zhuicon1/projects_bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw
L0=/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0
for s in raw_diversity_oriented raw_disagreement_aware; do mkdir -p "$W/data_root/$s"; ln -s "$R/$s/tokenized" "$W/data_root/$s/tokenized"; done
ln -s "$R/manifest.json" "$W/data_root/manifest.json"   # verified = Hub @ afd1f177 (§2)
mkdir -p "$W/data_root/signal-disagreement-lambda0"; ln -s "$L0/tokenized" "$W/data_root/signal-disagreement-lambda0/tokenized"
"$E" - "$A/scripts/profile.yaml" "$W" <<'PY'
import sys, yaml
p, w = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p))
c = d["clusters"]["marc-cluster"]
c["data_root"], c["ckpt_root"], c["wandb"]["dir"] = f"{w}/data_root", f"{w}/ckpt", f"{w}/wandb"
yaml.safe_dump(d, open(f"{w}/profile.yaml", "w"), sort_keys=False)
PY
cd "$REPO"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
pass=0; fail=0
for S in 42 43 44; do
  "$E" tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed $S --settings $W1 --profile "$W/profile.yaml" \
      --out-dir "$W/filled/seed$S" > "$A/logs/assert54/fill_seed$S.log" 2>&1 || { echo "FILL FAIL seed $S"; continue; }
  for c in "$W"/filled/seed$S/*.yaml; do
    n=$(basename "$c" .yaml)
    if ( set -a; source "${c%.yaml}.env"; "$E" tools/assert_invariants.py --config "$c" ) > "$A/logs/assert54/$n.log" 2>&1; then
      pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL $n"; fi
  done
done
echo "full assert_invariants: pass=$pass fail=$fail (expected 54/0)"
grep -h "tokens\b.*10,000\|10000002885\|10000005069\|10000002556" "$A"/logs/assert54/*trunk1.log | sort | uniq -c | head
