cd "$CODE_ROOT"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset",
                  revision="afd1f177aa08f5bab9f5695048cc7b1be428d6fa", local_dir=os.environ["DATA_ROOT"],
                  allow_patterns=["manifest.json", "tokenizer/*", "raw_text/raw_diversity_oriented/*",
                                  "raw_text/raw_disagreement_aware/*", "signal-disagreement-lambda0/tokenized/*"])
snapshot_download("wytro/Know-Your-Sources-init", revision="87e1335647a3105471928e0fd261d92846021d1a",
                  local_dir=os.environ["INIT_ROOT"])
PY
for S in 42 43 44; do
  python tools/hash_init_checkpoint.py "$INIT_ROOT/_init_1.5B_seed$S/0" --check "$INIT_ROOT/init_1.5B_seed$S.hash.json"
  find "$INIT_ROOT/_init_1.5B_seed$S/0" -type f | wc -l     # must print 180
done
