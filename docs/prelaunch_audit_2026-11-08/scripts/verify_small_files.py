"""Hash the non-LFS files of blab-jhu/KYS-Pre-Rewritten that verify_data.py could not check via LFS metadata
(tokenizer/*, signal-disagreement-lambda0/tokenized/*.ds.metadata) by downloading them, and list the commit history."""
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
REPO = "blab-jhu/KYS-Pre-Rewritten"
EV = Path(__file__).resolve().parent.parent / "evidence"
api = HfApi()
rev = api.dataset_info(REPO).sha
commits = [(c.commit_id[:8], c.created_at.isoformat(), c.title) for c in api.list_repo_commits(REPO, repo_type="dataset")]
m = json.loads((EV / "hub" / "manifest.json").read_text())
sums = {}
for line in (EV / "hub/signal-disagreement-lambda0/tokenized/SHA256SUMS").read_text().splitlines():
    if line.strip():
        h, n = line.split(maxsplit=1); sums[Path(n.strip()).name] = h
res = {"revision": rev, "commits": commits, "tokenizer": {}, "metadata": {}}
dl = EV / "hub_small"
for n, h in m["tokenizer"]["sha256"].items():
    p = hf_hub_download(REPO, f"tokenizer/{n}", repo_type="dataset", revision=rev, local_dir=dl)
    res["tokenizer"][n] = hashlib.sha256(open(p, "rb").read()).hexdigest() == h
for n in sorted(k for k in sums if k.endswith(".metadata")):
    p = hf_hub_download(REPO, f"signal-disagreement-lambda0/tokenized/{n}", repo_type="dataset", revision=rev, local_dir=dl)
    res["metadata"][n] = hashlib.sha256(open(p, "rb").read()).hexdigest() == sums[n]
print("revision", rev)
for c in commits[:12]: print(" ", *c)
print("tokenizer", res["tokenizer"])
print("metadata", len(res["metadata"]), "all match:", all(res["metadata"].values()))
(EV / "verify_small_files.json").write_text(json.dumps(res, indent=1))
