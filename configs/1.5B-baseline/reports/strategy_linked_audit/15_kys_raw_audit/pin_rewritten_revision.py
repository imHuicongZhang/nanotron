"""Which wytro/Know-Your-Sources revision(s) are byte-identical to the local hf_parquet/<arm> copy the raw build read?"""
import hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi
H = Path('/projects/bvandur1/zhuicon1/kys/hf_parquet'); R = 'wytro/Know-Your-Sources'
ARMS = ['diversity_oriented', 'disagreement_aware', 'wrap_inspired', 'rewire_inspired']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while c := f.read(32 << 20): h.update(c)
    return str(p.relative_to(H)), h.hexdigest(), p.stat().st_size
api = HfApi()
commits = api.list_repo_commits(R, repo_type='dataset')
local_files = sorted(p for a in ARMS for p in (H / a).iterdir() if p.is_file())
with ProcessPoolExecutor(3) as ex:
    local = {n: (h, s) for n, h, s in ex.map(sha, local_files)}
out = {'local_files': len(local), 'commits': []}
for c in commits:
    tree = {f.path: f for f in api.list_repo_tree(R, repo_type='dataset', revision=c.commit_id, recursive=True)
            if hasattr(f, 'size') and f.path.split('/')[0] in ARMS}
    same = diff = missing = 0
    for n, (h, s) in local.items():
        f = tree.get(n)
        if f is None: missing += 1; continue
        rh = f.lfs.sha256 if f.lfs else None
        if rh is None:  # small non-LFS file (metadata.json): compare git blob sha1 is not sha256; download-free check by size
            same += int(f.size == s); diff += int(f.size != s); continue
        same += int(rh == h and f.size == s); diff += int(not (rh == h and f.size == s))
    extra = sorted(set(tree) - set(local))
    out['commits'].append({'commit': c.commit_id, 'date': str(c.created_at), 'title': c.title, 'same': same,
                           'different': diff, 'missing_on_hub': missing, 'hub_only_files': extra[:10], 'n_hub_only': len(extra)})
    print(c.commit_id[:8], c.created_at, same, diff, missing, len(extra), c.title[:60], flush=True)
out['local_sha256'] = {n: h for n, (h, s) in local.items()}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
