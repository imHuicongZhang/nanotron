from huggingface_hub import HfApi, hf_hub_download
import os
from pathlib import Path
# agent_eval/hfdl, next to this scripts/ folder (the audit first ran this from a node-local /tmp scratchpad)
api=HfApi(); D=str(Path(__file__).resolve().parent.parent / "hfdl")
for repo,rev in [("wytro/KYS-1.5B-Rewritten-v2","b22c8733"),("wytro/KYS-1.5B-Rewritten-v2",None)]:
    try:
        info=api.model_info(repo,revision=rev, files_metadata=True)
        print(repo,rev,"->",info.sha,"private",info.private)
        files=[(s.rfilename,s.size,getattr(s,'lfs',None) and s.lfs.sha256) for s in info.siblings]
        print(len(files),"files")
        for f in files:
            if "diversity" in f[0].lower() and "seed42" in f[0] or not "/" in f[0]: print("  ",f)
        os.makedirs(D+"/"+(rev or "main"),exist_ok=True)
        p=hf_hub_download(repo,"README.md",revision=info.sha,local_dir=D+"/"+(rev or "main"))
        print(" readme",p)
    except Exception as e: print(repo,rev,"ERR",repr(e)[:300])
for repo in ["blab-jhu/KYS-1.5B-Raw-Selected-Baselines"]:
    try:
        info=api.model_info(repo); print(repo,info.sha,"private",info.private)
        fs=api.list_repo_files(repo); print(len(fs)); print("\n".join(sorted(fs)[:80]))
    except Exception as e: print(repo,"ERR",repr(e)[:300])
