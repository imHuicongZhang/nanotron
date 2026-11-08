import sys
from huggingface_hub import hf_hub_download, list_repo_files
out=sys.argv[1]
fs=list_repo_files("wytro/KYS-1.5B-Rewritten-v2",revision="b22c8733")
want=[f for f in fs if f.endswith("nanotron/config.yaml")]
print(len(want))
for f in want:
    p=hf_hub_download("wytro/KYS-1.5B-Rewritten-v2",f,revision="b22c8733",cache_dir=out+"/cache",local_dir=out+"/files")
print("done")
