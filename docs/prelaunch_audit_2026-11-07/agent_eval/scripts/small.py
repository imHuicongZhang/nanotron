from huggingface_hub import hf_hub_download
import hashlib,json
from pathlib import Path
# agent_eval/hfdl/small, next to this scripts/ folder (the audit first ran this from a node-local /tmp scratchpad)
D=str(Path(__file__).resolve().parent.parent / "hfdl" / "small")
e=json.load(open("/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval/reference/expected_diversity_oriented_seed42_ep3.json"))["checkpoint"]["files_sha256"]
for f in ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json"]:
    p=hf_hub_download("wytro/KYS-1.5B-Rewritten-v2","rewrite-1p5b/seed42/diversity_oriented/ep3/hf/"+f,revision="b22c87333783945e1c655ae46886b7e842ee4203",local_dir=D)
    h=hashlib.sha256(open(p,"rb").read()).hexdigest(); print(f,h==e[f])
