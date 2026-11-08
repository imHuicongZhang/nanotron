import sys
sys.path.insert(0,"/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval")
from kys_eval import config as C
from huggingface_hub import HfApi
api=HfApi()
for r,rev in C.DATASET_REVISIONS.items():
    try:
        head=api.dataset_info(r).sha
        info=api.dataset_info(r,revision=rev).sha
        print(r, "pinned exists" if info==rev else info, "HEAD==pin" if head==rev else "HEAD="+head)
    except Exception as e: print(r,"ERR",repr(e)[:200])
