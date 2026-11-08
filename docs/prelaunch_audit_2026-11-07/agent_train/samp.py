import sys,itertools
sys.path.insert(0,"/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-lambda0/src")
from nanotron.data.samplers import MegatronPretrainingSampler
import torch.distributed as dist, os
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29911")
dist.init_process_group("gloo",rank=0,world_size=1)
def steps(dp,accum,consumed,nsteps):
    its=[iter(MegatronPretrainingSampler(total_samples=consumed+nsteps*1024,consumed_samples=consumed,micro_batch_size=32,
        data_parallel_rank=d,data_parallel_size=dp,drop_last=True,global_batch_size=1024)) for d in range(dp)]
    out=[]
    for s in range(nsteps):
        mbs=[]
        for k in range(accum):
            for d in range(dp): mbs.append(tuple(next(its[d])))
        out.append(sorted(mbs))
    return out
for c in (0, 4292*1024, 8583*1024):
    a=steps(4,8,c,5); b=steps(8,4,c,5)
    print(c, a==b, a[0][0][:3], len(a[0]), all(len(set(x))==32 for x in a[0]), a[0][-1][-1]-a[0][0][0]+1)
