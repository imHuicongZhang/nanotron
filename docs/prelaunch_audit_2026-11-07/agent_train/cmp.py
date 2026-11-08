import sys,yaml,glob,os,collections,re,json
sys.path.insert(0,"/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-lambda0/src")
import nanotron; assert "nanotron-kys-lambda0" in nanotron.__file__, nanotron.__file__
from nanotron.config import get_config_from_file
R="/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-lambda0/configs"
S=sys.argv[1]; V=sys.argv[2]
SET=["raw_diversity_oriented","raw_disagreement_aware","signal-disagreement-lambda0"]
KINDS=["trunk1","trunk2","trunk3","ep1","ep2","ep3"]
COMP={"raw_diversity_oriented":"diversity_oriented","raw_disagreement_aware":"disagreement_aware","signal-disagreement-lambda0":"disagreement_aware"}
def flat(d,p=""):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,f"{p}.{k}" if p else k))
    elif isinstance(d,list):
        for i,v in enumerate(d): o.update(flat(v,f"{p}[{i}]"))
    else: o[p]=d
    return o
IDS={"general.run","checkpoints.checkpoints_path","checkpoints.resume_checkpoint_path","data_stages[0].data.dataset.dataset_folder[0]","tokenizer.tokenizer_name_or_path","data_stages[0].data.dataset.tokenizer_name"}
def norm(f,seed):
    out={}
    for k,v in f.items():
        if k in IDS: continue
        out[k]=v
    return out
# (a) templates across the 9 per kind
print("=== (a) templates: fields differing across 9 chains within a kind (after dropping run name) ===")
T={}
for s in SET:
  for sd in (42,43,44):
    for k in KINDS:
      p=f"{R}/1.5B-baseline-seed{sd}/templates/{s}_seed{sd}_{k}.yaml"
      T[(s,sd,k)]=flat(yaml.safe_load(open(p)))
print("loaded templates",len(T))
for k in KINDS:
    rows={key:v for key,v in T.items() if key[2]==k}
    keys=set().union(*[r.keys() for r in rows.values()])
    for f in sorted(keys):
        if f=="general.run": continue
        vals=collections.defaultdict(list)
        for key,r in rows.items(): vals[repr(r.get(f,"<MISSING>"))].append(key)
        if len(vals)>1:
            print(k,f,{v:len(n) for v,n in vals.items()} if f in("general.seed","data_stages[0].data.seed") else dict(vals))
# check seeds consistent
bad=[(key,r["general.seed"],r["data_stages[0].data.seed"]) for key,r in T.items() if not (r["general.seed"]==key[1]==r["data_stages[0].data.seed"])]
print("seed mismatches",bad)
# shipped placeholder versions vs templates
print("=== shipped placeholder-rendered (one level up) vs template: any experiment field diffs ===")
for (s,sd,k),r in T.items():
    p=f"{R}/1.5B-baseline-seed{sd}/{s}_seed{sd}_{k}.yaml"
    sh=flat(yaml.safe_load(open(p)))
    for f in set(r)|set(sh):
        if f in r and r.get(f)!=sh.get(f): print("SHIPPED DIFF",s,sd,k,f,r.get(f),sh.get(f))
# (b) rendered via nanotron parse vs v2
print("=== (b) rendered (parsed+serialized by nanotron) vs v2 config.yaml ===")
def parsed(p):
    c=get_config_from_file(p)
    return flat(c.as_dict())
diffsum=collections.Counter(); examples={}
for sub in ("dp4","dp8"):
  for p in sorted(glob.glob(f"{S}/rebuttal_render/{sub}/*.yaml")):
    name=os.path.basename(p)[:-5]; s,rest=name.rsplit("_seed",1); sd,k=rest.split("_")
    if not k.startswith("ep"): continue
    a=parsed(p)
    vp=f"{V}/rewrite-1p5b/seed{sd}/{COMP[s]}/{k}/nanotron/config.yaml"
    b=flat(yaml.safe_load(open(vp)))
    for f in sorted(set(a)|set(b)):
        if f in IDS: continue
        va,vb=a.get(f,"<MISSING>"),b.get(f,"<MISSING>")
        if str(va)!=str(vb):
            key=(sub,f,str(va),str(vb)); diffsum[key]+=1
for key,n in sorted(diffsum.items()): print(n,key)
