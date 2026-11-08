import sys,yaml,glob,os,collections
def flat(d,p=""):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,f"{p}.{k}" if p else k))
    elif isinstance(d,list):
        for i,v in enumerate(d): o.update(flat(v,f"{p}[{i}]"))
    else: o[p]=d
    return o
V=sys.argv[1]
rows={}
for f in sorted(glob.glob(V+"/rewrite-1p5b/seed*/*/ep*/nanotron/config.yaml")):
    parts=f.split("/"); seed,arm,ep=parts[-5],parts[-4],parts[-3]
    rows[f"{seed}/{arm}/{ep}"]=flat(yaml.safe_load(open(f)))
keys=sorted(set().union(*[r.keys() for r in rows.values()]))
skip=("run","checkpoints_path","resume_checkpoint_path","dataset_folder","general.seed","data.seed")
for k in keys:
    vals=collections.Counter()
    by=collections.defaultdict(list)
    for n,r in rows.items():
        v=r.get(k,"<MISSING>"); vals[repr(v)]+=1; by[repr(v)].append(n)
    if len(vals)>1:
        if any(s in k for s in skip) and k!="general.seed": 
            print("VAR(id)",k,len(vals)); continue
        print("VAR",k)
        for v,ns in by.items():
            print("   ",v,len(ns), ns if len(ns)<8 else ns[:3])
