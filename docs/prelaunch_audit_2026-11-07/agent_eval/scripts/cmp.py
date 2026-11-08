import json,glob,csv,sys,collections
V2="/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/08_evaluation/04_results_1p5_v2/downstream"
R="/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval/reference/"
stem={"quality-base":"quality_base","quality-first":"quality_first","diversity-first":"diversity_oriented","signal-disagreement-lambda05":"disagreement_aware","wrap":"wrap_inspired","rewrite":"rewire_inspired"}
ep={"4768":"ep1","9537":"ep2","14305":"ep3"}
v2={}
meta=collections.Counter()
fp=collections.Counter()
for f in glob.glob(V2+"/*/*/results/**/results_*.json",recursive=True):
    parts=f[len(V2)+1:].split("/")
    run,step=parts[0],parts[1]
    s,seed=run.rsplit("-10B-1.5B-seed",1)
    d=json.load(open(f))
    g=d["config_general"]
    meta[(g.get("max_samples"),g.get("lighteval_sha"),json.dumps({k:g["model_config"].get(k) for k in ("dtype","batch_size")}))]+=1
    for k,t in d["config_tasks"].items():
        meta[("shots",t.get("num_fewshots"))]+=1
        meta[("rev",t.get("hf_repo"),t.get("hf_revision"),tuple(t.get("evaluation_splits") or []))]+=1
    for k,v in d["summary_tasks"].items():
        fp[v.get("hashes",{}).get("hash_full_prompts")]+=1
    v2[(stem[s],int(seed),ep[step])]=(d,f)
print("n v2",len(v2))
for k,v in sorted(meta.items(),key=str): print(v,k)
print("hash_full_prompts distinct:",fp.most_common(5))
# CSV
rows=list(csv.DictReader(open(R+"rewritten_v2_54_per_task.csv")))
print("csv rows",len(rows))
keys=set(); bad=0; n=0; maxd=0
for r in rows:
    k=(r["setting"],int(r["seed"]),r["epoch"]); keys.add(k)
    d=v2[k][0]["results"]
    ent=d.get(r["task"]+"|0")
    if ent is None: print("missing",k,r["task"]); bad+=1; continue
    for m in ("acc","acc_norm"):
        n+=1; dd=abs(float(r[m])-ent[m]); maxd=max(maxd,dd)
        if dd>1e-9: bad+=1; print("diff",k,r["task"],m,r[m],ent[m])
print("csv keys",len(keys),"cells checked",n,"bad",bad,"maxdiff",maxd)
# per checkpoint tasks covered
cnt=collections.Counter((r["setting"],r["seed"],r["epoch"]) for r in rows)
print("tasks per ckpt",set(cnt.values()))
# json tasks not in csv
for k,(d,f) in v2.items():
    jt={t.split("|")[0] for t in d["results"] if t!="all"}
    ct={r["task"] for r in rows if (r["setting"],int(r["seed"]),r["epoch"])==k}
    if jt!=ct: print("taskset diff",k,jt^ct)
# expected json
e=json.load(open(R+"expected_diversity_oriented_seed42_ep3.json"))
d,f=v2[("diversity_oriented",42,"ep3")]
print("v2 file",f.rsplit("/",1)[1], "expected file", e["recorded"]["results_file"])
bad=0;n=0
for t,ex in e["tasks"].items():
    got=d["results"][t+"|0"]; h=d["summary_tasks"].get(t+"|0",{}).get("hashes",{})
    for m in ("acc","acc_norm","acc_stderr","acc_norm_stderr"):
        n+=1
        if ex.get(m)!=got.get(m): bad+=1; print("exp diff",t,m)
    for hf,hv in ex.get("hashes",{}).items():
        n+=1
        if h.get(hf)!=hv: bad+=1; print("hash diff",t,hf)
print("expected json ntasks",len(e["tasks"]),"fields",n,"bad",bad)
print("expected gpu", e["recorded"], d["config_general"].get("lighteval_sha"), d["config_general"]["model_config"].get("model_name"))
import statistics
an={t.split("|")[0]:v["acc_norm"] for t,v in d["results"].items() if t!="all"}
cs=["rw_arc:easy","rw_hellaswag","rw_piqa","rw_siqa","rw_openbookqa","rw_commonsense_qa"]
print("mean6",statistics.fmean(an[t] for t in cs), e["aggregates"]["mean6_acc_norm"])
mm=[v for t,v in an.items() if t.startswith("rw_mmlu:") and t!="rw_mmlu:_average"]
print("mmlu",len(mm),statistics.fmean(mm), e["aggregates"]["mmlu_macro_acc_norm"])
