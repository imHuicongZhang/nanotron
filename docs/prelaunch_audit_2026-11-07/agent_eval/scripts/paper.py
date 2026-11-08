import sys,statistics as st,importlib.util
sys.path.insert(0,"/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval")
from kys_eval import config as C, aggregate as A, common
sp=importlib.util.spec_from_file_location("mc","/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/08_evaluation/02_code/mmlu_categories.py")
m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
print("category map identical:", m.SUBJECT_CATEGORY==C.MMLU_SUBJECT_CATEGORY, len(m.SUBJECT_CATEGORY))
from collections import Counter; print(Counter(C.MMLU_SUBJECT_CATEGORY.values()))
rw=A.load_rewritten()
def stats(s,e,b): v=[rw[(s,sd,e)][b] for sd in C.SEEDS]; return st.fmean(v)*100, st.stdev(v)*100
def dstats(s,e,b): v=[(rw[(s,sd,e)][b]-rw[("quality_base",sd,e)][b])*100 for sd in C.SEEDS]; return st.fmean(v), st.stdev(v), sum(x>0 for x in v)
order=["quality_first","diversity_oriented","wrap_inspired","rewire_inspired","disagreement_aware"]
print("per-task ep3 QB:",[ "%.2f±%.2f"%stats("quality_base","ep3",b) for b in ["arc_easy","hellaswag","piqa","siqa","openbookqa","commonsense_qa"]])
for s in order: print(s,[ "%+.2f±%.2f"%dstats(s,"ep3",b)[:2] for b in ["arc_easy","hellaswag","piqa","siqa","openbookqa","commonsense_qa"]])
for b in ["mean6","mmlu"]:
  print(b)
  for s in ["quality_base"]+order:
    print(" ",s,["%.2f±%.2f"%stats(s,e,b)+("" if s=="quality_base" else " %+.2f±%.2f %d/3"%dstats(s,e,b)) for e in C.EPOCHS])
cats=[common.category_key(c) for c in C.MMLU_CATEGORIES]
print("cat ep3 QB",["%.2f±%.2f"%stats("quality_base","ep3",k) for k in cats])
for s in order: print(s,["%+.2f[%d]"%(dstats(s,"ep3",k)[0],dstats(s,"ep3",k)[2]) for k in cats])
print("cat ep2 QB",["%.2f"%stats("quality_base","ep2",k)[0] for k in cats])
for s in order: print(s,["%+.2f"%dstats(s,"ep2",k)[0] for k in cats])
