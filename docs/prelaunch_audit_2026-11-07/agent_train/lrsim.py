import torch
def mk(ds,dsteps,peak=5e-4,warm=500,minlr=0.0):
    def lam(t):
        if t<=warm: v=peak*t/warm
        elif t<ds: v=peak
        elif t<ds+dsteps: v=minlr+(peak-minlr)*(dsteps-(t-ds))/dsteps
        else: v=minlr
        return v/peak
    return lam
p=torch.nn.Parameter(torch.zeros(1))
o=torch.optim.AdamW([p],lr=5e-4)
s=torch.optim.lr_scheduler.LambdaLR(o,mk(12875,1430))
used={}
for t in range(1,4293):
    used[t]=o.param_groups[0]['lr']; o.step(); s.step()
sd=s.state_dict(); osd=o.state_dict()
print("trunk step1 lr",used[1],"step500",used[500],"step501",used[501],"step4292",used[4292],"saved last_epoch",sd['last_epoch'])
o2=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=5e-4); o2.load_state_dict(osd)
s2=torch.optim.lr_scheduler.LambdaLR(o2,mk(4292,476)); s2.load_state_dict(sd); s2._initial_step()
u2={}
for t in range(4293,4769):
    u2[t]=o2.param_groups[0]['lr']; o2.step(); s2.step()
print("branch 4293",u2[4293],"4294",u2[4294],"4767",u2[4767],"4768",u2[4768], "n nonzero",sum(v>0 for v in u2.values()))
# also trunk2 resume from trunk1 (same lambda)
o3=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=5e-4); o3.load_state_dict(osd)
s3=torch.optim.lr_scheduler.LambdaLR(o3,mk(12875,1430)); s3.load_state_dict(sd); s3._initial_step()
print("trunk2 first step lr", o3.param_groups[0]['lr'], "last_epoch", s3.last_epoch)
