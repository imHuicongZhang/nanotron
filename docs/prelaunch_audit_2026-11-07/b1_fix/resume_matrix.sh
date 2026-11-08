S=$1; E=$2; cd $3
start_of() { case $1 in trunk1) echo 0;; trunk2) echo 4292;; trunk3) echo 8583;; esac; }
AI() { ( set -a; source $2; $E tools/assert_invariants.py --config $1 --check-resume ) > $3 2>&1; }
p=0; f=0; r_ok=0; r_bad=0; guard=0; parse=0
for c in $S/filled/*/*.yaml $S/filled-dp8/*.yaml; do
  env=${c%.yaml}.env; k=$(basename $c .yaml); k=${k##*_}
  # the trunk directory of this chain (ep configs resume from <trunk dir>/<step>)
  T=$($E -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]))['checkpoints']['resume_checkpoint_path'];print(c.rsplit('/',1)[0] if c.rsplit('/',1)[1].isdigit() else c)" $c)
  mkdir -p $T; for st in 0 1500 3000 4292 8583 12875; do mkdir -p $T/$st; touch $T/$st/model_config.json; done
  # (1) clean start of this segment
  if [[ $k == trunk* ]]; then echo $(start_of $k) > $T/latest.txt; fi
  R=$($E tools/kys_raw/trunk_restart_config.py $c 2>/dev/null)
  if [[ $R == $c ]] && AI $c $env ${c%.yaml}.resume.log; then p=$((p+1)); else f=$((f+1)); echo "CLEAN FAIL $c"; fi
  # (2) trunk1 restarted from 1500
  if [[ $k == trunk1 ]]; then
    echo 1500 > $T/latest.txt
    AI $c $env ${c%.yaml}.restart_raw.log && echo "GUARD MISSED $c" || { grep -q 'restart the LR schedule' ${c%.yaml}.restart_raw.log && guard=$((guard+1)); }
    R=$($E tools/kys_raw/trunk_restart_config.py $c 2>/dev/null)
    if [[ $R != $c ]] && AI $R $env ${c%.yaml}.restart.log; then r_ok=$((r_ok+1)); else r_bad=$((r_bad+1)); echo "RESTART FAIL $c"; fi
    $E -c "
import sys,logging; logging.disable(logging.CRITICAL)
from nanotron.config import get_config_from_file as g
a=g(sys.argv[1]); b=g(sys.argv[2])
assert (b.checkpoints.load_optimizer, b.checkpoints.load_lr_scheduler)==(True,True)
for f in ('general','model','optimizer','tokens','data_stages','tokenizer','logging'): assert getattr(a,f)==getattr(b,f), f
import dataclasses
for f in dataclasses.fields(a.parallelism):
    x,y=getattr(a.parallelism,f.name),getattr(b.parallelism,f.name)
    assert (type(x)==type(y)) if f.name=='pp_engine' else x==y, f.name
a.checkpoints.load_optimizer=a.checkpoints.load_lr_scheduler=True; assert a.checkpoints==b.checkpoints
" $c $R >/dev/null 2>&1 && parse=$((parse+1)) || echo "PARSE DIFF $c"
    echo 0 > $T/latest.txt
  fi
done
echo "clean-start --check-resume: pass=$p fail=$f"
echo "trunk1 restart@1500: original refused by guard=$guard, state-loading copy passes=$r_ok fail=$r_bad, nanotron-parsed copy == original except load flags: $parse"
