cd "$CODE_ROOT"
(
  n_ok=0; n_fail=0
  for c in configs/1.5B-baseline-seed4{2,3,4}/filled/*_seed4?_*.yaml configs/1.5B-baseline-seed44/filled-dp8/*.yaml; do
    [ -f "$c" ] || continue
    if ( set -a; source "${c%.yaml}.env"; python tools/assert_invariants.py --config "$c" ) > "${c%.yaml}.invariants.log" 2>&1; then
      n_ok=$((n_ok + 1)); echo "OK   $c"
    else
      n_fail=$((n_fail + 1)); echo "FAIL $c  (see ${c%.yaml}.invariants.log)"
    fi
  done
  echo "invariants: $n_ok OK, $n_fail FAIL (expected 54 OK, 0 FAIL)"
  if [ "$n_fail" -ne 0 ] || [ "$n_ok" -ne 54 ]; then echo "INVARIANT CHECK FAILED: launch is blocked"; exit 1; fi
  echo "invariant check passed"
)
