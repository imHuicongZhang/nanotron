# Raw baselines on skipjack — deployment tools (20-GPU project ceiling)

Deployment-only additions for the 2026-10-16 owner workflow (seven raw baselines, seeds 42 → 43 → 44, on
skipjack). Nothing here changes a dataset, a template, a rendered scientific value, the WSD graph or the
in-job guards of `deploy/slurm/kys_segment.sbatch`. Base: `huicong-dev` @ `488d00f579da60c6a4ea785c7ce91892902f0022`.

| file | purpose |
|---|---|
| `download_inputs.py` | pinned `snapshot_download` of the dataset and init repos; sha256 of every manifest file; `hash_init_checkpoint.py --check` + 180-file check per seed |
| `tokenize_all.sh` | the unchanged `tools/kys_raw/tokenize_raw_text.sh` for all seven settings, after a clean verification report; refuses partial output |
| `smoke.sbatch` | INSTALL.md §3.1 + full `assert_invariants.py` + 20-step real-data run on 4 × H100, online W&B (group `smoke`), nvidia-smi peak |
| `capped_submit.py` | the only supported way to submit training: five serial lanes, `afterok`, state + lock + reconcile |
| `test_capped_submit.py` | fake-Slurm tests: full schedule ≤ 20 GPUs, partial submission, requeue, duplicate invocation, failed upstream + retry, foreign jobs, sbatch-option parity with `plan_submit.py` |

Profile: `deploy/clusters.yaml` → `skipjack_1p5b_baselines`; `baseline_seed_assignment` 42/43/44 → it.

## The ceiling

`plan_submit.py` alone would run 7 trunks at once and let branches overlap later trunks (up to 56 GPUs).
`capped_submit.py` places the 42 segments of a seed into **5 lanes**. Each segment depends (`afterok`) on its
lane predecessor *and* on its scientific upstream, so at most one job per lane can run: **≤ 5 × 4 = 20 GPUs**,
enforced by Slurm, held across restarts and requeues, and costing no idle allocation. Placement is greedy
critical-path list scheduling (≈148 h per seed at 22.8 s/it vs ≈199 h for "two whole chains in one lane").
The branch sources are unchanged: ep1/ep2/ep3 still resume from the fixed trunk steps 4292/8583/12875.

Every project GPU job carries the `kys_` name prefix. `submit` refuses while any other `kys_` job is live
(another seed, a smoke test, an evaluation); `guard --need-gpus N` refuses a non-training job that would take
held + waiting + N above 20. Smoke tests run before a seed's grid, and evaluation runs after its training jobs
have all ended.

## Usage

```bash
P=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python
$P deploy/skipjack_1p5b/capped_submit.py plan      --seed 42 --s-per-it <measured>   # lane table, makespan, peak
$P deploy/skipjack_1p5b/capped_submit.py test-only --seed 42 --s-per-it <measured>   # sbatch --test-only x 42
$P deploy/skipjack_1p5b/capped_submit.py submit    --seed 42 --s-per-it <measured>   # idempotent
$P deploy/skipjack_1p5b/capped_submit.py status    --seed 42                         # progress table + GPUs
$P deploy/skipjack_1p5b/capped_submit.py usage                                       # project GPUs right now
```

State: `<data root>/reports/capped_submit/seed<S>/state.json` (plan, every attempt, event log) and
`submit.lock`. A seed > 42 is refused until `seed<S-1>/SEED_COMPLETE` exists, which is written by hand only after
that seed's 21 finals are verified, uploaded and evaluated, and its GPU jobs have ended.

**`submit` is safe to rerun at any time.** It reconciles the saved state with squeue, sacct and the final
checkpoints on disk, then submits only what is missing:

| situation | behaviour |
|---|---|
| segment's final checkpoint exists, no live job | DONE, skipped |
| job live (pending, running, requeued, completing) | kept; dependents attach `afterok` to it |
| crash between `sbatch` and recording the id | the single job of that name since the intent is adopted; two → refuse |
| a live `kys_` job with a planned name the tool did not record | refuse (unknown dependencies) |
| job ended without its final (FAILED, TIMEOUT, NODE_FAIL, manual cancel, COMPLETED w/o final) | **lane halts**, reported; resubmitted only with `--retry <segment>` after inspection |
| dependents cancelled by `kill_invalid_depend` | DEP_KILLED; resubmitted automatically once their upstream is live or done again |

`afterok` is attached only to live jobs (`MinJobAge` = 30 s here). A finished upstream counts as satisfied only
when its final checkpoint is on disk. If `sbatch` itself fails, nothing is recorded for that segment and the
command exits; rerun it.

## Tests

```bash
pip install --target /tmp/pt pytest   # anywhere outside the training env
PYTHONPATH=/tmp/pt $P -m pytest -q -p no:cacheprovider deploy/skipjack_1p5b/test_capped_submit.py
```
