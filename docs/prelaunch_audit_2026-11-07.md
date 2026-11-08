# KYS rebuttal set — prelaunch reproducibility and alignment audit (2026-11-07)

**Scope.** Nine chains: `raw_diversity_oriented`, `raw_disagreement_aware` and `signal-disagreement-lambda0`, each at
seeds 42, 43 and 44. Each chain has six segments, so 54 segments and 27 annealed checkpoints.

**Audited revisions:**
- `imHuicongZhang/nanotron` `huicong-dev` @ `0614206b36393a0ff52deffc682960dc186acca0` (worktree
  `projects/nanotron-kys-lambda0`);
- `imHuicongZhang/kys-eval` `main` @ `c8aa8ff0bdbf2cb159d9f391f88fc1ea5d9cfc04`;
- Hub: `blab-jhu/KYS-Pre-Rewritten` `main` = `afd1f177` (data identical since `8a394d84`; local copies used),
  `wytro/Know-Your-Sources@9e5ff241`, `wytro/Know-Your-Sources-init@87e1335`, `wytro/KYS-1.5B-Rewritten-v2@b22c8733`.

**Mode.** Read-only:
- no repository file, data file or job was changed or submitted;
- this report and its evidence folder [`prelaunch_audit_2026-11-07/`](prelaunch_audit_2026-11-07/) are the only
  files written. They are untracked in this clone, which stays on `skipjack-1p5b-baselines`.

**Paper sources.**
- `paper-writing/06_iclr/` holds only the ICLR appendix: `appendix_training_and_per_task.tex` (Table 11 =
  `tab:hyperparams`, per-task tables), `08_table3_mmlu_category.tex` and `09_appendixF_mmlu_category.tex`.
- **Main-text §3 and §5.1 / 5.3 / 5.4 are not on this cluster.** The only full manuscript, `05_paper/*.pdf`
  (2026-08-24), is the August version. Requirements from those sections are therefore taken from the audit brief
  and are marked as such.
- Table numbers (11–15) are mapped by LaTeX label, because the numbering is not visible in the fragments.

The report is written incrementally: Part F first, then A, B, C, D, E, G.

---

## Part F — prelaunch checklist for the collaborator's agent

Run these in order. Stop at any **BLOCKER**. `$R` is the clone root, `<data_root>` the KYS-Pre-Rewritten download,
`<init_root>` the init download.

### F0. Blockers to resolve before any GPU submission (owner action, not the collaborator's)

| # | blocker | why | how to clear |
|---|---|---|---|
| **B1** | **trunk1 restart silently resets training** (finding T-F1, Part B) | trunk1 renders `load_optimizer: false` and `load_lr_scheduler: false` (`tools/generate_configs.py:175-176`). On any restart after its step-1500 save, `latest.txt` points at 1500 or 3000. nanotron then loads those *weights* (`src/nanotron/trainer.py:1046-1058`), but because `load_lr_scheduler` is false it builds fresh metadata: step 0, consumed samples 0 (`trainer.py:240-259`). Warmup and the data stream restart from sample 0 on already-trained weights, and the run still writes a normal 4292 checkpoint and exits 0. `assert_invariants.py --check-resume` only checks that `latest.txt` resolves (`tools/assert_invariants.py:378-391`). A restart is *likely*: `kys_segment.sbatch` runs with `#SBATCH --requeue` (line 4); §7 allows `slurm.time` ≥ 12 h, while trunk1 takes 27.2 h at 22.8 s/it; and §9 calls resubmission safe. This risk was already flagged on skipjack (2026-10-27) and is not fixed on `huicong-dev`. | **Fix in code before handoff.** Make trunk1 fail closed when `latest.txt` ≠ `0`, or load optimizer, LR and metadata when `latest.txt` > 0, as the 3B project's sbatch does. Correct the "requeue loses at most 1500 steps" claim in WORKFLOW §9 and RUNBOOK step 7. **Interim procedural guard, if the code cannot change:** set `slurm.time` ≥ `1-06:00:00`. Never resubmit a trunk1 that wrote any step > 0; instead delete the trunk directory's steps > 0, write `0` to `latest.txt`, and rerun the whole trunk1. After every trunk1, confirm in its log that the first logged iteration is 1 and that the run started from `latest.txt = 0`. |

The remaining items are checks the collaborator runs. **BLOCKER** means stop and report; **WARN** means record
and continue.

### F1. Revisions (BLOCKER on mismatch)

```bash
git -C $R rev-parse HEAD                     # expect 0614206b36393a0ff52deffc682960dc186acca0, or a later huicong-dev commit
                                             #   with the B1 fix; then `git diff 0614206b -- tools deploy configs` must touch only that fix
git -C <kys-eval> rev-parse HEAD             # expect c8aa8ff0bdbf2cb159d9f391f88fc1ea5d9cfc04 or later
python -c "from huggingface_hub import HfApi as A; a=A(); print(a.repo_info('blab-jhu/KYS-Pre-Rewritten', repo_type='dataset').sha)"
                                             # b264b0ce or later (current main afd1f177; data files unchanged since 8a394d84)
```

Init: download `wytro/Know-Your-Sources-init` at revision `87e1335647a3105471928e0fd261d92846021d1a`. The repo is
gated, so an HF token is needed.

### F2. Inputs: only what the rebuttal set needs

- Download `raw_text/raw_diversity_oriented/*`, `raw_text/raw_disagreement_aware/*`,
  `signal-disagreement-lambda0/tokenized/*`, `tokenizer/*` and `manifest.json`.
- `sha256sum <data_root>/tokenizer/tokenizer.json` → `81bb383c6138159665cd350a59fbcb5aa1fe65712cf5ed9b6481bc802cdbd8e5`
  (BLOCKER).

### F3. Tokenize the two raw controls (BLOCKER on any non-OK line)

```bash
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_diversity_oriented     # KYS_TOKENIZE_WORKERS=4 if RAM < ~170 GiB
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_disagreement_aware
```

Expected output:
- `[tok] tokenizer …: N files match manifest.json sha256`;
- `[tok] <setting>: 16 parquet files match manifest.json sha256`;
- `[tok] raw_diversity_oriented: 16 shards, 10,000,002,885 tokens` / `[tok] raw_disagreement_aware: 16 shards, 10,000,005,069 tokens`;
- `[tok] OK`.

Do not tokenize the other five raw settings.

### F4. λ = 0 corpus: checksum first, then metadata (BLOCKER)

```bash
cd <data_root>/signal-disagreement-lambda0/tokenized && sha256sum -c SHA256SUMS && cd -    # expect 78 × "OK"
python tools/fix_ds_metadata.py --output-folder <data_root>/signal-disagreement-lambda0/tokenized --tokenizer-dir <data_root>/tokenizer
```

After `fix_ds_metadata.py`, a second `sha256sum -c` *will* report the 26 `.ds.metadata` as FAILED, because line 1 is
rewritten. That is expected. Do not use it as a gate after the fix.

### F5. Byte-level corpus check (BLOCKER on mismatch)

`assert_invariants.py` sums only `.ds.metadata` line 2 (Part B, D-F1), so check the bytes directly:

```bash
python - <<'EOF'
import glob, os, numpy as np
for s, exp, n in [('raw_diversity_oriented', 10_000_002_885, 16), ('raw_disagreement_aware', 10_000_005_069, 16),
                  ('signal-disagreement-lambda0', 10_000_002_556, 26)]:
    ds = sorted(glob.glob(f'<data_root>/{s}/tokenized/*.ds'))
    b = sum(os.path.getsize(f) // 2 for f in ds)
    m = sum(int(open(f + '.metadata').read().split('\n')[1]) for f in ds)
    last_ok = all(int(np.fromfile(f + '.index', dtype=np.uint64)[-1]) == os.path.getsize(f) // 2 for f in ds)
    print(s, len(ds), b, m, last_ok, 'OK' if (len(ds), b, m, last_ok) == (n, exp, exp, True) else 'MISMATCH')
EOF
```

Expected: three `OK` lines. Document counts from `.index` are 7,138,615 / 7,018,928 / 12,566,224.

### F6. Init identity (BLOCKER)

```bash
for s in 42 43 44; do python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed$s/0 --check <init_root>/init_1.5B_seed$s.hash.json; done
```

Expected rolling sha256 (identical to the Hub manifests at `87e1335`):
- 42: `2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba`
- 43: `78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3`
- 44: `a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247`

Each has 180 files. `plan_submit.py` checks only the file count, not the hash (T-F10).

### F7. Cluster entries

- Fill `marc-cluster` in `deploy/clusters.yaml`:
  - `micro_batch_size: 32`;
  - `recompute_layer: true` on 80 GB cards;
  - `slurm.time` ≥ `1-06:00:00` (interim B1 guard);
  - `repo_dir` = this clone (`plan_submit.py:100-103` refuses otherwise; T-F7).
- Only if one node has 8 GPUs, add `marc-cluster-8gpu`: a copy of `marc-cluster` with `gpus_per_node: 8`, `dp: 8`,
  `gres` for 8 GPUs and scaled `cpus_per_task`. Keep `micro_batch_size: 32`.
- WARN: record the GPU model. kys-eval requires H100 for scoring (Part E).

### F8. Render (BLOCKER on any failure)

```bash
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42 --settings $W1        # "wrote 18 filled configs"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 43 --settings $W1        # "wrote 18 filled configs"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings raw_diversity_oriented,raw_disagreement_aware   # "wrote 12"
# wave 2, dp 8 only:
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --out-dir configs/1.5B-baseline-seed44/filled-dp8                                            # "wrote 6"
```

Without dp 8, run instead
`fill_placeholders.py --cluster marc-cluster --seed 44 --settings signal-disagreement-lambda0` ("wrote 6").
Every file line should read `OK`.

### F9. Full invariant check on all filled configs (BLOCKER)

```bash
for c in configs/1.5B-baseline-seed4{2,3,4}/filled*/{raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0}_seed4?_*.yaml; do
  ( set -a; source ${c%.yaml}.env; python tools/assert_invariants.py --config $c ) >/dev/null 2>&1 && echo PASS || echo "FAIL $c"
done | sort | uniq -c
```

- Expected: `54 PASS` (dp 4), or 48 at dp 4 plus 6 in `filled-dp8/`.
- Tokens per step must be `2,097,152`. Accum must be 8 at dp 4 and 4 at dp 8.
- On this cluster the identical check passed on 54/54 at dp 4 and 6/6 at dp 8 (Part C).

### F10. Smoke (WARN thresholds; BLOCKER on crash or NaN)

- Run `tools/kys_raw/smoke_20steps.py` on one dp-4 trunk1 config (WORKFLOW §8), and on the `filled-dp8` trunk1 if
  dp 8 is used.
- Report `lm_loss` at steps 1/10/20 (reference 10.8 / 8.83 / 8.19), s/it and peak memory.
- Use the measured s/it in F11.

### F11. Dry-run submission (BLOCKER on any deviation)

```bash
for s in 42 43; do python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed $s --s-per-it <measured> --settings $W1 --test-only; done
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it <measured> --settings raw_diversity_oriented,raw_disagreement_aware --test-only
grep -c '=\$(sbatch' configs/1.5B-baseline-seed42/filled/submit_seed42.sh       # 18 (43: 18, 44: 12)
grep -lE 'raw_random|raw_rewire_inspired|raw_top10b' configs/1.5B-baseline-seed4?/filled*/submit_seed4?.sh    # must print nothing
squeue -u $USER -h -o %j | grep '^kys_'                                          # must print nothing before --submit
```

- **Always pass `--settings`.** Without it, `plan_submit.py` and `fill_placeholders.py` default to all eight settings
  (T-F3).
- `--test-only` must report every segment accepted.
- Only then rerun the same three commands with `--submit` (wave 1).
- Wave 2 goes in only when no wave-1 segment is waiting for a slot (WORKFLOW §1b).

### F12. Evaluation readiness (before scoring; details in Part E)

```bash
cd <kys-eval> && ./install.sh && source .venv/bin/activate && export HF_HOME=$PWD/hf_cache
python -m kys_eval.check_install      # must end "OK: environment matches the pinned v2 setup"
python -m kys_eval.selftest           # "selftest passed"
python -m kys_eval.prefetch
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203   # PASS on H100
export KYS_GRID_ARGS="--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0"
python -m kys_eval.run_grid --list --check-hub $KYS_GRID_ARGS     # "27 cells; SLURM: --array=0-26", all "present" after uploads
mkdir -p logs && sbatch --array=0-26%8 --export=ALL,KYS_EVAL_ROOT=$PWD slurm/eval_grid_array.sbatch
python -m kys_eval.run_grid --list $KYS_GRID_ARGS                 # all 27 "done"
python -m kys_eval.aggregate
```

- Indices 0–8 are seed 42, 9–17 seed 43 and 18–26 seed 44. Within each seed: `raw_diversity_oriented`,
  `raw_disagreement_aware`, `signal-disagreement-lambda0`, each ep1 / ep2 / ep3.
- **BLOCKER-level operational rule:**
  - `--array=0-26` must always go with `KYS_GRID_ARGS`. Without it, the array silently scores the wrong 27 cells of
    the 72-cell grid and misses 18 rebuttal cells (E-F2).
  - The array's default is `0-71`.
- **Prerequisites:**
  - The 27 checkpoints must first be uploaded to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`, which currently holds
    only `.gitattributes`. Upload them at `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf`.
  - Use a fresh results root. Use `--force` for any cell whose checkpoint is re-uploaded, because existing results
    are reused without checking which checkpoint they came from (E-F1).
  - Score only on H100 80GB HBM3 (SXM), the device of all 54 v2 results.

---

## Part A — verdict

**CONDITIONAL PASS. Not launch-ready until B1 is cleared.**

| criterion | result |
|---|---|
| PASS requires: experiment definition identical to v2 and the paper; data provenance verified; eval aligned; no blocker | definition, data and eval: **met** (Parts C, D, E). Blocker: **B1 open** |
| CONDITIONAL PASS: alignment met; remaining blockers are launcher or operational defects that can be fixed before GPU submission without changing the experiment | **this case** |
| FAIL: a confirmed mismatch that makes the nine runs incomparable with v2 | **none found** |
| INSUFFICIENT EVIDENCE: a requirement central to comparability that cannot be checked | **no**. The unverifiable items (Part G5) are secondary: v2 trunk configs, main-text paper wording, hardware on the collaborator's cluster |

**Conditions for launch:**
1. B1 is fixed in code, or the interim procedural guard of F0 is adopted in writing.
2. F1–F11 all pass on the collaborator's cluster.
3. For evaluation: the 27 checkpoints are uploaded, and the F12 commands are used verbatim.

## Part B — findings

Severity scale: Critical / High / Medium / Low / Note / Unverified. Evidence: code / artifact / documentation /
inferred. Prefixes:
- **T-** = training protocol and launcher (sub-audit, verified by the main session where marked);
- **D-** = data and documentation (main session);
- **E-** = evaluation (sub-audit).

Paths are relative to the nanotron clone unless prefixed with `kys-eval/`. Line numbers are at `0614206b` and
`c8aa8ff`.

| ID | file:lines | paper / protocol requirement | actual behavior | severity | evidence | required fix |
|---|---|---|---|---|---|---|
| T-F1 (**B1**) | `tools/generate_configs.py:175-176`; `src/nanotron/trainer.py:240-259, 1046-1058`; `tools/assert_invariants.py:378-391`; `deploy/slurm/kys_segment.sbatch:4` (`--requeue`) | "Requeue … a trunk resumes from its newest checkpoint, losing at most 1500 steps" (WORKFLOW §9, RUNBOOK step 7). Per-chain schedule as in Table 11 | A trunk1 restart after its step-1500 or 3000 save loads those weights, but restarts metadata (step 0), AdamW state, warmup and the data stream from sample 0. It still writes a normal 4292 checkpoint and exits 0. `--check-resume` passes. This is likely in practice: §7 allows `time` ≥ 12 h against a 27.2 h trunk1, and §9 calls resubmission safe. Flagged on skipjack 2026-10-27; still unfixed. Verified by the main session. | **High (launch blocker)** | code | trunk1 must fail closed when `latest.txt` ≠ 0, or load optimizer, LR and metadata when `latest.txt` > 0; correct WORKFLOW §7/§9 and RUNBOOK step 7. Interim: F0 guard |
| T-F3 | `tools/kys_raw/plan_submit.py:64`; `tools/kys_raw/fill_placeholders.py:44`; `tools/kys_raw/registry.py:55-58`; WORKFLOW §9 (lines ~515-523); RUNBOOK step 6 | Only the nine rebuttal chains may be queued | With `--settings` omitted, both tools default to all 8 settings (48 segments per seed). The §9 and RUNBOOK examples omit `--settings`, contradicting §1b. If all 8 filled configs exist, the 5 excluded settings are submitted. Otherwise the tool exits before writing, so it fails safe. | Medium | code + doc | make `--settings` required (or default to the rebuttal set); fix the examples |
| E-F1 | `kys-eval/kys_eval/eval_checkpoint.py:394-396, 438-440`; `run_grid.py:69, 82-94` | A cell is reused only if its results belong to that checkpoint | `is_complete` checks only the schema and the `complete` flag, not the cell, subfolder, HF commit or weights hash. A stale result is skipped silently and aggregated. | Medium | code | compare the stored cell, subfolder and commit before skipping. For the rebuttal: fresh results root, `--force` after any re-upload |
| E-F2 | `kys-eval/slurm/eval_grid_array.sbatch:9, 23` | Score exactly the 27 rebuttal cells | Default `--array=0-71` runs the full grid, and missing checkpoints fail loudly. `--array=0-26` without `KYS_GRID_ARGS` silently scores the wrong 27 cells. | Medium (operational) | code | the exact commands of F12 |
| E-F12 | `kys-eval/kys_eval/aggregate.py:185` (+ stale list, Part E) | The documentation agrees with the settled fact: v2 is the source of the ICLR tables | `aggregate.py` writes "These are not the paper's printed table values." into every generated `raw_vs_rewritten.md` | Medium | code | replace the sentence |
| D-F7 | paper §3 / Figure 1 (main text, not on this cluster; wording from the audit brief) | The 5M analysis sample was "excluded from all training-data construction" | False for every 1.5B corpus. Confirmed against `projects/rewrite/docs/analysis_sample_exclusion_check.md`: 205,623 sample documents in the shared anchor; 0 overlap for the 50,000 holdout and for the 50,427 scorer documents | Medium (paper accuracy; does not affect the runs) | artifact | correct §3 / Figure 1 wording (scorer-training exclusion only) |
| E-F11 | Hub `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` | 27 checkpoints scored from `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf` | The repo holds only `.gitattributes` (commit `10a1133b`) | Prerequisite (blocks eval, not training) | artifact | upload, then `run_grid --list --check-hub` |
| D-F1 | `tools/assert_invariants.py:78-79` (comment) vs `:245-282` (code); `tools/kys_raw/tokenize_raw_text.sh` post-check | Corpus identity checked on the data, not the metadata | Both sum `.ds.metadata` line 2 only; the comment's "cross-checked against raw .ds bytes/2" is not in the code. On this cluster bytes = metadata for all 58 shards of the three corpora (Part D). λ0 `.ds` files are additionally covered by `SHA256SUMS`. | Low | code + artifact | add a bytes/2 check, or keep F5 in the checklist |
| T-F7 | `deploy/slurm/kys_segment.sbatch:25` | The audited code is what runs | Nothing asserts that the imported `nanotron` is under `$REPO`. `repo_dir` is required and checked on `marc-cluster` (`plan_submit.py:100-103`). | Low | code | assert `nanotron.__file__` is under `$REPO` |
| T-F10 | `tools/kys_raw/plan_submit.py:122-135` | Init = the published per-seed init | Only the 180-file count is checked, not the shipped `*.hash.json` | Low | code | run `hash_init_checkpoint.py --check` (F6), optionally in the generated script |
| E-F4 | `kys-eval/README.md:29, 120, 134`; `reference_check.py:7`; `slurm/reference_check.sbatch:17`; `reference/expected…json:17` | Reference-check instructions point at an existing checkpoint | An old local path, and "not on the public Hub"; it is public at `wytro/KYS-1.5B-Rewritten-v2@b22c8733` | Low | artifact + doc | use the pinned `hf://` spec |
| E-F5 | `kys-eval/kys_eval/common.py:69-77`; `config.py:157-158` | Score on the v2 device | Non-H100 is refused unless `--allow-any-gpu` (then recorded and flagged). The substring match also accepts H100 NVL/PCIe; v2 used SXM. | Low (documented) | code | optional exact-name check |
| E-F6, E-F8 | `kys-eval/kys_eval/aggregate.py:116-133`; `install.sh:18-19` | Environment deviations are visible; the patched tree = base + patch | Only GPU mismatches are surfaced in aggregates; the "already patched" shortcut checks file hashes, not HEAD | Low | code | surface env flags; assert HEAD |
| D-F4 | `configs/1.5B-baseline/README.md:40, 53, 209-210` | The docs describe the current 8-setting layout | Still "42 experiment templates (7 settings × 6)", "all 126", "21 chains: 2,093 node-hours" | Low | doc | update the counts (48 / 144 / 24 chains / 2,392) |
| D-F2 | WORKFLOW §5 | Checksum before `fix_ds_metadata.py` | Correct order (sha256sum -c, then fix). Not stated: a later `sha256sum -c` fails on the 26 rewritten `.metadata` files | Low (doc) | doc + code | add a one-line note (F4 says it) |
| T-F2 / D-F6 | v2 `seed42/quality_base/ep{1,2,3}` config.yaml; Table 11 footnote | Same objective for all comparator runs | QB seed 42 ran mbs 16 / accum 16. Under per-micro-batch `masked_mean` that is a slightly different objective (~1e-3 relative). It affects only cells paired with QB seed 42 (none of the rebuttal comparisons, which use `diversity_oriented` / `disagreement_aware`). Documented in WORKFLOW §13 and RUNBOOK step 3. The footnote's "optimization therefore identical" overstates it. | Note (documented exception); footnote Low | artifact | footnote wording only |
| T-F5 | `src/nanotron/models/llama.py:747, 773, 781` | Recompute does not change the math | Reentrant checkpoint, RNG preserved, dropout 0. Cross-hardware results are fp-level only and documented in WORKFLOW §13 ("no bitwise reproducibility"). | Note | code | report recompute and the GPU type |
| T-F6 | `src/nanotron/serialize/optimizer.py:360-370`; `helpers.py:97-161` | Branches decay linearly to 0 from the branch point | A resumed branch uses λ(t): ep1 step 4293 at 475/476 of peak, step 4768 at 0. Same code path as v2. | Note | code + simulation | none |
| T-F8 | `tools/kys_raw/fill_placeholders.py:67` vs `tools/render_config.py:175-185` | mbs uniform across cluster entries | The cross-entry mbs check never sees two entries; uniformity rests on `assert_invariants` pinning mbs 32 | Note | code | keep mbs 32 in `marc-cluster-8gpu` |
| T-F4 | `src/nanotron/serialize/main.py:229-235`; `tools/assert_invariants.py:357-408`; `kys_segment.sbatch:21, 50, 57` | A wrong resume path must not silently random-init | Guarded before torchrun under `set -euo pipefail` (missing path, wrong branch step, missing `latest.txt` or `model_config.json`). The remaining gap is B1. | Note (guarded) | code | via B1 |
| D-F3 | λ0 `shuffled/` vs replay (Part D) | Exactly one document-level shuffle | Reproduced exactly with inputs ordered `rewritten/` then `shared-top-5B/`, B = 16. `doc_id` repeats for dual-rewrite sources, so the order of the two prompt rows of one source is not independently pinned. | Note | artifact | none |
| D-F5 | registry `expected_corpus`; RUNBOOK λ0 section | λ0 total = 10,000,002,556 | Two empty `wikipedia` rewrites (`doc_id` 79663246, 84276725) have no datatrove document, so the total is 2 below the 10,000,002,558 budget. Documented. | Note (documented exception) | artifact | none |
| E-F7, E-F9, E-F10 | kys-eval (see `agent_eval/FINDINGS.md`) | – | `hash_full_prompts` wording; unrecorded auto batch size (same as v2 on H100 80GB); `--hf-revision` defaults to main (the resolved commit is recorded) | Note | code | wording / pin after upload |
| T-F9 | – | Trunk-only fields match v2 | v2 publishes only ep configs | Unverified | – | – |

**Hypotheses from the brief, classified:**

| hypothesis | classification |
|---|---|
| QB seed 42 at mbs 16 | documented exception (T-F2); no rebuttal cell affected |
| recompute / cross-hardware numerics | harmless (math) and documented (fp-level) (T-F5) |
| equal-token raw controls | documented and verified exactly (Part D) |
| Quality-Base as the top-ranked counterpart of Quality-First | documented (WORKFLOW §2, line 260); not part of the rebuttal set |
| greedy decoding without a seed | **not applicable**: all 63 tasks are log-likelihood (Part E) |
| wave 1 running seeds concurrently | harmless for comparability: chains share nothing but read-only inputs, the data order is fixed, and the seed selects only the init. Documented (§1b) |
| λ0 two tokens short | documented exception (D-F5) |
| `plan_submit.py` defaults to 8 settings | **real issue** (T-F3) |
| `assert_invariants.py` trusts `.ds.metadata` | **real issue, Low** (D-F1); data verified here |
| wrong resume path → random init | guarded (T-F4), **except** the trunk1 case, which is a **real issue** (B1) |
| `run_grid.py` defaults to 72 | harmless (E-F3) |
| SLURM array defaults to 0-71 | real operational issue (E-F2) |
| `eval_checkpoint.py` skip without an identity check | **real issue** (E-F1) |
| stale reference-check instructions | real, Low (E-F4) |
| non-H100 GPU | documented and guarded (E-F5) |
| stale v2 claims | real (E-F12 and the Part E list) |
| four- or seven-setting layout docs | real, Low (D-F4); kys-eval clean (E-F13) |

## Part C — template comparison (54 templates vs Table 11 vs v2)

**Method:**
- All 54 templates (`configs/1.5B-baseline-seed{42,43,44}/templates/{raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0}_seed<S>_{trunk1,trunk2,trunk3,ep1,ep2,ep3}.yaml`)
  were diffed within each segment kind. **Only `general.seed` and `data_stages[0].data.seed` differ, and both equal
  the chain's seed.**
- The shipped placeholder configs equal their templates on every key.
- Rendered ep configs were parsed with this repo's `nanotron.config.get_config_from_file`, so defaults are filled in.
  They were diffed key by key against the matching v2 `config.yaml` (`wytro/KYS-1.5B-Rewritten-v2@b22c8733`,
  `diversity_oriented` and `disagreement_aware`, seeds 42/43/44, ep1–ep3). After normalizing run names and paths,
  the only differences are the runtime counters `general.step` and `general.consumed_train_samples`.
- The dp-8 render differs from dp 4 only in `parallelism.dp: 8` and `batch_accumulation_per_replica: 4`.
- **Rendered against this cluster's real data:** the full `assert_invariants.py` (corpus, tokenizer metadata, batch,
  env, placeholders) passes on **54/54** configs at dp 4 / mbs 32 / accum 8, and **6/6** λ0 seed-44 configs at dp 8 /
  mbs 32 / accum 4. The rendered copies are in `prelaunch_audit_2026-11-07/data/rendered_configs/`.
- Scripts: `agent_train/cmp.py`, `flat.py`, `lrsim.py`, `samp.py`.

| field | Table 11 | templates (54) | v2 | verdict |
|---|---|---|---|---|
| layers / hidden / FFN | 28 / 2048 / 5632 SwiGLU | 28 / 2048 / 5632, silu | same | ✔ |
| heads / kv heads | 16, MHA | 16 / 16 | same | ✔ |
| vocab / tied embeddings | 32000 / yes | 32000 / true | same | ✔ |
| RoPE θ / RMSNorm ε | 10⁴ / 10⁻⁵ | 10000.0 / 1e-5 | same | ✔ |
| AdamW β / eps | (0.9, 0.95) | (0.9, 0.95) / 1e-8, fused | same | ✔ |
| weight decay / clip | 0.1 / 1.0 | 0.1 / 1.0 | same | ✔ |
| peak LR / warmup | 5e-4 / 500 linear | 5e-4 / 500 linear | same | ✔ |
| trunk ends | 4292 / 8583 / 12875 | `train_steps` 4292 / 8583 / 12875; constant LR to 12875 | not published | ✔ (v2 side unverified) |
| branches | end 4768 / 9537 / 14305, linear to 0 | decay start 4292 / 8583 / 12875, 476 / 954 / 1430 steps, min LR 0 | same | ✔ |
| branch resume point | the trunk step at the decay start | `<trunk>/<lr_decay_starting_step>` (`render_config.py:320-322`), enforced by `assert_invariants.py:394-399` | same | ✔ |
| seq / global batch / tokens per step | 2048 / 1024 / 2.10M | 2048; 32×8×4 = 1024 → **2,097,152** (dp 8: 32×4×8) | 32×8×4 (QB s42 16×16×4) | ✔ |
| precision | bf16 params; fp32 grads and optimizer | bfloat16; `accumulate_grad_in_fp32: true`; zero 0 | same | ✔ |
| init | per seed, shared across settings | the same `<init_root>/_init_1.5B_seed<S>/0` copied into every setting's trunk (`plan_submit.py:121-135`) | – | ✔ by construction |
| data order at training | fixed, no shuffle | `shuffle_files` false (`src/nanotron/config/config.py:165`); the data seed is used only for file shuffling (`tokenized_bytes.py:252-256`) | same | ✔ (seed has no effect) |
| recompute | – | from the cluster entry (true on 80 GB) | false | ✔ (no change to the math, T-F5) |
| trunk1 load flags | – | `load_optimizer` / `load_lr_scheduler` false; all later segments true | ep: true | ✔ on a clean run; **B1 on restart** |

**dp 8 / mbs 32 / accum 4 vs dp 4 / mbs 32 / accum 8: per-token weighting is preserved exactly, not only the global
batch.**
- `MegatronPretrainingSampler` (`src/nanotron/data/samplers.py:188-224`) cuts each step's 1024 consecutive samples
  into blocks of mbs·dp and gives rank *d* the slice [32d, 32(d+1)). Both layouts therefore form the same 32
  contiguous 32-sequence micro-batches per step. This was enumerated at consumed = 0, 4292·1024 and 8583·1024.
- Each micro-batch's loss is a `masked_mean` over its own tokens (`llama.py:984-1006`), divided by accum
  (`pipeline_parallel/engine.py:53-55`).
- Gradients are accumulated in fp32 (`gradient_accumulator.py`), then AVG all-reduced over dp (`helpers.py:459-461`).
- Every micro-batch carries weight 1/32 in both layouts; only the fp summation order differs. The 1/4, 1/8 and ÷dp
  factors are powers of two.

## Part D — data provenance

Every check below is reproducible from `prelaunch_audit_2026-11-07/data/`: the scripts plus `*.json` results.
- Local corpora: `/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/` (the KYS-Pre-Rewritten copy)
  and `/weka/scratch/jhu/bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0/`.
- Arms are read from `wytro/Know-Your-Sources@9e5ff241`.

| check | raw_diversity_oriented | raw_disagreement_aware | signal-disagreement-lambda0 |
|---|---|---|---|
| tokens = Σ `.ds` bytes / 2 | **10,000,002,885** | **10,000,005,069** | **10,000,002,556** |
| Σ `.ds.metadata` line 2 | equal | equal | equal |
| shards; last `.index` offset = bytes/2 | 16; ✔ | 16; ✔ | 26; ✔ |
| documents (`.index`) | 7,138,615 | 7,018,928 | 12,566,224 |
| every doc ends in `</s>` (id 2) / starts with `<s>` (id 1) | all / 0 | all / 0 | all / 0 |
| re-tokenized sample = `.ds` (text, `add_special_tokens=False`, + id 2) | 3,000 / 3,000 (shards 0, 7, 15) | 3,000 / 3,000 | 6,000 / 6,000 (shards 0, 13, 25; rerun today) |
| shard *i* = parquet file *i*, row order kept | ✔ | ✔ | ✔ (26 tasks, 26 files) |
| anchor rows = shared 5B anchor (4,120,164 docs, 5,000,002,332 tokens) | ✔ | ✔ | ✔ |
| strategy half | 3,018,451 docs / 5,000,000,553 tokens | 2,898,764 docs / 5,000,002,737 tokens | 8,446,062 rewrite rows / 5,000,000,226 tokens |
| relation to the rewritten arm | ⊂ the 5,867,876 unique sources of `diversity_oriented`'s rewritten half (51.4%) | ⊂ the 5,592,424 unique sources of `disagreement_aware`'s rewritten half (51.8%) | the λ = 0 selection's rewrites |
| seed-42 subsampling (`build_raw_sources.py:186-193`: `default_rng(42).permutation`, shortest prefix ≥ 5B) replayed | **identical** | **identical** | n/a |
| exactly one upstream shuffle: `pp_io.bucketed_shuffle(seed=42)` replayed on IDs | **identical** full order (B = 16, 200 inputs) | **identical** (B = 16) | **identical** (B = 16; inputs `rewritten/` then `shared-top-5B/`) |
| no shuffle at tokenization | `preprocess_data_parquet.py:82` `shuffle_documents=False`; 16 tasks = 16 files | same | 26 = 26 |
| no shuffle in training; same order every epoch | `shuffle_files: false`; sequential sampler (Part C) | same | same |
| tokenizer | `tokenizer.json` sha256 `81bb383c…d8e5` | same | same |

The raw controls are **equal-token, not identical-document**: S is a seed-42 uniform random 5B of the arm's
unique sources. Both are documented as such in the RUNBOOK "Purpose" section and the manifest `role`.

**λ = 0 vs λ = 0.5** (`projects/rewrite/06_lambda_grid/select_lambda_grid.py`, artifact `lambda_grid.npz`):
- **Shared construction.** The candidate domain U (each scorer's top-10% by tokens over the pool minus the holdout
  minus the shared top 5B), the floor τ_q = Q30(q over U) = 0.6956847 and the cap τ_v = Q90(v over U) = 0.1036492 are
  computed once, before the λ loop (`:101-118`). Only `u = q + λ·√v` changes per λ (`:137`). The script also requires
  a byte-exact rebuild of the saved λ = 0 / 0.5 / 1 id sets before writing (`:151-161`).
- **Artifact checks:**
  - The npz q and v recompute exactly from the pool's three `*-ranking-v2` columns.
  - All 10,435,667 survivors satisfy the floor and the cap.
  - The λ = 0 corpus's rewrite sources (5,532,945) lie within the λ = 0 selection, and the published
    `disagreement_aware` sources (5,592,424) within the λ = 0.5 selection. Selection = `ranks <= cutoffs`, the code's
    own definition at `:184`.
  - 9,717 (λ0) and 10,052 (λ0.5) selected documents have no rewrite row: failed rewrites, dropped the same way in both.
  - The anchor is identical in both.
- **Composition.** λ0 has 5,527,293 `wikipedia` + 2,918,769 `distill` rewrite rows. The published λ0.5 has 5,587,114
  `wikipedia` + 2,855,159 `distill`. That is the same two-prompt construction, with counts that follow the selection.
- **Shortfall.** 10,000,002,556 = the assembly budget 10,000,002,558 − 2. The only `train_tokens == 0` rows are two
  empty `wikipedia` rewrites, `doc_id` 79663246 (`part_00020`, row 173739) and 84276725 (`part_00023`, row 336529);
  datatrove writes no document for an empty text.

**Checksum order.** WORKFLOW §5 runs `sha256sum -c SHA256SUMS` before `fix_ds_metadata.py` ✔. A later re-check fails
on the 26 rewritten `.metadata` files; that is expected (D-F2).

**Init identity.**
- The `wytro/Know-Your-Sources-init@87e1335` manifests give rolling sha256 42 `2ede6612…3aba`, 43 `78ab44e2…a3`,
  44 `a967df1c…247`, identical to RUNBOOK lines 44-46.
- 180 files per seed; `model_config.json` identical across the seeds.
- The local seed-42 manifest is byte-identical to the Hub's.
- The weights themselves were not re-hashed here: the local seed-42 copy holds only `optimizer/`, and 43/44 were not
  downloaded. The collaborator must run F6.

**5M analysis sample (settled; numbers confirmed).** `projects/rewrite/docs/analysis_sample_exclusion_check.md`
reports:
- **205,623** sample documents in the shared anchor;
- **0** overlap for the 50,000 validation holdout and for the 50,427 ModernBERT scorer documents, in every corpus.

The paper's §3 / Figure 1 wording ("excluded from all training-data construction") **must be corrected** (D-F7). It
does not affect the runs: the comparators contain the sample at the same rate.

## Part E — evaluation alignment (kys-eval @ c8aa8ff)

Evidence: `prelaunch_audit_2026-11-07/agent_eval/`, with `FINDINGS.md`, comparison outputs and scripts. The paper
tables map by label, by inference: Table 12 = `tab:per-task-epoch3`, 13 = `tab:mean6-by-epoch`, 14 =
`tab:mmlu-by-epoch`, 15 = `09_appendixF_mmlu_category.tex`; main Table 3 = `08_table3_mmlu_category.tex`.

| requirement | code | verdict |
|---|---|---|
| ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA; Mean6 = unweighted mean | `config.py:82-89, 129-133`; `common.py:232-234` | ✔. Every printed Mean6 cell, std and k/3 count reproduced through kys-eval's own aggregation |
| MMLU = unweighted macro over 57 subjects (not over categories); categories 18 / 13 / 12 / 14 | `common.py:235-240`; `config.py:99-125` | ✔. Every `tab:mmlu-by-epoch`, Table 3 and Appendix F cell reproduced; one documented double-rounding exception (ep2 QB STEM 26.87 → "26.9") |
| 0-shot; full splits; no `max_samples` | `config.py:74`; `common.py:251-255`; `eval_checkpoint.py:402-408` | ✔. Split sizes 2376 / 10042 / 1838 / 1954 / 500 / 1221, MMLU 14042; v2 used 0-shot in all 3402 task configs |
| `acc_norm` = log-likelihood ÷ continuation tokens | patch `LoglikelihoodAcc(LogProbTokenNorm())`; `check_install.py:410-413` | ✔ |
| task definitions | `rewrite_tasks.py` sha256 `a87d708d…` = the original `projects/lighteval/community_tasks/rewrite_tasks.py`; `kys_tasks.py` only pins `hf_revision` and aborts on a hash mismatch | ✔ |
| dataset revisions | 7 pins (`config.py:64-72`) = the v2-era cache snapshots (v2 itself ran unpinned; documented) | ✔ |
| LightEval 10b9104 + patch | `third_party` diff = `patches/lighteval-10b9104-kys-v2.patch`; the 4 patched-file sha256 = the original v2 tree; enforced at `eval_checkpoint.py:443-449` | ✔ |
| reference checkpoint and hashes | expected JSON = the v2 `diversity-first-10B-1.5B-seed42/14305` JSON (64 tasks, 508 fields, 0 differences); the checkpoint is public at `wytro/KYS-1.5B-Rewritten-v2@b22c8733` `rewrite-1p5b/seed42/diversity_oriented/ep3/hf` (`model.safetensors` sha256 `733262ae…` matches) | ✔ |
| comparator table | `reference/rewritten_v2_54_per_task.csv`: all 6912 acc / acc_norm cells = the 54 v2 JSONs; all scored on H100 80GB HBM3 | ✔ |
| decoding | all 63 tasks are log-likelihood; no generation, so no decoding seed | ✔ (n/a) |
| λ0 ↔ λ0.5 pairing | `config.py:232-235`: `signal-disagreement-lambda0` → `disagreement_aware` (v2 stem `signal-disagreement-lambda05`) = nanotron `registry.py:37-38` | ✔ |
| 72-cell grid; 27-cell rebuttal subset | `run_grid --list --settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0` → 27 cells, `--array=0-26`, indices 0-8 / 9-17 / 18-26 = seeds 42 / 43 / 44 | ✔ (`agent_eval/list27.txt`) |
| target repo | `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` holds only `.gitattributes` | prerequisite (E-F11) |

**Operational risks:** E-F1, a skipped cell is reused without an identity check, and E-F2, the array/args pairing.
Both are handled by F12.

**Stale "v2 ≠ the paper's printed tables" references.** These contradict the settled fact; full quotes are in
`agent_eval/FINDINGS.md`.
- **HF `wytro/KYS-1.5B-Rewritten-v2` README.md** (b22c8733 = main): lines 35, 37, 37-39, 39-40, 40-41. The section
  "Not the paper-era checkpoint", "These are **not** the checkpoints behind the numbers printed in the paper's
  tables", the paper-era steps 4770 / 9540 / 14305 / 14147, "differ … by up to about 1 accuracy point", and "use the
  `blab-jhu/KYS-1.5B-*` repos to reproduce the paper's tables".
- **kys-eval README.md:** 10, 10-11, 11-13 ("v2 numbers are not the numbers printed in the paper's tables" …), and
  134-135 ("not on the public Hub"; false).
- **kys_eval/config.py:** 8, 8-10, 10-11 (module docstring).
- **Also:** `kys_eval/aggregate.py:185`, which is emitted into every generated report, and
  `reference/expected_diversity_oriented_seed42_ep3.json:17`.

## Part G — direct answers

**G1. Is there a confirmed mismatch that makes the nine runs incomparable with v2? No.**
- **Configs:** the 54 templates equal the v2 configs and Table 11 on every experimental field.
- **Parallelism:** dp 8 / accum 4 forms the same micro-batches with the same per-token weighting.
- **Data:** the three corpora match their recorded totals at the byte level. Their construction (anchor,
  subsampling, single shuffle, EOS convention, λ-only change) was replayed exactly.
- **Evaluation:** the protocol reproduces every printed cell of the ICLR appendix tables.
- **Caveat:** this holds for a clean run; a restarted trunk1 would break it (G2).

**G2. Silent-failure risks:**
- **Training: B1 (T-F1).** A restarted trunk1 silently restarts the schedule and the data stream on trained weights.
  This is the only one that corrupts results.
- **Launcher: T-F3.** Omitting `--settings` plans the five excluded settings; it fails safe only if their configs
  were not filled.
- **Resumption:** otherwise guarded (T-F4).
- **Evaluation:**
  - E-F1: reused results are not tied to their checkpoint;
  - E-F2: `--array=0-26` without `KYS_GRID_ARGS` scores the wrong cells;
  - E-F12: `aggregate.py` prints a stale v2 disclaimer into every report.

**G3. Fix before GPU submission:**
- **B1:** a code fix, or the written interim guard in F0.
- **T-F3:** recommended, so `--settings` cannot be forgotten; at minimum, the F11 grep gate.
- **Prerequisites:** pass F1–F11 on the target cluster, including F5 (bytes) and F6 (init hashes).

**G4. Needs documentation only:**
- WORKFLOW §7 / §9 and RUNBOOK step 7: requeue claims (with B1);
- WORKFLOW §9 and RUNBOOK step 6 examples: add `--settings`;
- `configs/1.5B-baseline/README.md` counts (D-F4);
- the post-fix checksum note (D-F2);
- the stale v2 statements: model card, kys-eval README, `config.py`, `aggregate.py:185` (string),
  `reference/…json`;
- the reference-check instructions (E-F4);
- the Table 11 footnote "optimization therefore identical" (T-F2);
- the paper's §3 / Figure 1 analysis-sample wording (D-F7);
- `assert_invariants.py:78-79`, which claims a byte check the code does not do (D-F1).

**G5. Unverifiable from here:**
- **Paper main text:** §3 and §5.1 / 5.3 / 5.4 are not on this cluster; only the August PDF is.
- **Table numbering:** 11–15 mapped by label.
- **v2 trunk-segment configs:** not published.
- **v2 code and torch versions:** beyond the config equality.
- **Init weights:** the weights of seeds 42–44 against their manifests; only the manifests were compared here.
- **The collaborator's cluster:** GPU type; SLURM requeue / timeout / afterok behavior; wall limit; s/it at dp 8.
- **GPU-dependent eval checks:** the reference check on a new node; `model_cache` races.
- **Empty target repo:** whether uploads land at the expected paths, since `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`
  is empty.

---
*Evidence index:*
- `prelaunch_audit_2026-11-07/data/`: byte and metadata check, re-tokenization, subsampling replay, shuffle replay,
  λ checks, init manifests and the rendered configs;
- `agent_train/`: config diff, sampler and LR simulations, v2 configs, init metadata;
- `agent_eval/`: eval comparisons and the stale-reference evidence.

*Inline checks:* the byte check and the raw re-tokenization were run inline; their code is `F5` above and
`data/retokenize_lambda0.py`, adapted. Their results are `data/ds_vs_metadata.json` and `data/retokenize_check.json`.

*Correction:* `data/lambda0_check.json` reports two "subset" checks as false. That was an off-by-one in the audit
script, which used `ranks < cutoffs`; the code's definition is `ranks <= cutoffs`. `data/lambda_idspace_check.json`
and the follow-up check in Part D supersede it.

---

## Addendum (2026-11-07, later) — B1 fixed on `huicong-dev` @ `ebb7607a`

The parts above are unchanged and describe `0614206b`. This addendum records the fix for B1 and what it supersedes.

**Fix** (`ebb7607a`, "kys: trunk1 restart resumes its full state"). trunk1 now **loads its full saved state on a
restart**:
- `tools/kys_raw/trunk_restart_config.py <cfg>` prints the config to run. For a weights-only trunk config
  (trunk1) whose `latest.txt` is > 0, it first writes `<trunk dir>.restart/<run>_from<step>.yaml`, a copy that
  differs **only** in `load_optimizer: true` and `load_lr_scheduler: true` (asserted in the tool). The step must
  have a `model_config.json` and be below `train_steps`, otherwise the tool exits 1.
- `deploy/slurm/kys_segment.sbatch` runs `assert_invariants.py --check-resume` and `torchrun` on that config.
- `assert_invariants.py --check-resume` **refuses** a trunk config with either flag false whose `latest.txt`
  ≠ 0. A manual launch of the original config therefore fails closed.
- WORKFLOW §7 / §9 and RUNBOOK step 7 now say what restores state and why.
- Why this is sound:
  - the restart copy takes the same load path that trunk2 uses from trunk1's 4292 save;
  - the step-1500 and 3000 interval saves and the final save both call `Trainer.save_checkpoint()`
    (`src/nanotron/trainer.py:604, 609`);
  - with `load_lr_scheduler` true, the step and consumed-sample counters come from the checkpoint
    (`trainer.py`, "Define iteration start state"), and `run_train.py:343` resumes the data stream from them.
- No template, shipped config or rendered config changed. `git diff 0614206b ebb7607a -- tools deploy configs`
  touches only the five files of the fix.

**Verification on this cluster** (`prelaunch_audit_2026-11-07/b1_fix/`). Configs were rendered with
`fill_placeholders.py` against the local corpora, using a `marc-cluster` profile with local paths (`profile.yaml`;
recompute on, `slurm.time` 1-06:00:00).

| check | result |
|---|---|
| full `assert_invariants.py` (corpus, tokenizer metadata, batch, env, placeholders) | **54/54** at dp 4 / mbs 32 / accum 8; **6/6** λ0 seed 44 at dp 8 / mbs 32 / accum 4. Totals 10,000,002,885 / 10,000,005,069 / 10,000,002,556; 2,097,152 tokens per step |
| `--check-resume` on a clean start of every segment (mock trunk directories) | **60/60** pass; the helper returns the rendered config unchanged for all 60 |
| trunk1 restarted at `latest.txt = 1500`, original config | **10/10 refused** by the new guard |
| trunk1 restarted at 1500, the helper's copy | **10/10 pass** the full check plus `--check-resume`. Parsed with nanotron's `get_config_from_file`, the copy equals the original except for the two load flags |
| helper edge cases | `latest.txt` ≥ `train_steps` or non-numeric → exit 1; ep configs and trunk2/3 → unchanged |

The script is `b1_fix/resume_matrix.sh`; per-config logs are in `b1_fix/assert_logs/{dp4,dp8}/`. Two restart
copies are kept as samples in `b1_fix/restart_configs/`. The checkpoint directories were mocks (empty step
folders), so no GPU step was run. **On the collaborator's cluster, the first restarted trunk1 is still the real
test** (see below).

**What changes in Part F:**
- **F0 / B1: cleared in code.** The interim procedural guard is no longer required. It remains harmless:
  `slurm.time` ≥ 1-06:00:00 still avoids trunk restarts.
- **F1:** expect `huicong-dev` at the commit that adds this addendum or later. `git diff 0614206b -- tools deploy
  configs` must show only the files of `ebb7607a`.
- **After any trunk1 restart:** the log shows `[restart-config] … latest.txt = <step> > 0 -> loading optimizer,
  LR and metadata`. The first logged iteration is then `<step> + 1`, not 1, and the LR is flat at 5e-4 (past
  warmup). Report a trunk1 restart that logs iteration 1 as a BLOCKER.
- **Part A verdict:** the one listed blocker is cleared. The remaining launch conditions are F1–F11 on the target
  cluster, and for evaluation the 27 uploads plus the F12 commands. The other findings (T-F3, E-F1, E-F2, E-F12,
  D-F1 …) are unchanged and still open.

**Not covered by this fix:** the main clone `nanotron-kys` (branch `skipjack-1p5b-baselines`), whose queued
raw-baseline jobs submit their own copy of `kys_segment.sbatch`. It still has the pre-fix behaviour.

**Evidence housekeeping:**
- `agent_eval/scripts/hf.py` and `small.py` now resolve `agent_eval/hfdl/` from their own location instead of the
  audit node's deleted `/tmp` scratchpad. `small.py` was rerun from another directory: the 4 files are re-verified,
  and the folder contents are unchanged.
- The rendered configs in `data/rendered_configs/` still name the audit-time `/tmp` checkpoint and data paths.
  They are a record of that render, not inputs.

### Addendum 2 (2026-11-08) — T-F3 fixed

`--settings` is now required by `tools/kys_raw/fill_placeholders.py` and `tools/kys_raw/plan_submit.py`, and
`registry.parse_settings` no longer maps an empty value to every setting. All eight settings are planned only for the
explicit value `all`, which the rebuttal set never uses. The WORKFLOW §6 and §9, RUNBOOK steps 2 and 6, README and
`deploy/clusters.yaml` examples now pass `--settings`. Checked locally with `b1_fix/profile.yaml`: both tools refuse a
missing flag (argparse, exit 2), an empty value and an unknown name; seed 42 with the three wave-1 settings renders
and plans 18 segments (assert_invariants passes on each); `--settings all` renders and plans 48. Nothing was submitted.
The F11 grep gate in Part F is no longer needed for T-F3, but stays harmless.

### Addendum 3 (2026-11-08) — E-F1, E-F2, E-F12 fixed in kys-eval; D-F4 fixed

- kys-eval `main` @ `216c1ba876b6702d7000a1395b31e7f76a87ded4` (required revision in WORKFLOW and RUNBOOK is now
  `216c1ba` or later):
  - E-F1: a cell is skipped only if its complete `results.json` has the same model path (HF repo + subfolder or
    local dir), the pinned revision when one is given, and the same setting / seed / epoch / step / subfolder /
    repo. Otherwise it is re-evaluated; `run_grid --list` marks it `stale`. A re-upload to the same path and
    revision is still not detectable: keep `--force` for that case (F12).
  - E-F2: `eval_grid_array.sbatch` exits 2 without `KYS_GRID_ARGS` (`all` = 72 cells), and each task exits 2
    unless `--array` is exactly `0..N-1` for the N selected cells (`run_grid --expect-cells`);
    `KYS_ARRAY_PARTIAL=1` allows re-running single indices. Default array `0-26%8`. The F12 commands still work
    unchanged, since they already export `KYS_GRID_ARGS`.
  - E-F12: `aggregate.py` now writes that the v2 checkpoints are the source of the paper's ICLR tables. The kys-eval
    README's own stale v2 sentence (line 10, Part E list) was not changed.
  - `python -m kys_eval.selftest` passes, with new identity and array-size cases; the sbatch guard was exercised with
    stubbed `python` / `nvidia-smi` for: no args (refused), rebuttal 0-26 (runs, `--expect-cells 27`), `all` 0-71,
    `--array=7-12` (refused; runs with `KYS_ARRAY_PARTIAL=1`), and full grid with a 27-task array (refused).
- D-F4: `configs/1.5B-baseline/README.md` now says 48 templates / 48 rendered configs per seed, 144 identical after
  renaming (the 18 λ0 configs checked against `raw_random`'s, 18/18), 8 × 99.7 = 798 node-hours per seed, 24 chains
  = 2,392 node-hours, rebuttal set 897.
