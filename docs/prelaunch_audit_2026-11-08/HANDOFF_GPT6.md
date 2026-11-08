# Handoff: final independent review of the KYS rebuttal launch set (for GPT-6)

You are doing the final, independent review before a collaborator trains nine 1.5B chains for an ICLR 2027
rebuttal. Two audits precede you (2026-11-07 and 2026-11-08). Do not take their conclusions on trust: re-derive
what you can, and say where you agree, where you disagree and what you could not check. Do not propose new
experiments and do not rewrite the collaborator's workflow documents; report findings only.

## What the rebuttal needs

The paper ("Know Your Sources: Data Selection Matters when Rewriting for Data-Constrained Pretraining", ICLR 2027
submission) trains 54 checkpoints: six data settings × seeds 42/43/44 × three epoch checkpoints (steps 4768, 9537,
14305), 1.5B parameters, 10B-token mixtures (a shared 5B fastText anchor + a 5B strategy-specific half) repeated
three times, WSD schedule with a trunk and three decay branches. These are the "v2" checkpoints; they are both the
comparator and the source of the paper's tables.

Reviewers asked for two controls, answered by three new settings at seeds 42, 43, 44 (9 chains, 27 checkpoints):
1. Non-rewritten counterparts at matched token budget: `raw_diversity_oriented` (vs `diversity_oriented`) and
   `raw_disagreement_aware` (vs `disagreement_aware`). Equal-token-budget, not identical-document: each raw half is a
   seed-42 uniform random 5B of the sources of the counterpart's rewritten half; same anchor.
2. Disagreement-Aware with λ = 0 (`signal-disagreement-lambda0`), same candidate pool, quality floor and variance
   cap as the published λ = 0.5 arm (`disagreement_aware`), to isolate the disagreement bonus.

The rebuttal must compute, per new setting, per seed and epoch: acc_norm (LightEval, 0-shot, full splits) on ARC-Easy,
HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA, their unweighted mean (Mean6), MMLU macro over 57 subjects and the
four category means; then paired within-seed differences against `disagreement_aware`, `diversity_oriented` and
`quality_base` as mean ± SD (ddof 1) over three seeds with the number of positive seeds, Appendix F format.

## Code and data (all public on GitHub / Hugging Face unless noted)

- Training: `github.com/imHuicongZhang/nanotron`, branch `huicong-dev`. Revision the collaborator uses:
  `6fe10b9b130a22338bfc1c2d199761131b711a88`; the audit commit on top of it changes only `docs/`
  (`git log -1 -- docs/prelaunch_audit_2026-11-08.md`).
  Key files: `tools/kys_raw/registry.py`, `configs/1.5B-baseline-seed{42,43,44}/`, `tools/kys_raw/fill_placeholders.py`,
  `tools/kys_raw/plan_submit.py`, `tools/assert_invariants.py`, `tools/kys_raw/trunk_restart_config.py`,
  `deploy/slurm/kys_segment.sbatch`, `tools/kys_raw/tokenize_raw_text.sh`, `tools/fix_ds_metadata.py`,
  `tools/kys_eval/convert_to_hf.py`, `configs/1.5B-baseline/{RUNBOOK.md,WORKFLOW_RAW_BASELINES.md,README.md}`.
- Evaluation: `github.com/imHuicongZhang/kys-eval`, branch `main` @ `1c191acfba7f035258f541aa389ffa186898908d`
  (code = `216c1ba`; one README commit after). Key files: `kys_eval/{config,common,eval_checkpoint,run_grid,aggregate,reference_check}.py`,
  `slurm/eval_grid_array.sbatch`, `reference/rewritten_v2_54_per_task.csv`, `reference/expected_diversity_oriented_seed42_ep3.json`.
- Data: `huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten` @ `afd1f177` (public): `manifest.json`,
  `raw_text/raw_diversity_oriented/` and `raw_text/raw_disagreement_aware/` (16 parquet each),
  `signal-disagreement-lambda0/tokenized/` (26 datatrove shards + `SHA256SUMS`), `tokenizer/`.
- Init: `wytro/Know-Your-Sources-init` @ `87e13356` (gated, auto-approve). Comparators: `wytro/KYS-1.5B-Rewritten-v2`
  @ `b22c8733` (public). Destination: `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` (gated, empty).

## Prior audit records (read these first, then verify)

- `docs/prelaunch_audit_2026-11-07.md` (+ folder): conditional pass; addenda 1–3 record fixes B1 (trunk1 restart
  state), T-F3 (`--settings` required), E-F1/E-F2/E-F12 (kys-eval result identity, SLURM array guard, v2 wording),
  D-F4 (README counts).
- `docs/prelaunch_audit_2026-11-08.md` (+ `docs/prelaunch_audit_2026-11-08/`): this second audit. Sections:
  §2 data verification, §3 config identity vs v2, §4 output side, §1/§5 goal mapping, pairing, HF export round trip,
  §6 tests, §7 gap table (G-N*, G-O*, G-D*), §8 deviations. Scripts in `scripts/`, logs in `logs/`, evidence in
  `evidence/`. `work/` (large downloads) is not committed.

## What to check independently (suggested order)

1. **Identity of the protocol.** Diff each new setting's ep configs against the v2 `nanotron/config.yaml` of its
   comparator on the Hub (and of `quality_base`). Expected: only `parallelism.recompute_layer` (speed only) and, for
   Quality-Base seed 42, mbs 16 / accum 16 (a known deviation). Check that the trunk/branch schedule
   (trunk 4292/8583/12875, branches 4768/9537/14305) and the LR schedule match v2.
2. **Data.** sha256 / row counts of the two raw-text folders vs `manifest.json`; λ0 shards vs `SHA256SUMS`; totals
   10,000,002,885 / 10,000,005,069 / 10,000,002,556 tokens. If you can, tokenize one raw shard with
   `tokenize_raw_text.sh` and compare with the expected per-shard totals.
3. **λ0 construction.** Confirm from `select_lambda_grid.py` / `lambda_grid.npz` (if available to you) or from the
   documented manifests that λ0 ranks the same feasible domain (floor Q30(q), cap Q90(v)) and that its rewritten
   sources are the λ = 0 selection. Note the two id spaces: `doc_id` (scored pool) vs `orig_doc_id` (100M reservoir).
4. **Restart safety.** `trunk_restart_config.py` + `kys_segment.sbatch` + `assert_invariants.py --check-resume`:
   a requeued trunk1 must load optimizer and LR state, or refuse; a restarted trunk1 must log step+1, not 1.
5. **Output path.** `results.json` schema (`common.build_results`), `aggregate.py` pairing and statistics, and whether
   anything is aggregated before the per-seed pairing. Re-run the selftest. Check the sign convention against
   Appendix F (aggregate uses comparator − new).
6. **Upload and scoring layout.** `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf` in the destination repo vs
   `config.hf_subfolder`; the reference check's criteria (weights hash, 64 tasks' LightEval hashes, tolerances).
7. **Gaps in §7 of the 2026-11-08 audit.** For each, agree/disagree and give the severity you would assign.
   Especially G-O1 (no Quality-Base pairing for the new settings), G-O3 (the existing paired-analysis script reads
   v2 LightEval JSONs, not kys-eval's `results.json`), G-D1 (two gated repos documented as public), G-N1 (λ0 seed 44
   at dp 8).
8. **Anything the two audits missed.** In particular: data-loader behaviour with a different number of shards than
   v2 (16 or 26 vs v2's), `fix_ds_metadata.py`'s effect on the tokenizer string nanotron checks, wandb/env leakage of
   tokens into configs, and whether `plan_submit.py` can still submit a setting outside the rebuttal set.

## Constraints

- Do not push changes that alter training behaviour (configs, rendered configs, data loading, schedules, resume,
  SLURM launch, anything under `configs/`, `tools/kys_raw/`, `deploy/slurm/`). Report them instead, with files,
  lines, old vs new behaviour, effect on comparability with v2, tests, and the commit the collaborator would need.
- Do not overwrite published Hugging Face artifacts, do not force-push, do not put credentials in any file.
- If you have no GPU, do not claim the reference check passed; state its command and criteria.

## Deliverable

A short report: per rebuttal goal, whether the code and data produce exactly what is needed (name the file); a
table of findings (location, expected, found, severity, number-changing or wording-only); your view on each open
item from the 2026-11-08 audit; and what you could not access.
