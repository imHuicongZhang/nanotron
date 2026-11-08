# Handoff, round 2: final independent review of the KYS rebuttal launch set (for GPT-6)

You reviewed this set once. Your findings were repaired in a second round, described here. Review again
independently: re-derive what you can, try to falsify the claims below, and say where you agree, where you
disagree and what you could not check. Do not propose new experiments, do not rewrite the collaborator's workflow,
and report findings rather than pushing changes that alter training.

## 1. What the rebuttal needs (unchanged)

- **The paper.** "Know Your Sources: Data Selection Matters when Rewriting for Data-Constrained Pretraining" (ICLR 2027
  submission). It has 54 "v2" checkpoints: six settings × seeds 42/43/44 × steps 4768/9537/14305. They are 1.5B models
  trained on 10B-token mixtures (a shared 5B fastText anchor plus a 5B strategy half) repeated three times. They are
  published as `wytro/KYS-1.5B-Rewritten-v2@b22c8733`, and they are both the comparator and the source of the paper's
  tables.
- **Two controls, nine chains, 27 checkpoints:**
  1. `raw_diversity_oriented` (vs `diversity_oriented`) and `raw_disagreement_aware` (vs `disagreement_aware`):
     equal-token-budget raw controls.
  2. `signal-disagreement-lambda0` (vs `disagreement_aware`, λ = 0.5): the same feasible domain, ranked with λ = 0.
- **Outputs needed.** Per seed and epoch: acc_norm (LightEval, 0-shot, full splits) on the six commonsense tasks,
  Mean6, MMLU macro over 57 subjects and the four category means. Then paired within-seed differences, setting minus
  comparator, against the primary comparator and against `quality_base`: mean ± SD (ddof 1) over three seeds with the
  number of positive seeds (Appendix F format).

## 2. Repositories and revisions

| repo | revision the collaborator may hold | current | changes since |
|---|---|---|---|
| `github.com/imHuicongZhang/nanotron` `huicong-dev` | `6fe10b9b` | `git log -1` (docs only after `91ae7f16`) | `f70e1662` audit report; `dd621e92` addendum §11 + identity test; `91ae7f16` WORKFLOW/RUNBOOK wording, deviations, greedy-decoding sentence, copy of the paired analysis; then the addendum §12 + this handoff. **No file under `tools/`, `deploy/`, `src/`, `configs/1.5B-baseline-seed*/` changed; only `docs/` and `configs/1.5B-baseline/{WORKFLOW_RAW_BASELINES,RUNBOOK}.md`.** |
| `github.com/imHuicongZhang/kys-eval` `main` | `216c1ba` | `6da264bf` | `1c191acf` README v2 note; `668a9dd3` aggregate pairs every setting also with `quality_base`, Δ = setting − comparator, CSV columns `comparator_*` + `comparison`; `6da264bf` full-precision v2 reference (Issue B) and wording. **No change to `common.py`, `eval_checkpoint.py`, `run_grid.py`, `reference_check.py`, `tasks/`, `patches/`, `slurm/`.** |
| `projects/rewrite` (local, **no remote**) | — | `c639080` | `ecf7d74` paired analysis as of 2026-11-06; `c639080` `--kys-results` loader. Remote copy: nanotron `docs/rebuttal_q1_paired_analysis/` (byte-identical, `SOURCE.md`). |

**Data.**
- `blab-jhu/KYS-Pre-Rewritten@afd1f177` (public) holds the raw text, the λ0 tokenized shards and the tokenizer.
- `wytro/Know-Your-Sources-init@87e13356` is public now; it was gated in round 1.
- `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` is gated and empty. It is the upload target and the scoring source.
- `wytro/Know-Your-Sources@9e5ff24` (public) holds the published v2 training corpora; it was used for Issue C.

**Records.**
- `docs/prelaunch_audit_2026-11-07.md` is the first audit, with addenda B1 / T-F3 / E-F1 / E-F2 / E-F12 / D-F4.
- `docs/prelaunch_audit_2026-11-08.md` is the second audit. It covers §0–§10, addendum §11 (analysis gaps) and
  addendum §12 (this round).
- Scripts, logs and evidence are under `docs/prelaunch_audit_2026-11-08/`. `work/` (large downloads) is not committed.

## 3. What changed this round, and the claims to test

**Issue B: positive-seed counts near zero (kys-eval `6da264bf`).**
- *Claim:* the v2 comparator CSV is rounded to 10 decimals (642/3,456 values bit-identical; max error 5.0e-11).
  Pairing it with full-precision new cells turned exact ties into ±1e-11 and miscounted seeds (78/108 primary rows in
  the round-1 identity test).
- *Fix:* `reference/rewritten_v2_54_per_task.json` holds the values at full precision, with each source sha256. It is
  built by `reference/build_rewritten_v2_reference.py`, which also checks that the CSV is its `.10f` rounding.
  `aggregate.load_rewritten` reads only this file and refuses the CSV. No tolerance is applied, on the argument that
  LightEval, kys-eval and the builder all serialize floats as shortest round-trip repr, and `benchmark_scores` sums in
  a fixed order.
- *Tests:*
  - `selftest.test_pairing_statistics`: JSON round trips; CSV hazard and refusal; mixed signs; exact ties; partial
    ties; mean, ddof-1 SD and positive-seed counts on 12 benchmarks.
  - `test_aggregation`: primary Δ = 0.0 exactly.
  - The audit's `issue_b/check_reference_json.py` shows the JSON equals the 54 v2 JSONs bit for bit, with 54/54
    sha256 matches.
  - The strict identity test `analysis_gaps/test_analysis_gaps.py` passes 16/16, with 0/108 positive primary rows.
- *Try to falsify:*
  - Is any code path still reading the CSV for pairing?
  - Can `benchmark_scores` order differ between the two sides, for example dict order of `tasks` in a real
    LightEval-produced `results.json` vs the reference?
  - Would an `acc_norm` that LightEval writes as an int, NaN or None break the "no tolerance" argument?
  - Does `json.dumps(indent=1)` ever change a float?
  - Are the expected values in `test_pairing_statistics` truly independent of the code under test?

**Issue C: λ0 vs λ0.5 prompt identity (audit §12, `prompt_identity/`).**
- *Records:* nothing usable.
  - The prompt files are untracked, and their mtimes postdate both runs.
  - The logs hold no prompt hashes or token counts.
  - The per-worker input-token totals are over different document sets.
  - `input_tokens_qwen` does not exist in the published corpus at `9e5ff24`.
- *Direct check:* all of published `disagreement_aware/part_00000` matched to local λ0 on (doc_id, source_prompt).

  | comparison | wikipedia exact | distill exact |
  |---|---:|---:|
  | λ0 vs λ0.5 | 51.9% (n = 197,689) | 45.9% (n = 95,794) |
  | same-prompt control, published `diversity_oriented` vs λ0.5 | 55.7% (5,498) | 49.9% (2,126) |
  | different-prompt control, wikipedia vs distill of the same document | 0.0% (3,811), 48% diverging at char 0 | — |

- *Pattern:* the rate falls with output length in the same way in the question and the control. Mismatches diverge
  partway through (median 43–47% of the text). Early divergences are near-synonym swaps. Per-shard rates span 27–74%;
  per-worker rates are flat.
- *Conclusion recorded:* same prompts, established from the outputs. Byte identity of the prompt files is not
  recoverable.
- *Try to falsify:*
  - Could a prompt edit that leaves most outputs unchanged (whitespace, a trailing newline, a word in a sentence the
    model ignores) produce exactly this picture?
  - Is the 1–7-point gap to the control within length bins explained by node/GPU differences, by sample size, or by
    something else?
  - Is the shard-level spread (|z| up to 14) consistent with batch nondeterminism?
  - Are the doc_id spaces really aligned between the published corpus and local λ0? Both use the scored-pool
    `doc_id`; `orig_doc_id` is the 100M-reservoir position.
  - Is part_00000 representative?

**Issue A and Item D.**
- *Issue A:* each new setting is also paired against `quality_base`, with the Appendix F sign. This is in kys-eval
  `668a9dd3` and the paired analysis `c639080`, with the identity test above.
- *Item D:* the paired analysis is backed up in nanotron `docs/rebuttal_q1_paired_analysis/`. Note the repo is public
  and that folder's README holds draft rebuttal text.
- *Try to falsify:*
  - Does `paired_analysis.py --kys-results` reject every malformed input? The tested cases are missing, wrong step,
    wrong model subfolder, incomplete, misplaced and duplicate.
  - Does it produce byte-identical v2-only outputs without the flag?
  - Do its LOTO and 2d outputs treat the new pairs correctly?

**Documentation this round.**
- WORKFLOW §0C: access, with the inputs public and the raw-selected repo gated with write needed for `blab-jhu`.
- WORKFLOW §1b: `--settings` is required.
- WORKFLOW §13: rebuttal deviations, namely λ0 seed 44 at dp 8 and training hardware.
- RUNBOOK λ0 section: greedy decoding.
- kys-eval `config.py`, README and reference JSON note: no "v2 is not the paper's tables" claims; the reference
  checkpoint is public.
- *Check:*
  - Is anything still wrong?
  - Is the stated v2 hardware ("the authors' cluster, dp 4 per checkpoint metadata") accurate?
- Left alone: the HF model card of `wytro/KYS-1.5B-Rewritten-v2`, still stale and pending authorization, and the
  manuscript.

## 4. Earlier claims worth re-testing

- **Protocol identity.** The new ep configs differ from the v2 configs of their comparators only in `recompute_layer`.
  Against `quality_base` seed 42 they also differ in mbs 16 / accum 16. Trunk/branch schedule 4292/8583/12875 →
  4768/9537/14305.
- **Data totals.** 10,000,002,885 / 10,000,005,069 / 10,000,002,556 tokens. Raw parquet matches the manifest sha256.
  λ0 shards match `SHA256SUMS`.
- **λ0 provenance.** Every λ0 rewritten source is in the λ = 0 selection over U_τ (floor Q30(q), cap Q90(v)), and
  `u_score` = q.
- **Export.** `tools/kys_eval/convert_to_hf.py` reproduces the released v2 `model.safetensors` byte for byte in both
  transformers 4.46.3 and 5.6.2.
- **Restart safety.** A requeued trunk1 must load optimizer and LR state or refuse (B1).
- **Launch scope.** `--settings` is required by `fill_placeholders.py` and `plan_submit.py`. The SLURM eval array
  refuses to run without `KYS_GRID_ARGS` and refuses an `--array` range that doesn't match the selection.

## 5. Not checked (needs Marc's cluster or a GPU)

- **Reference check** on the scoring H100:
  `python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203`.
  - Pass: weights sha256 `733262ae…3f45`; the four LightEval hashes equal for all 64 tasks; per-task within 0.005,
    Mean6/MMLU within 0.001; exact match expected on H100 80GB HBM3.
- Install, smoke and full invariants on the collaborator's data.
- The first real trunk1 restart.
- Gated-repo access on the scoring nodes.
- The 27-cell evaluation, `aggregate.py` and `paired_analysis.py --kys-results` on real results.

## 6. Deliverable

Produce a short report covering:
- per rebuttal goal, whether the code and data produce exactly what is needed, naming the file;
- a findings table with location, expected, found, severity, and whether it changes a number or only wording;
- your verdict on each falsification question above;
- what you could not access.
