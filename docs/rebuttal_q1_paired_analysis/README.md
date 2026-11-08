# Rebuttal Q1: paired comparisons among all six settings

**Reviewer question:** which strategy comparisons are still supported by paired uncertainty
estimates and leave-one-benchmark-out analyses? Appendix F compares each strategy only with
Quality-Base. This directory extends that analysis to all 15 pairs of settings and adds
per-seed ranks. It was computed on CPU from the existing LightEval outputs. Nothing was
re-evaluated, and no checkpoint or evaluation file was modified.

Re-run end to end with `python3 paired_analysis.py`. It uses the standard library only. The
script exits non-zero if any of the 54 result sets is missing or if Step 1 does not reproduce
the paper. Its console output is saved in `run_output.txt`.

## 1. Inputs

**Result files.** There are 54 raw LightEval JSONs from the v2 evaluation. Each cell has
exactly one results file.

```
/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/08_evaluation/04_results_1p5_v2/downstream/<run>/<step>/results/**/results_*.json
```

`/home/jhu/zhuicon1/scratch_bvandur1/...` is the same filesystem: `scratch_bvandur1` resolves to
`/weka/scratch/jhu/bvandur1`. All 54 files were produced with LightEval `0228edc` and scored on
an NVIDIA H100 80GB HBM3 (SXM). The paper setting names map to the run directory names as
follows:

| Paper name | Checkpoint dir (`rewrite-1p5b/<seed>/…`) | Run dir in `downstream/` |
|---|---|---|
| QUALITY-BASE | `quality_base` | `quality-base-10B-1.5B-seed*` |
| QUALITY-FIRST | `quality_first` | `quality-first-10B-1.5B-seed*` |
| DIVERSITY-ORIENTED | `diversity_oriented` | `diversity-first-10B-1.5B-seed*` |
| WRAP-INSPIRED | `wrap_inspired` | `wrap-10B-1.5B-seed*` |
| REWIRE-INSPIRED | `rewire_inspired` | `rewrite-10B-1.5B-seed*` |
| DISAGREEMENT-AWARE | `disagreement_aware` | `signal-disagreement-lambda05-10B-1.5B-seed*` |

**These are the branched-schedule checkpoints.** The script checks that each JSON's
`model_name` is `/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/<run>/<step>`.
`08_evaluation/02_code/ckpt_list_v2.txt` records that each of those HF directories was
converted from the annealed branch endpoint
`checkpoints/rewrite-1p5b/<seed>/<variant>/ep{1,2,3}/<variant>/<seed>/{4768,9537,14305}`.
No trunk checkpoint or v1/paper-era result was used. The v1 results under
`08_evaluation/03_results/` were not read, and neither were the `_nvl_original/` and
`_sxm_recheck/` copies. The `kys-eval` (50bb288) and `lighteval` projects contain no results
for these 54 checkpoints; their few results JSONs are for other models.

The full list of files used is in `result_files_used.csv` (absolute path, HF model path,
nanotron source checkpoint, LightEval sha, GPU) and in the appendix at the end of this file.

**Paper LaTeX used for the cross-check.** `paper-writing/05_paper/` contains only the
2026-08-24 PDF and no LaTeX. That PDF carries the older, pre-branched numbers (for example,
QUALITY-BASE Mean6 at epoch 1 is 41.04). The revised appendix that matches the reference values
is:

- `/weka/scratch/jhu/bvandur1/zhuicon1/paper-writing/06_iclr/appendix_training_and_per_task.tex`
  (`tab:per-task-epoch3`, `tab:mean6-by-epoch`, `tab:mmlu-by-epoch`, which I take to be Tables 12–14)
- `/weka/scratch/jhu/bvandur1/zhuicon1/paper-writing/06_iclr/09_appendixF_mmlu_category.tex`
  (MMLU by category at epoch 3)

## 2. Definitions

- **Metric:** `acc_norm`, the LightEval token-length-normalized log-likelihood accuracy
  (`LogProbTokenNorm`). All numbers are in accuracy points.
- **Mean6:** the unweighted mean of ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA and
  CommonsenseQA (`rw_arc:easy`, `rw_hellaswag`, `rw_piqa`, `rw_siqa`, `rw_openbookqa`,
  `rw_commonsense_qa`).
- **MMLU:** the unweighted macro-average over the 57 `rw_mmlu:<subject>` subsets.
  LightEval's `_average` entry is not used. Each category score (STEM 18, Humanities 13,
  Social Sciences 12, Other 14 subjects) is the unweighted mean over its subjects. The mapping
  is `08_evaluation/02_code/mmlu_categories.py`, the same one behind the paper's category
  tables.
- **Paired difference:** for settings A and B at an epoch, d_s = A_s − B_s for each seed
  s ∈ {42, 43, 44}. The six settings at a seed share the same initialization. Results are
  reported as mean ± SD of the three d_s (sample SD, ddof = 1, the paper's convention), plus the
  number of seeds with d_s > 0. No t-tests and no p-values.
- **Rank:** within each (epoch, seed), the six settings are ranked by score, with 1 as best.
  There were no exact ties.
- **Leave-one-task-out (LOTO):** the five-task mean with one task removed, then paired as
  above. A variant is flagged when the sign of its mean differs from the sign of the full
  Mean6 Δ.
- **Classification (Step 2d):** each comparison is oriented so that the winner is listed first.
  A pair where A loses to B in all three seeds counts as "B > A in 3/3".
  - *Supported:* 3/3 seeds and |mean| > SD.
  - *Consistent only:* 3/3 seeds but |mean| ≤ SD.
  - *Unsupported:* everything else.

## 3. Step 1 verification: PASS

All **135** reference cells were reproduced, and the largest deviation from a printed value is
0.0050 points, which is rounding. The cells are:

- 132 parsed directly from the two LaTeX files: per-task epoch 3; Mean6 and MMLU by epoch,
  absolute and Δ with seed counts; MMLU category Δ with seed counts.
- 3 quoted in the request.

All seed counts match exactly. See `step1_verification.md` and `step1_verification.csv`.

| Reference (from the request) | Reproduced |
|---|---|
| QUALITY-BASE Mean6 ep 1/2/3: 41.75 ± 0.67, 44.80 ± 0.07, 46.27 ± 0.67 | identical |
| QUALITY-BASE MMLU ep 3: 30.59 ± 0.03 | identical |
| DIVERSITY-ORIENTED Mean6 Δ: +0.76 ± 0.82 (3/3), +0.60 ± 0.31 (3/3), +0.14 ± 0.86 (2/3) | identical |
| DISAGREEMENT-AWARE MMLU Δ ep 2/3: +0.60 ± 0.28 (3/3), +0.49 ± 0.42 (3/3) | identical |
| DIVERSITY-ORIENTED LOTO ep 3 ranges −0.14 to +0.42; negative without HellaSwag or PIQA | identical (−0.14 without PIQA, −0.10 without HellaSwag, +0.42 without CSQA) |

**Note for the revision.** The revised appendix LaTeX in `06_iclr/` has no LOTO table. The
LOTO table in the older `04_appendix/appendix_f_per_task_results.tex` (`tab:loto-diversity`)
still prints the pre-branched values (+0.43 to +0.81 at epoch 3, positive in all 21 variants).
Its prose says the margin "stays positive in all 21 variants", which the current checkpoints
contradict (see §4c). If that table or prose is still in the revised paper, it needs
replacing. `loto_mean6.csv` has the current values.

## 4. Results

### Files

| File | Content |
|---|---|
| `per_checkpoint_scores.csv` | 54 rows: six tasks, Mean6, MMLU macro and the four categories (base table) |
| `pairwise_mean6_mmlu.csv` | 2a: 15 pairs × 3 epochs × {Mean6, MMLU}: mean, SD, seeds A > B, the three per-seed differences |
| `pairwise_per_task_ep3.csv` | 2a: epoch 3, 15 pairs × (6 tasks + 4 MMLU categories). The four requested pairs have `requested_pair = 1` |
| `ranks_per_seed.csv` | 2b: rank per seed, mean rank, whether the rank is the same in all seeds |
| `loto_mean6.csv` | 2c: 5 strategies vs QB, plus DO vs DA, × 3 epochs × {none, 6 dropped tasks} |
| `summary_supported_3of3_mean_gt_sd.csv` | 2d, table 1 |
| `summary_consistent_3of3_mean_le_sd.csv` | 2d, table 2 |
| `tables.tex` | LaTeX: pairwise Mean6, pairwise MMLU, rank matrices (Mean6, MMLU), in the Appendix F style |

### 4a. Pairwise Mean6 and MMLU

The complete tables are in `pairwise_mean6_mmlu.csv` and `tables.tex`. Highlights:

- **DIVERSITY-ORIENTED vs DISAGREEMENT-AWARE.**
  - Mean6: DO is ahead at every epoch in 3/3 seeds: +0.40 ± 0.35, +0.58 ± 0.28, +0.48 ± 0.10.
  - MMLU at epoch 3: DA is ahead, +0.26 ± 0.27 (3/3, mean < SD).
- **DIVERSITY-ORIENTED vs QUALITY-FIRST.**
  - Mean6 at epochs 2 and 3: DO ahead, +0.44 ± 0.40 and +0.35 ± 0.26 (3/3).
  - MMLU at epoch 1: QF ahead (+0.33 ± 0.30, 3/3).
  - MMLU at epochs 2 and 3: unsupported.
- **DISAGREEMENT-AWARE vs QUALITY-FIRST.**
  - MMLU at epoch 3: DA ahead, +0.44 ± 0.28 (3/3).
  - Mean6 at epoch 2: QF ahead, +0.14 ± 0.12 (3/3). One seed's margin is +0.003 points, which
    is below a single test example, so treat this one as a tie.
- **QUALITY-FIRST vs REWIRE-INSPIRED:** unsupported on Mean6 and on MMLU at every epoch.

### 4b. Per-seed ranks

The rank matrices are in `ranks_per_seed.csv` and `tables.tex`.

Mean rank on **Mean6** (1 = best):

| Setting | ep1 | ep2 | ep3 |
|---|---|---|---|
| Quality Base | 4.33 | 3.67 | 3.00 |
| Quality First | 2.67 | 3.33 | 3.33 |
| Diversity Oriented | **1.67** | **1.00** (1,1,1) | **1.33** (1,1,2) |
| WRAP Inspired | 5.33 | 5.00 | 5.00 |
| REWIRE Inspired | 3.00 | 3.33 | 4.33 |
| Disagreement Aware | 4.00 | 4.67 | 4.00 (4,4,4) |

Mean rank on **MMLU**:

| Setting | ep1 | ep2 | ep3 |
|---|---|---|---|
| Quality Base | 3.67 | 4.33 | 4.00 (4,4,4) |
| Quality First | **2.00** | 2.33 | 3.33 |
| Diversity Oriented | 3.33 | 2.67 | 2.67 |
| WRAP Inspired | 5.33 | 6.00 (6,6,6) | 6.00 (6,6,6) |
| REWIRE Inspired | 3.67 | 4.00 | 3.67 |
| Disagreement Aware | 3.00 | **1.67** | **1.33** (1,2,1) |

Only five (setting, epoch, metric) cells hold the same rank in all three seeds:

- DO is 1st on Mean6 at epoch 2.
- DA is 4th on Mean6 at epoch 3.
- WRAP is 6th on MMLU at epochs 2 and 3.
- QB is 4th on MMLU at epoch 3.

Quality-Base itself ranks anywhere from 1st to 6th on Mean6 at epoch 3 (ranks 6, 2, 1).

### 4c. Leave-one-task-out

All values are in `loto_mean6.csv`. Of the 108 LOTO variants (15 strategy-vs-QB cells plus 3
DO-vs-DA cells = 18 cells, × 6 dropped tasks), 6 change sign relative to the full Mean6 Δ:

| Comparison | Epoch | Dropped | LOTO Δ | Full Δ |
|---|---|---|---|---|
| DO − QB | 3 | HellaSwag | −0.10 ± 1.07 (2/3) | +0.14 ± 0.86 (2/3) |
| DO − QB | 3 | PIQA | −0.14 ± 0.95 (2/3) | +0.14 ± 0.86 (2/3) |
| WRAP − QB | 2 | ARC-e | +0.05 ± 0.33 (1/3) | −0.18 ± 0.39 (1/3) |
| REWIRE − QB | 2 | HellaSwag | −0.00 ± 0.32 (1/3) | +0.14 ± 0.27 (2/3) |
| DA − QB | 2 | ARC-e | −0.10 ± 0.49 (1/3) | +0.02 ± 0.40 (1/3) |
| DA − QB | 2 | OBQA | −0.03 ± 0.46 (1/3) | +0.02 ± 0.40 (1/3) |

Every sign flip falls on a comparison that was already unsupported (≤ 2/3 seeds). The
comparisons that are positive in 3/3 seeds behave as follows:

- **DO − QB, epochs 1–2:** stays positive in all 12 variants.
- **REWIRE − QB, epoch 1:** stays positive in all 6 variants.
- **DO − DA:** stays positive in all 18 variants, ranging from +0.14 to +0.77. It drops to 2/3
  (or 1/3) seeds in four of them:
  - without HellaSwag at every epoch: 1/3, 2/3, 2/3;
  - without OBQA at epoch 1: 2/3.

  HellaSwag carries the largest share of DO's lead (DO − DA on HellaSwag at epoch 3:
  +2.22 ± 0.25).

### 4d. Summary classification

There were 240 comparisons in total:

- 15 pairs × 3 epochs × 2 families = 90;
- plus 15 pairs × 10 benchmarks at epoch 3 = 150.

Of these, **103 are supported** (3/3 seeds, mean > SD) and **8 are consistent only**
(3/3 seeds, mean ≤ SD). The other 129 are unsupported.

**Table 1: supported (3/3 seeds, mean Δ > SD).** Each row is winner > loser: mean ± SD
[per-seed differences for seeds 42, 43, 44].

*Commonsense Mean6*

| Ep | Comparison | Δ |
|---|---|---|
| 1 | DO > WRAP | +0.53 ± 0.45 [1.06, 0.30, 0.25] |
| 1 | DO > DA | +0.40 ± 0.35 [0.80, 0.15, 0.24] |
| 1 | DA > WRAP | +0.14 ± 0.12 [0.26, 0.14, 0.01] † |
| 2 | QF > DA | +0.14 ± 0.12 [0.21, 0.00, 0.19] † |
| 2 | DO > QB | +0.60 ± 0.31 [0.85, 0.69, 0.25] |
| 2 | DO > QF | +0.44 ± 0.40 [0.16, 0.90, 0.27] |
| 2 | DO > WRAP | +0.78 ± 0.40 [1.21, 0.42, 0.70] |
| 2 | DO > REWIRE | +0.46 ± 0.31 [0.81, 0.25, 0.32] |
| 2 | DO > DA | +0.58 ± 0.28 [0.37, 0.90, 0.46] |
| 2 | REWIRE > WRAP | +0.32 ± 0.13 [0.40, 0.17, 0.39] |
| 3 | QF > WRAP | +0.25 ± 0.13 [0.30, 0.10, 0.34] |
| 3 | DO > QF | +0.35 ± 0.26 [0.15, 0.64, 0.26] |
| 3 | DO > WRAP | +0.60 ± 0.14 [0.46, 0.74, 0.60] |
| 3 | DO > REWIRE | +0.53 ± 0.23 [0.79, 0.40, 0.39] |
| 3 | DO > DA | +0.48 ± 0.10 [0.53, 0.55, 0.37] |

*MMLU macro*

| Ep | Comparison | Δ |
|---|---|---|
| 1 | QF > DO | +0.33 ± 0.30 [0.09, 0.23, 0.66] |
| 1 | QF > WRAP | +0.82 ± 0.32 [1.18, 0.56, 0.71] |
| 1 | DA > WRAP | +0.62 ± 0.26 [0.87, 0.65, 0.35] |
| 2 | QB > WRAP | +0.59 ± 0.25 [0.31, 0.65, 0.80] |
| 2 | QF > QB | +0.48 ± 0.38 [0.49, 0.09, 0.84] |
| 2 | QF > WRAP | +1.06 ± 0.50 [0.80, 0.75, 1.65] |
| 2 | DO > QB | +0.40 ± 0.16 [0.46, 0.21, 0.52] |
| 2 | DO > WRAP | +0.98 ± 0.30 [0.76, 0.87, 1.32] |
| 2 | REWIRE > WRAP | +0.60 ± 0.48 [0.06, 0.99, 0.76] |
| 2 | DA > QB | +0.60 ± 0.28 [0.92, 0.40, 0.47] |
| 2 | DA > WRAP | +1.18 ± 0.11 [1.23, 1.06, 1.27] |
| 2 | DA > REWIRE | +0.58 ± 0.55 [1.17, 0.06, 0.51] |
| 3 | QB > WRAP | +0.87 ± 0.42 [0.42, 1.23, 0.97] |
| 3 | QF > WRAP | +0.92 ± 0.21 [0.74, 0.86, 1.16] |
| 3 | DO > QB | +0.23 ± 0.15 [0.40, 0.14, 0.15] |
| 3 | DO > WRAP | +1.10 ± 0.28 [0.82, 1.37, 1.12] |
| 3 | REWIRE > WRAP | +0.87 ± 0.81 [0.20, 1.77, 0.63] |
| 3 | DA > QB | +0.49 ± 0.42 [0.97, 0.19, 0.31] |
| 3 | DA > QF | +0.44 ± 0.28 [0.64, 0.56, 0.12] |
| 3 | DA > WRAP | +1.36 ± 0.07 [1.39, 1.42, 1.28] |

*Commonsense tasks (epoch 3), 38 comparisons; MMLU categories (epoch 3), 30 comparisons.*
These are listed in full in `summary_supported_3of3_mean_gt_sd.csv` and `run_output.txt`. The
recurring patterns are:

- **HellaSwag:** DO > every other setting (+0.86 to +2.22).
- **PIQA:** every strategy > QB, and DO > every other strategy.
- **OBQA:** QB, QF, DO and DA > REWIRE.
- **CSQA:** QF and REWIRE > DO.
- **SIQA:** QB > WRAP and DA.
- **MMLU categories:** WRAP is beaten by every setting in most categories. QB > QF and DO on
  STEM. DA > QB and QF on Humanities. DO > QB on Humanities and Social Sciences. QF > QB and DO
  on Other. DA > QB and DO on Other.

† The smallest per-seed margin is under 0.01 points, which is less than one test example in the
six-task mean. The CSVs carry a `smallest_seed_margin_4dp` column for this check.

**Table 2: consistent only (3/3 seeds, mean Δ ≤ SD).**

| Family | Ep | Comparison | Δ |
|---|---|---|---|
| Commonsense Mean6 | 1 | QF > WRAP | +0.44 ± 0.48 [0.36, 0.96, 0.01] |
| Commonsense Mean6 | 1 | DO > QB | +0.76 ± 0.82 [0.53, 0.07, 1.67] |
| Commonsense Mean6 | 1 | REWIRE > QB | +0.54 ± 0.54 [0.14, 0.32, 1.16] (0.539 vs 0.542) |
| MMLU macro | 1 | DO > WRAP | +0.49 ± 0.54 [1.09, 0.33, 0.05] |
| MMLU macro | 3 | DA > DO | +0.26 ± 0.27 [0.57, 0.05, 0.16] |
| CSQA | 3 | QF > DA | +0.41 ± 0.57 [1.06, 0.08, 0.08] |
| CSQA | 3 | DA > DO | +0.71 ± 0.80 [0.25, 1.64, 0.25] |
| MMLU Social Sci. | 3 | DA > QF | +0.21 ± 0.25 [0.07, 0.50, 0.07] |

## 5. (omitted from this public copy)

The draft rebuttal text and its caveats are kept only in the canonical copy in `projects/rewrite`.

## 6. Adding the three rebuttal settings (kys-eval results)

Added 2026-11-08. Once the collaborator returns the kys-eval results of the 27 rebuttal checkpoints
(`raw_diversity_oriented`, `raw_disagreement_aware`, `signal-disagreement-lambda0` × seeds 42/43/44 × ep1–ep3):

```
python3 paired_analysis.py --kys-results <dir with their results.json files> --out-dir <new dir>
```

- **Input.** kys-eval `results.json` files (schema `kys-eval/results/v1`) anywhere below the directory: kys-eval's
  own `results/raw_selected/seed<S>/<setting>/ep<N>/` tree, or a tree laid out like the checkpoints,
  `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/results.json`. Each file must name one checkpoint consistently: the stored
  `cell` (setting, seed, epoch, step 4768/9537/14305, `hf_subfolder` = `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf`), the
  stored model spec/subfolder, and the file's path if it contains such a segment. All 27 must be present, complete
  (63 tasks, no protocol violation) and unique, otherwise the script lists the problems and exits 1.
- **Scores** are computed from the stored per-task `acc_norm` with the same arithmetic as for the v2 files.
- **Pairs added** (setting minus comparator, the Appendix F orientation): Raw Diversity Oriented − Diversity
  Oriented, Raw Disagreement Aware − Disagreement Aware, Disagreement Aware (λ = 0) − Disagreement Aware, and each of
  the three minus Quality Base. They appear as a third block in `tables.tex`, and as extra rows in
  `pairwise_mean6_mmlu.csv`, `pairwise_per_task_ep3.csv`, `loto_mean6.csv`/`loto_summary.md`, the two 2d summary
  CSVs, `per_checkpoint_scores.csv` and `result_files_used.csv` (model spec, resolved HF commit, GPU).
- **Unchanged:** Step 1 and the per-seed ranks use the six v2 settings only. Without `--kys-results` every output is
  byte-identical to the 2026-11-06 version (checked).
- **Identity test** (2026-11-08): the 27 cells were filled with their comparator's v2 results through kys-eval's
  own packaging code; every rebuttal score equals the comparator's bit for bit, the 15 v2 pairs are unchanged,
  setting − comparator is 0.00 everywhere, setting − Quality Base equals comparator − Quality Base field for field,
  and missing, wrong-step, wrong-model, incomplete, misplaced and duplicate files are refused. Script and log:
  nanotron `docs/prelaunch_audit_2026-11-08/analysis_gaps/`.
- The Quality-Base seed-42 checkpoints used micro-batch 16 (all others 32); keep that footnote on the Quality-Base pairs.

## Appendix: result files used (54)

Prefix for the result path: `/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/08_evaluation/04_results_1p5_v2/downstream/`.
Prefix for the nanotron source: `/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b/`.

| Setting | Seed | Epoch | Step | Result JSON | Nanotron source |
|---|---|---|---|---|---|
| Quality Base | 42 | 1 | 4768 | `quality-base-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed42/4768/results_2026-09-07T14-55-27.222136.json` | `seed42/quality_base/ep1/quality_base/seed42/4768` |
| Quality Base | 42 | 2 | 9537 | `quality-base-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed42/9537/results_2026-09-07T13-16-13.960279.json` | `seed42/quality_base/ep2/quality_base/seed42/9537` |
| Quality Base | 42 | 3 | 14305 | `quality-base-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed42/14305/results_2026-09-07T15-10-26.213302.json` | `seed42/quality_base/ep3/quality_base/seed42/14305` |
| Quality Base | 43 | 1 | 4768 | `quality-base-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed43/4768/results_2026-09-07T12-38-06.518044.json` | `seed43/quality_base/ep1/quality_base/seed43/4768` |
| Quality Base | 43 | 2 | 9537 | `quality-base-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed43/9537/results_2026-09-07T13-16-56.167048.json` | `seed43/quality_base/ep2/quality_base/seed43/9537` |
| Quality Base | 43 | 3 | 14305 | `quality-base-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed43/14305/results_2026-09-07T15-11-25.687380.json` | `seed43/quality_base/ep3/quality_base/seed43/14305` |
| Quality Base | 44 | 1 | 4768 | `quality-base-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed44/4768/results_2026-09-07T12-46-32.259436.json` | `seed44/quality_base/ep1/quality_base/seed44/4768` |
| Quality Base | 44 | 2 | 9537 | `quality-base-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed44/9537/results_2026-09-07T13-19-12.770794.json` | `seed44/quality_base/ep2/quality_base/seed44/9537` |
| Quality Base | 44 | 3 | 14305 | `quality-base-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-base-10B-1.5B-seed44/14305/results_2026-09-07T15-12-06.229730.json` | `seed44/quality_base/ep3/quality_base/seed44/14305` |
| Quality First | 42 | 1 | 4768 | `quality-first-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed42/4768/results_2026-09-07T12-47-42.678254.json` | `seed42/quality_first/ep1/quality_first/seed42/4768` |
| Quality First | 42 | 2 | 9537 | `quality-first-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed42/9537/results_2026-09-07T13-19-07.103422.json` | `seed42/quality_first/ep2/quality_first/seed42/9537` |
| Quality First | 42 | 3 | 14305 | `quality-first-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed42/14305/results_2026-09-07T13-35-37.179383.json` | `seed42/quality_first/ep3/quality_first/seed42/14305` |
| Quality First | 43 | 1 | 4768 | `quality-first-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed43/4768/results_2026-09-07T12-57-18.292754.json` | `seed43/quality_first/ep1/quality_first/seed43/4768` |
| Quality First | 43 | 2 | 9537 | `quality-first-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed43/9537/results_2026-09-07T13-20-20.559536.json` | `seed43/quality_first/ep2/quality_first/seed43/9537` |
| Quality First | 43 | 3 | 14305 | `quality-first-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed43/14305/results_2026-09-07T13-36-28.983690.json` | `seed43/quality_first/ep3/quality_first/seed43/14305` |
| Quality First | 44 | 1 | 4768 | `quality-first-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed44/4768/results_2026-09-07T12-56-15.035924.json` | `seed44/quality_first/ep1/quality_first/seed44/4768` |
| Quality First | 44 | 2 | 9537 | `quality-first-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed44/9537/results_2026-09-07T13-19-46.125006.json` | `seed44/quality_first/ep2/quality_first/seed44/9537` |
| Quality First | 44 | 3 | 14305 | `quality-first-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/quality-first-10B-1.5B-seed44/14305/results_2026-09-07T13-43-38.770964.json` | `seed44/quality_first/ep3/quality_first/seed44/14305` |
| Diversity Oriented | 42 | 1 | 4768 | `diversity-first-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed42/4768/results_2026-09-07T14-45-34.582238.json` | `seed42/diversity_oriented/ep1/diversity_oriented/seed42/4768` |
| Diversity Oriented | 42 | 2 | 9537 | `diversity-first-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed42/9537/results_2026-09-07T13-16-24.199709.json` | `seed42/diversity_oriented/ep2/diversity_oriented/seed42/9537` |
| Diversity Oriented | 42 | 3 | 14305 | `diversity-first-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed42/14305/results_2026-09-07T13-29-10.353632.json` | `seed42/diversity_oriented/ep3/diversity_oriented/seed42/14305` |
| Diversity Oriented | 43 | 1 | 4768 | `diversity-first-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed43/4768/results_2026-09-07T14-49-30.936275.json` | `seed43/diversity_oriented/ep1/diversity_oriented/seed43/4768` |
| Diversity Oriented | 43 | 2 | 9537 | `diversity-first-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed43/9537/results_2026-09-07T15-01-32.996156.json` | `seed43/diversity_oriented/ep2/diversity_oriented/seed43/9537` |
| Diversity Oriented | 43 | 3 | 14305 | `diversity-first-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed43/14305/results_2026-09-07T13-29-48.237693.json` | `seed43/diversity_oriented/ep3/diversity_oriented/seed43/14305` |
| Diversity Oriented | 44 | 1 | 4768 | `diversity-first-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed44/4768/results_2026-09-07T12-36-14.861551.json` | `seed44/diversity_oriented/ep1/diversity_oriented/seed44/4768` |
| Diversity Oriented | 44 | 2 | 9537 | `diversity-first-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed44/9537/results_2026-09-07T15-05-11.148432.json` | `seed44/diversity_oriented/ep2/diversity_oriented/seed44/9537` |
| Diversity Oriented | 44 | 3 | 14305 | `diversity-first-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/diversity-first-10B-1.5B-seed44/14305/results_2026-09-07T13-30-18.603662.json` | `seed44/diversity_oriented/ep3/diversity_oriented/seed44/14305` |
| WRAP Inspired | 42 | 1 | 4768 | `wrap-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed42/4768/results_2026-09-07T13-09-08.526398.json` | `seed42/wrap_inspired/ep1/wrap_inspired/seed42/4768` |
| WRAP Inspired | 42 | 2 | 9537 | `wrap-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed42/9537/results_2026-09-07T13-26-10.610428.json` | `seed42/wrap_inspired/ep2/wrap_inspired/seed42/9537` |
| WRAP Inspired | 42 | 3 | 14305 | `wrap-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed42/14305/results_2026-09-07T13-40-05.293035.json` | `seed42/wrap_inspired/ep3/wrap_inspired/seed42/14305` |
| WRAP Inspired | 43 | 1 | 4768 | `wrap-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed43/4768/results_2026-09-07T13-10-32.510163.json` | `seed43/wrap_inspired/ep1/wrap_inspired/seed43/4768` |
| WRAP Inspired | 43 | 2 | 9537 | `wrap-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed43/9537/results_2026-09-07T13-26-21.992861.json` | `seed43/wrap_inspired/ep2/wrap_inspired/seed43/9537` |
| WRAP Inspired | 43 | 3 | 14305 | `wrap-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed43/14305/results_2026-09-08T15-47-57.373452.json` | `seed43/wrap_inspired/ep3/wrap_inspired/seed43/14305` |
| WRAP Inspired | 44 | 1 | 4768 | `wrap-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed44/4768/results_2026-09-07T13-16-25.495612.json` | `seed44/wrap_inspired/ep1/wrap_inspired/seed44/4768` |
| WRAP Inspired | 44 | 2 | 9537 | `wrap-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed44/9537/results_2026-09-07T13-28-46.703664.json` | `seed44/wrap_inspired/ep2/wrap_inspired/seed44/9537` |
| WRAP Inspired | 44 | 3 | 14305 | `wrap-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/wrap-10B-1.5B-seed44/14305/results_2026-09-08T15-57-13.347631.json` | `seed44/wrap_inspired/ep3/wrap_inspired/seed44/14305` |
| REWIRE Inspired | 42 | 1 | 4768 | `rewrite-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed42/4768/results_2026-09-07T12-57-18.448080.json` | `seed42/rewire_inspired/ep1/rewire_inspired/seed42/4768` |
| REWIRE Inspired | 42 | 2 | 9537 | `rewrite-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed42/9537/results_2026-09-07T15-11-33.232329.json` | `seed42/rewire_inspired/ep2/rewire_inspired/seed42/9537` |
| REWIRE Inspired | 42 | 3 | 14305 | `rewrite-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed42/14305/results_2026-09-08T15-36-01.654856.json` | `seed42/rewire_inspired/ep3/rewire_inspired/seed42/14305` |
| REWIRE Inspired | 43 | 1 | 4768 | `rewrite-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed43/4768/results_2026-09-07T13-00-26.992930.json` | `seed43/rewire_inspired/ep1/rewire_inspired/seed43/4768` |
| REWIRE Inspired | 43 | 2 | 9537 | `rewrite-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed43/9537/results_2026-09-08T15-27-45.821850.json` | `seed43/rewire_inspired/ep2/rewire_inspired/seed43/9537` |
| REWIRE Inspired | 43 | 3 | 14305 | `rewrite-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed43/14305/results_2026-09-08T15-37-48.435825.json` | `seed43/rewire_inspired/ep3/rewire_inspired/seed43/14305` |
| REWIRE Inspired | 44 | 1 | 4768 | `rewrite-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed44/4768/results_2026-09-07T13-06-06.658007.json` | `seed44/rewire_inspired/ep1/rewire_inspired/seed44/4768` |
| REWIRE Inspired | 44 | 2 | 9537 | `rewrite-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed44/9537/results_2026-09-07T15-12-26.637611.json` | `seed44/rewire_inspired/ep2/rewire_inspired/seed44/9537` |
| REWIRE Inspired | 44 | 3 | 14305 | `rewrite-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/rewrite-10B-1.5B-seed44/14305/results_2026-09-07T13-38-20.954172.json` | `seed44/rewire_inspired/ep3/rewire_inspired/seed44/14305` |
| Disagreement Aware | 42 | 1 | 4768 | `signal-disagreement-lambda05-10B-1.5B-seed42/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed42/4768/results_2026-09-07T13-07-11.105586.json` | `seed42/disagreement_aware/ep1/disagreement_aware/seed42/4768` |
| Disagreement Aware | 42 | 2 | 9537 | `signal-disagreement-lambda05-10B-1.5B-seed42/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed42/9537/results_2026-09-07T13-27-50.154591.json` | `seed42/disagreement_aware/ep2/disagreement_aware/seed42/9537` |
| Disagreement Aware | 42 | 3 | 14305 | `signal-disagreement-lambda05-10B-1.5B-seed42/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed42/14305/results_2026-09-08T15-46-39.029231.json` | `seed42/disagreement_aware/ep3/disagreement_aware/seed42/14305` |
| Disagreement Aware | 43 | 1 | 4768 | `signal-disagreement-lambda05-10B-1.5B-seed43/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed43/4768/results_2026-09-07T13-12-39.364792.json` | `seed43/disagreement_aware/ep1/disagreement_aware/seed43/4768` |
| Disagreement Aware | 43 | 2 | 9537 | `signal-disagreement-lambda05-10B-1.5B-seed43/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed43/9537/results_2026-09-07T13-31-30.168469.json` | `seed43/disagreement_aware/ep2/disagreement_aware/seed43/9537` |
| Disagreement Aware | 43 | 3 | 14305 | `signal-disagreement-lambda05-10B-1.5B-seed43/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed43/14305/results_2026-09-07T13-38-57.965525.json` | `seed43/disagreement_aware/ep3/disagreement_aware/seed43/14305` |
| Disagreement Aware | 44 | 1 | 4768 | `signal-disagreement-lambda05-10B-1.5B-seed44/4768/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed44/4768/results_2026-09-07T13-09-15.017245.json` | `seed44/disagreement_aware/ep1/disagreement_aware/seed44/4768` |
| Disagreement Aware | 44 | 2 | 9537 | `signal-disagreement-lambda05-10B-1.5B-seed44/9537/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed44/9537/results_2026-09-07T13-31-32.484276.json` | `seed44/disagreement_aware/ep2/disagreement_aware/seed44/9537` |
| Disagreement Aware | 44 | 3 | 14305 | `signal-disagreement-lambda05-10B-1.5B-seed44/14305/results/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b-hf/signal-disagreement-lambda05-10B-1.5B-seed44/14305/results_2026-09-07T13-39-16.894660.json` | `seed44/disagreement_aware/ep3/disagreement_aware/seed44/14305` |
