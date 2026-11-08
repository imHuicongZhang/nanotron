# Raw-selected baselines: results

Generated 2026-11-08T17:49:59+00:00 from `/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-lambda0/docs/prelaunch_audit_2026-11-08/evidence/output_side/identity/results`.

Values are `acc_norm` in percent: 0-shot, full evaluation splits, token-length-normalized log-likelihood (protocol in `kys_eval/config.py`). Each cell is the mean ± sample standard deviation over seeds 42, 43, 44; `(n=k)` marks cells with fewer seeds.

Coverage: **27 / 72** checkpoints.

All checkpoints were scored on NVIDIA H100 80GB HBM3, the device used for the v2 rewritten results.

<details><summary>Checkpoints without complete results</summary>

- `raw_random` seed 42 ep1
- `raw_random` seed 42 ep2
- `raw_random` seed 42 ep3
- `raw_rewire_inspired` seed 42 ep1
- `raw_rewire_inspired` seed 42 ep2
- `raw_rewire_inspired` seed 42 ep3
- `raw_top10b_fineweb_edu` seed 42 ep1
- `raw_top10b_fineweb_edu` seed 42 ep2
- `raw_top10b_fineweb_edu` seed 42 ep3
- `raw_top10b_modernbert` seed 42 ep1
- `raw_top10b_modernbert` seed 42 ep2
- `raw_top10b_modernbert` seed 42 ep3
- `raw_top10b_consensus` seed 42 ep1
- `raw_top10b_consensus` seed 42 ep2
- `raw_top10b_consensus` seed 42 ep3
- `raw_random` seed 43 ep1
- `raw_random` seed 43 ep2
- `raw_random` seed 43 ep3
- `raw_rewire_inspired` seed 43 ep1
- `raw_rewire_inspired` seed 43 ep2
- `raw_rewire_inspired` seed 43 ep3
- `raw_top10b_fineweb_edu` seed 43 ep1
- `raw_top10b_fineweb_edu` seed 43 ep2
- `raw_top10b_fineweb_edu` seed 43 ep3
- `raw_top10b_modernbert` seed 43 ep1
- `raw_top10b_modernbert` seed 43 ep2
- `raw_top10b_modernbert` seed 43 ep3
- `raw_top10b_consensus` seed 43 ep1
- `raw_top10b_consensus` seed 43 ep2
- `raw_top10b_consensus` seed 43 ep3
- `raw_random` seed 44 ep1
- `raw_random` seed 44 ep2
- `raw_random` seed 44 ep3
- `raw_rewire_inspired` seed 44 ep1
- `raw_rewire_inspired` seed 44 ep2
- `raw_rewire_inspired` seed 44 ep3
- `raw_top10b_fineweb_edu` seed 44 ep1
- `raw_top10b_fineweb_edu` seed 44 ep2
- `raw_top10b_fineweb_edu` seed 44 ep3
- `raw_top10b_modernbert` seed 44 ep1
- `raw_top10b_modernbert` seed 44 ep2
- `raw_top10b_modernbert` seed 44 ep3
- `raw_top10b_consensus` seed 44 ep1
- `raw_top10b_consensus` seed 44 ep2
- `raw_top10b_consensus` seed 44 ep3

</details>

## Summary: average accuracy (Mean6)

Mean6 is the unweighted mean of ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA and CommonsenseQA, the paper's headline commonsense average (Fig. 3a).

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 42.50 ± 0.43 | 45.40 ± 0.29 | 46.41 ± 0.21 |
| `raw_disagreement_aware` | 42.11 ± 0.29 | 44.82 ± 0.34 | 45.93 ± 0.11 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 42.11 ± 0.29 | 44.82 ± 0.34 | 45.93 ± 0.11 |

## MMLU

Unweighted macro-average over the 57 subjects (Fig. 3b).

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 28.19 ± 0.26 | 29.93 ± 0.12 | 30.82 ± 0.13 |
| `raw_disagreement_aware` | 28.32 ± 0.07 | 30.13 ± 0.20 | 31.08 ± 0.40 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 28.32 ± 0.07 | 30.13 ± 0.20 | 31.08 ± 0.40 |

## ARC-Easy

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 45.24 ± 0.65 | 50.14 ± 0.78 | 52.03 ± 0.90 |
| `raw_disagreement_aware` | 46.66 ± 0.97 | 50.53 ± 0.40 | 51.91 ± 0.81 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 46.66 ± 0.97 | 50.53 ± 0.40 | 51.91 ± 0.81 |

## HellaSwag

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 40.80 ± 0.27 | 46.87 ± 0.18 | 49.72 ± 0.09 |
| `raw_disagreement_aware` | 39.17 ± 0.38 | 44.83 ± 0.43 | 47.49 ± 0.31 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 39.17 ± 0.38 | 44.83 ± 0.43 | 47.49 ± 0.31 |

## PIQA

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 68.32 ± 0.14 | 71.15 ± 0.88 | 71.89 ± 0.13 |
| `raw_disagreement_aware` | 67.48 ± 0.28 | 69.62 ± 0.44 | 71.04 ± 0.28 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 67.48 ± 0.28 | 69.62 ± 0.44 | 71.04 ± 0.28 |

## SIQA

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 39.34 ± 0.21 | 40.19 ± 0.56 | 39.82 ± 0.91 |
| `raw_disagreement_aware` | 39.13 ± 0.45 | 39.82 ± 0.62 | 40.00 ± 0.37 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 39.13 ± 0.45 | 39.82 ± 0.62 | 40.00 ± 0.37 |

## OpenBookQA

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 32.00 ± 1.22 | 33.27 ± 1.01 | 34.20 ± 1.00 |
| `raw_disagreement_aware` | 30.87 ± 0.31 | 33.27 ± 1.03 | 33.60 ± 0.72 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 30.87 ± 0.31 | 33.27 ± 1.03 | 33.60 ± 0.72 |

## CommonsenseQA

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 29.32 ± 0.62 | 30.79 ± 0.22 | 30.82 ± 1.02 |
| `raw_disagreement_aware` | 29.32 ± 1.06 | 30.88 ± 0.50 | 31.53 ± 0.22 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 29.32 ± 1.06 | 30.88 ± 0.50 | 31.53 ± 0.22 |

## MMLU by category

Mean over each category's subjects (Table 3 breakdown).

### MMLU STEM

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 25.41 ± 0.25 | 26.35 ± 0.21 | 26.87 ± 0.47 |
| `raw_disagreement_aware` | 25.54 ± 0.70 | 26.47 ± 0.41 | 27.31 ± 0.57 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 25.54 ± 0.70 | 26.47 ± 0.41 | 27.31 ± 0.57 |

### MMLU Humanities

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 27.80 ± 0.40 | 30.33 ± 0.28 | 31.15 ± 0.41 |
| `raw_disagreement_aware` | 28.13 ± 0.34 | 30.33 ± 0.13 | 31.58 ± 0.37 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 28.13 ± 0.34 | 30.33 ± 0.13 | 31.58 ± 0.37 |

### MMLU Social Sciences

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 29.72 ± 0.37 | 31.81 ± 0.20 | 33.01 ± 0.82 |
| `raw_disagreement_aware` | 29.77 ± 0.72 | 31.49 ± 0.93 | 32.43 ± 0.31 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 29.77 ± 0.72 | 31.49 ± 0.93 | 32.43 ± 0.31 |

### MMLU Other

| Setting | ep1 (step 4768) | ep2 (step 9537) | ep3 (step 14305) |
|---|---|---|---|
| `raw_diversity_oriented` | 30.82 ± 0.47 | 32.56 ± 0.23 | 33.72 ± 0.60 |
| `raw_disagreement_aware` | 30.83 ± 0.35 | 33.49 ± 0.35 | 34.32 ± 0.43 |
| `raw_random` | — | — | — |
| `raw_rewire_inspired` | — | — | — |
| `raw_top10b_fineweb_edu` | — | — | — |
| `raw_top10b_modernbert` | — | — | — |
| `raw_top10b_consensus` | — | — | — |
| `signal-disagreement-lambda0` | 30.83 ± 0.35 | 33.49 ± 0.35 | 34.32 ± 0.43 |
