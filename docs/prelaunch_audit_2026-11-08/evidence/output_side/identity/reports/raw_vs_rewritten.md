# Raw-selected vs v2 comparators

Each raw-selected setting next to its v2 comparator, per epoch. Columns are `acc_norm` in percent, mean ± std over seeds.

Three kinds of pairs. The four **strategy-linked** controls are compared with their rewritten counterpart. The three **global Top-10B** selections are compared with the existing fastText Quality-Base arm: that is a comparison between selection scores on raw data, not a rewriting comparison. The **λ grid** arm `signal-disagreement-lambda0` (Disagreement-Aware selection at λ = 0, rewritten) is compared with the published λ = 0.5 Disagreement-Aware arm: both are rewritten, so Δ isolates the disagreement term.

**Δ = comparator − raw**, in accuracy points: positive means the comparator scores higher. Δ is the mean of the per-seed differences (the two arms share each seed's initialization), ± the standard deviation of those differences; *Seeds Δ>0* counts seeds with a positive difference.

Comparator side: the v2 evaluation of the 54 rewritten-grid checkpoints (`reference/rewritten_v2_54_per_task.csv`, all scored on NVIDIA H100 80GB HBM3). These v2 checkpoints are the source of the paper's ICLR tables.

| Raw-selected | Kind | Comparator |
|---|---|---|
| `raw_diversity_oriented` | strategy-linked | Diversity Oriented (`diversity_oriented`) |
| `raw_disagreement_aware` | strategy-linked | Disagreement Aware (`disagreement_aware`) |
| `raw_random` | strategy-linked | WRAP Inspired (`wrap_inspired`) |
| `raw_rewire_inspired` | strategy-linked | REWIRE Inspired (`rewire_inspired`) |
| `raw_top10b_fineweb_edu` | global Top-10B | Quality Base (`quality_base`) |
| `raw_top10b_modernbert` | global Top-10B | Quality Base (`quality_base`) |
| `raw_top10b_consensus` | global Top-10B | Quality Base (`quality_base`) |
| `signal-disagreement-lambda0` | λ grid (λ = 0 vs 0.5) | Disagreement Aware (`disagreement_aware`) |

All checkpoints were scored on NVIDIA H100 80GB HBM3, the device used for the v2 rewritten results.

## Mean6 (commonsense average)

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 42.50 ± 0.43 | 42.50 ± 0.43 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 45.40 ± 0.29 | 45.40 ± 0.29 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 46.41 ± 0.21 | 46.41 ± 0.21 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 42.11 ± 0.29 | 42.11 ± 0.29 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 44.82 ± 0.34 | 44.82 ± 0.34 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 45.93 ± 0.11 | 45.93 ± 0.11 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 41.97 ± 0.39 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 44.62 ± 0.44 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 45.81 ± 0.24 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 42.28 ± 0.14 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 44.94 ± 0.31 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 45.89 ± 0.23 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 41.75 ± 0.67 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 44.80 ± 0.07 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 46.27 ± 0.67 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 41.75 ± 0.67 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 44.80 ± 0.07 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 46.27 ± 0.67 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 41.75 ± 0.67 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 44.80 ± 0.07 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 46.27 ± 0.67 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 42.11 ± 0.29 | 42.11 ± 0.29 | +0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 44.82 ± 0.34 | 44.82 ± 0.34 | -0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 45.93 ± 0.11 | 45.93 ± 0.11 | +0.00 ± 0.00 | 2/3 |

## MMLU (57-subject macro-average)

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 28.19 ± 0.26 | 28.19 ± 0.26 | +0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 29.93 ± 0.12 | 29.93 ± 0.12 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 30.82 ± 0.13 | 30.82 ± 0.13 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 28.32 ± 0.07 | 28.32 ± 0.07 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 30.13 ± 0.20 | 30.13 ± 0.20 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 31.08 ± 0.40 | 31.08 ± 0.40 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 27.70 ± 0.33 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 28.95 ± 0.19 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 29.72 ± 0.39 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 28.19 ± 0.24 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 29.55 ± 0.38 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 30.59 ± 0.50 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 28.16 ± 0.27 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 29.53 ± 0.09 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 30.59 ± 0.03 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 28.16 ± 0.27 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 29.53 ± 0.09 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 30.59 ± 0.03 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 28.16 ± 0.27 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 29.53 ± 0.09 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 30.59 ± 0.03 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 28.32 ± 0.07 | 28.32 ± 0.07 | -0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 30.13 ± 0.20 | 30.13 ± 0.20 | +0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 31.08 ± 0.40 | 31.08 ± 0.40 | +0.00 ± 0.00 | 2/3 |

## ARC-Easy

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 45.24 ± 0.65 | 45.24 ± 0.65 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 50.14 ± 0.78 | 50.14 ± 0.78 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 52.03 ± 0.90 | 52.03 ± 0.90 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 46.66 ± 0.97 | 46.66 ± 0.97 | -0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 50.53 ± 0.40 | 50.53 ± 0.40 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 51.91 ± 0.81 | 51.91 ± 0.81 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 44.40 ± 0.94 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 48.55 ± 0.56 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 51.00 ± 1.12 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 44.73 ± 0.51 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 49.28 ± 1.46 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 51.00 ± 0.75 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 45.29 ± 0.99 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 49.92 ± 0.19 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 52.20 ± 1.67 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 45.29 ± 0.99 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 49.92 ± 0.19 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 52.20 ± 1.67 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 45.29 ± 0.99 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 49.92 ± 0.19 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 52.20 ± 1.67 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 46.66 ± 0.97 | 46.66 ± 0.97 | -0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 50.53 ± 0.40 | 50.53 ± 0.40 | -0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 51.91 ± 0.81 | 51.91 ± 0.81 | +0.00 ± 0.00 | 2/3 |

## HellaSwag

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 40.80 ± 0.27 | 40.80 ± 0.27 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 46.87 ± 0.18 | 46.87 ± 0.18 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 49.72 ± 0.09 | 49.72 ± 0.09 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 39.17 ± 0.38 | 39.17 ± 0.38 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 44.83 ± 0.43 | 44.83 ± 0.43 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 47.49 ± 0.31 | 47.49 ± 0.31 | -0.00 ± 0.00 | 1/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 39.72 ± 0.09 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 45.10 ± 0.23 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 48.02 ± 0.49 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 40.09 ± 0.19 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 46.09 ± 0.47 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 48.86 ± 0.34 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 38.99 ± 0.28 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 45.24 ± 0.48 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 48.37 ± 0.54 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 38.99 ± 0.28 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 45.24 ± 0.48 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 48.37 ± 0.54 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 38.99 ± 0.28 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 45.24 ± 0.48 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 48.37 ± 0.54 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 39.17 ± 0.38 | 39.17 ± 0.38 | +0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 44.83 ± 0.43 | 44.83 ± 0.43 | -0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 47.49 ± 0.31 | 47.49 ± 0.31 | -0.00 ± 0.00 | 1/3 |

## PIQA

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 68.32 ± 0.14 | 68.32 ± 0.14 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 71.15 ± 0.88 | 71.15 ± 0.88 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 71.89 ± 0.13 | 71.89 ± 0.13 | +0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 67.48 ± 0.28 | 67.48 ± 0.28 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 69.62 ± 0.44 | 69.62 ± 0.44 | +0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 71.04 ± 0.28 | 71.04 ± 0.28 | +0.00 ± 0.00 | 3/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 68.15 ± 0.68 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 70.26 ± 0.77 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 71.44 ± 0.05 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 68.32 ± 0.13 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 70.13 ± 0.30 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 71.15 ± 0.38 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 67.23 ± 1.17 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 69.82 ± 0.75 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 70.35 ± 0.42 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 67.23 ± 1.17 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 69.82 ± 0.75 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 70.35 ± 0.42 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 67.23 ± 1.17 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 69.82 ± 0.75 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 70.35 ± 0.42 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 67.48 ± 0.28 | 67.48 ± 0.28 | +0.00 ± 0.00 | 3/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 69.62 ± 0.44 | 69.62 ± 0.44 | +0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 71.04 ± 0.28 | 71.04 ± 0.28 | +0.00 ± 0.00 | 3/3 |

## SIQA

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 39.34 ± 0.21 | 39.34 ± 0.21 | -0.00 ± 0.00 | 2/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 40.19 ± 0.56 | 40.19 ± 0.56 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 39.82 ± 0.91 | 39.82 ± 0.91 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 39.13 ± 0.45 | 39.13 ± 0.45 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 39.82 ± 0.62 | 39.82 ± 0.62 | +0.00 ± 0.00 | 2/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 40.00 ± 0.37 | 40.00 ± 0.37 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 39.59 ± 0.11 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 40.07 ± 0.76 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 39.87 ± 0.44 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 39.44 ± 0.15 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 40.29 ± 0.37 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 40.02 ± 0.27 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 39.59 ± 0.18 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 40.02 ± 0.67 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 40.52 ± 0.38 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 39.59 ± 0.18 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 40.02 ± 0.67 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 40.52 ± 0.38 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 39.59 ± 0.18 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 40.02 ± 0.67 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 40.52 ± 0.38 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 39.13 ± 0.45 | 39.13 ± 0.45 | +0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 39.82 ± 0.62 | 39.82 ± 0.62 | +0.00 ± 0.00 | 2/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 40.00 ± 0.37 | 40.00 ± 0.37 | +0.00 ± 0.00 | 2/3 |

## OpenBookQA

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 32.00 ± 1.22 | 32.00 ± 1.22 | +0.00 ± 0.00 | 0/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 33.27 ± 1.01 | 33.27 ± 1.01 | +0.00 ± 0.00 | 0/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 34.20 ± 1.00 | 34.20 ± 1.00 | +0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 30.87 ± 0.31 | 30.87 ± 0.31 | +0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 33.27 ± 1.03 | 33.27 ± 1.03 | +0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 33.60 ± 0.72 | 33.60 ± 0.72 | +0.00 ± 0.00 | 0/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 30.13 ± 0.64 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 32.00 ± 1.06 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 32.53 ± 0.83 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 30.80 ± 0.60 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 32.27 ± 1.33 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 32.00 ± 0.40 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 29.27 ± 1.72 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 33.00 ± 1.00 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 34.13 ± 1.53 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 29.27 ± 1.72 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 33.00 ± 1.00 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 34.13 ± 1.53 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 29.27 ± 1.72 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 33.00 ± 1.00 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 34.13 ± 1.53 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 30.87 ± 0.31 | 30.87 ± 0.31 | +0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 33.27 ± 1.03 | 33.27 ± 1.03 | +0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 33.60 ± 0.72 | 33.60 ± 0.72 | +0.00 ± 0.00 | 0/3 |

## CommonsenseQA

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 29.32 ± 0.62 | 29.32 ± 0.62 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 30.79 ± 0.22 | 30.79 ± 0.22 | +0.00 ± 0.00 | 3/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 30.82 ± 1.02 | 30.82 ± 1.02 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 29.32 ± 1.06 | 29.32 ± 1.06 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 30.88 ± 0.50 | 30.88 ± 0.50 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 31.53 ± 0.22 | 31.53 ± 0.22 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 29.81 ± 0.87 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 31.75 ± 0.61 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 32.02 ± 0.73 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 30.33 ± 0.48 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 31.59 ± 0.58 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 32.30 ± 0.45 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 30.11 ± 0.41 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 30.82 ± 1.36 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 32.05 ± 0.90 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 30.11 ± 0.41 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 30.82 ± 1.36 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 32.05 ± 0.90 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 30.11 ± 0.41 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 30.82 ± 1.36 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 32.05 ± 0.90 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 29.32 ± 1.06 | 29.32 ± 1.06 | +0.00 ± 0.00 | 3/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 30.88 ± 0.50 | 30.88 ± 0.50 | -0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 31.53 ± 0.22 | 31.53 ± 0.22 | +0.00 ± 0.00 | 2/3 |

## MMLU STEM

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 25.41 ± 0.25 | 25.41 ± 0.25 | +0.00 ± 0.00 | 2/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 26.35 ± 0.21 | 26.35 ± 0.21 | +0.00 ± 0.00 | 2/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 26.87 ± 0.47 | 26.87 ± 0.47 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 25.54 ± 0.70 | 25.54 ± 0.70 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 26.47 ± 0.41 | 26.47 ± 0.41 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 27.31 ± 0.57 | 27.31 ± 0.57 | +0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 25.23 ± 0.17 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 25.76 ± 0.21 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 26.07 ± 0.46 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 26.16 ± 0.14 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 26.58 ± 0.50 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 27.13 ± 0.96 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 26.09 ± 0.35 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 26.87 ± 0.62 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 27.52 ± 0.25 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 26.09 ± 0.35 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 26.87 ± 0.62 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 27.52 ± 0.25 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 26.09 ± 0.35 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 26.87 ± 0.62 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 27.52 ± 0.25 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 25.54 ± 0.70 | 25.54 ± 0.70 | -0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 26.47 ± 0.41 | 26.47 ± 0.41 | +0.00 ± 0.00 | 3/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 27.31 ± 0.57 | 27.31 ± 0.57 | +0.00 ± 0.00 | 2/3 |

## MMLU Humanities

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 27.80 ± 0.40 | 27.80 ± 0.40 | -0.00 ± 0.00 | 2/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 30.33 ± 0.28 | 30.33 ± 0.28 | -0.00 ± 0.00 | 0/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 31.15 ± 0.41 | 31.15 ± 0.41 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 28.13 ± 0.34 | 28.13 ± 0.34 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 30.33 ± 0.13 | 30.33 ± 0.13 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 31.58 ± 0.37 | 31.58 ± 0.37 | -0.00 ± 0.00 | 2/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 26.99 ± 0.20 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 28.93 ± 0.34 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 29.61 ± 0.14 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 27.41 ± 0.42 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 29.16 ± 0.66 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 30.26 ± 0.54 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 27.23 ± 0.42 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 29.16 ± 0.74 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 30.58 ± 0.25 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 27.23 ± 0.42 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 29.16 ± 0.74 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 30.58 ± 0.25 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 27.23 ± 0.42 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 29.16 ± 0.74 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 30.58 ± 0.25 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 28.13 ± 0.34 | 28.13 ± 0.34 | -0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 30.33 ± 0.13 | 30.33 ± 0.13 | -0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 31.58 ± 0.37 | 31.58 ± 0.37 | -0.00 ± 0.00 | 2/3 |

## MMLU Social Sciences

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 29.72 ± 0.37 | 29.72 ± 0.37 | +0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 31.81 ± 0.20 | 31.81 ± 0.20 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 33.01 ± 0.82 | 33.01 ± 0.82 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 29.77 ± 0.72 | 29.77 ± 0.72 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 31.49 ± 0.93 | 31.49 ± 0.93 | -0.00 ± 0.00 | 1/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 32.43 ± 0.31 | 32.43 ± 0.31 | -0.00 ± 0.00 | 1/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 28.71 ± 0.94 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 30.17 ± 0.36 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 32.06 ± 0.57 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 29.65 ± 0.55 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 30.69 ± 0.37 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 32.20 ± 0.62 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 29.14 ± 0.37 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 30.70 ± 0.20 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 31.89 ± 0.50 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 29.14 ± 0.37 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 30.70 ± 0.20 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 31.89 ± 0.50 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 29.14 ± 0.37 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 30.70 ± 0.20 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 31.89 ± 0.50 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 29.77 ± 0.72 | 29.77 ± 0.72 | -0.00 ± 0.00 | 0/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 31.49 ± 0.93 | 31.49 ± 0.93 | -0.00 ± 0.00 | 1/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 32.43 ± 0.31 | 32.43 ± 0.31 | -0.00 ± 0.00 | 1/3 |

## MMLU Other

| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |
|---|---|---|---|---|---|---|
| `raw_diversity_oriented` | Diversity Oriented | ep1 | 30.82 ± 0.47 | 30.82 ± 0.47 | +0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep2 | 32.56 ± 0.23 | 32.56 ± 0.23 | -0.00 ± 0.00 | 1/3 |
| `raw_diversity_oriented` | Diversity Oriented | ep3 | 33.72 ± 0.60 | 33.72 ± 0.60 | -0.00 ± 0.00 | 0/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep1 | 30.83 ± 0.35 | 30.83 ± 0.35 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep2 | 33.49 ± 0.35 | 33.49 ± 0.35 | +0.00 ± 0.00 | 3/3 |
| `raw_disagreement_aware` | Disagreement Aware | ep3 | 34.32 ± 0.43 | 34.32 ± 0.43 | +0.00 ± 0.00 | 1/3 |
| `raw_random` | WRAP Inspired | ep1 | — | 30.67 ± 0.68 | — | — |
| `raw_random` | WRAP Inspired | ep2 | — | 32.01 ± 0.33 | — | — |
| `raw_random` | WRAP Inspired | ep3 | — | 32.50 ± 0.61 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep1 | — | 30.29 ± 0.39 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep2 | — | 32.75 ± 0.66 | — | — |
| `raw_rewire_inspired` | REWIRE Inspired | ep3 | — | 33.96 ± 0.64 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep1 | — | 30.85 ± 0.51 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep2 | — | 32.30 ± 0.65 | — | — |
| `raw_top10b_fineweb_edu` | Quality Base | ep3 | — | 33.44 ± 0.22 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep1 | — | 30.85 ± 0.51 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep2 | — | 32.30 ± 0.65 | — | — |
| `raw_top10b_modernbert` | Quality Base | ep3 | — | 33.44 ± 0.22 | — | — |
| `raw_top10b_consensus` | Quality Base | ep1 | — | 30.85 ± 0.51 | — | — |
| `raw_top10b_consensus` | Quality Base | ep2 | — | 32.30 ± 0.65 | — | — |
| `raw_top10b_consensus` | Quality Base | ep3 | — | 33.44 ± 0.22 | — | — |
| `signal-disagreement-lambda0` | Disagreement Aware | ep1 | 30.83 ± 0.35 | 30.83 ± 0.35 | +0.00 ± 0.00 | 3/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep2 | 33.49 ± 0.35 | 33.49 ± 0.35 | +0.00 ± 0.00 | 3/3 |
| `signal-disagreement-lambda0` | Disagreement Aware | ep3 | 34.32 ± 0.43 | 34.32 ± 0.43 | +0.00 ± 0.00 | 1/3 |
