| seed | ep | new setting | v2 config | keys in new | equal | differing (non-run) | only in v2 (count) |
|---|---|---|---|---:|---:|---|---:|
| 42 | ep1 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep1 | `raw_diversity_oriented` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep1 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep1 | `raw_disagreement_aware` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep1 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep1 | `signal-disagreement-lambda0` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `raw_diversity_oriented` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `raw_disagreement_aware` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep2 | `signal-disagreement-lambda0` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `raw_diversity_oriented` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `raw_disagreement_aware` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 42 | ep3 | `signal-disagreement-lambda0` | `quality_base` | 71 | 63 | `tokens.micro_batch_size` 32 vs 16, `tokens.batch_accumulation_per_replica` 8 vs 16, `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep1 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep2 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 43 | ep3 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep1 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep2 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `raw_diversity_oriented` | `diversity_oriented` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `raw_diversity_oriented` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `raw_disagreement_aware` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `raw_disagreement_aware` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `signal-disagreement-lambda0` | `disagreement_aware` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |
| 44 | ep3 | `signal-disagreement-lambda0` | `quality_base` | 71 | 65 | `parallelism.recompute_layer` '{{RECOMPUTE_LAYER}}' vs False | 34 |

Differing keys across all rows:

- `parallelism.recompute_layer`: ['["{{RECOMPUTE_LAYER}}", false]']
- `tokens.micro_batch_size`: ['[32, 16]']
- `tokens.batch_accumulation_per_replica`: ['[8, 16]']
