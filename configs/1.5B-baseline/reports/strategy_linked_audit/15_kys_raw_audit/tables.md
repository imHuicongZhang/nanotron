### T1. Input → successful rewrite sources → raw half (documents / source TRAIN tokens)

| setting | I: rewriting input | R: unique sources of final rewritten half | S: raw strategy half | I − R (dropped before training) |
|---|---:|---:|---:|---:|
| `raw_diversity_oriented` | 5,876,747 / 10,000,077,737 | 5,867,876 / 9,728,111,104 | 3,018,451 / 5,000,000,553 | 8,871 / 271,966,633 |
| `raw_disagreement_aware` | 5,602,476 / 10,000,002,827 | 5,592,424 / 9,647,286,028 | 2,898,764 / 5,000,002,737 | 10,052 / 352,716,799 |
| `raw_random` | 10,604,458 / 10,000,000,190 | 10,573,523 / 9,520,542,699 | 5,554,525 / 5,000,001,419 | 30,935 / 479,457,491 |
| `raw_rewire_inspired` | 21,214,299 / 20,000,000,679 | 8,445,785 / 10,899,228,624 | 3,872,986 / 5,000,000,799 | 12,768,514 / 9,100,772,055 |

### T2. Subset relations and rule re-execution

| setting | R ⊆ I | S ⊆ R | S ∩ anchor | source_tokens.npy = pool tokens-llama2+1 | seed-42 rule re-executed on R reproduces S |
|---|---|---|---:|---|---|
| `raw_diversity_oriented` | True | True | 0 | True | True |
| `raw_disagreement_aware` | True | True | 0 | True | True |
| `raw_random` | True | True | 0 | True | True |
| `raw_rewire_inspired` | True | True | 0 | True | True |

### T3. Directional coverage (documents; source-TRAIN-token weighted in parentheses)

Coverage of A by B = |A ∩ B| / |A|; token-weighted = Σ(tokens-llama2+1 of A ∩ B) / Σ(tokens-llama2+1 of A).

| setting | S in R | R in S | Jaccard(S,R) | S in I | I in S | Jaccard(S,I) | I in R |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_diversity_oriented` | 100.00% (100.00%) | 51.44% (51.40%) | 0.5144 | 100.00% (100.00%) | 51.36% (50.00%) | 0.5136 | 99.85% (97.28%) |
| `raw_disagreement_aware` | 100.00% (100.00%) | 51.83% (51.83%) | 0.5183 | 100.00% (100.00%) | 51.74% (50.00%) | 0.5174 | 99.82% (96.47%) |
| `raw_random` | 100.00% (100.00%) | 52.53% (52.52%) | 0.5253 | 100.00% (100.00%) | 52.38% (50.00%) | 0.5238 | 99.71% (95.21%) |
| `raw_rewire_inspired` | 100.00% (100.00%) | 45.86% (45.87%) | 0.4586 | 100.00% (100.00%) | 18.26% (25.00%) | 0.1826 | 39.81% (54.50%) |

### T4. Raw half vs the rewritten outputs (one-to-many)

Rewritten rows are counted per output row; a source rewritten by two prompts contributes two rows. Output-token coverage = rewritten TRAIN tokens (rewritten_tokens + 1) whose source is in S, over all rewritten TRAIN tokens of the half.

| setting | rewrite rows (by prompt) | sources with 1 / 2 outputs | rewritten TRAIN tokens | rows with source in S | output-token coverage by S |
|---|---|---:|---:|---:|---:|
| `raw_diversity_oriented` | distill 2,474,728, wikipedia 5,861,683 | 3,399,341 / 2,468,535 | 4,889,635,504 | 4,288,640 | 51.46% |
| `raw_disagreement_aware` | distill 2,855,159, wikipedia 5,587,114 | 2,742,575 / 2,849,849 | 5,000,000,003 | 4,376,795 | 51.84% |
| `raw_random` | distill 2,456,200, wrap_easy 2,646,480, wrap_hard 2,631,853, wrap_qa 2,643,215, wrap_wiki 2,644,343 | 8,124,955 / 2,448,568 | 5,000,000,087 | 6,840,890 | 52.52% |
| `raw_rewire_inspired` | distill 7,024,123, wikipedia 5,266,321 | 4,601,126 / 3,844,659 | 5,000,000,351 | 5,636,442 | 45.87% |

### T5. Distribution drift caused by conditioning (TRAIN-token-weighted 24-topic TV / JS in bits; mean percentiles; length)

| setting | topic TV R vs I | topic TV S vs I | JS S vs I | mean fastText pct I → R → S | mean q I → R → S | mean TRAIN tokens/doc I → R → S | I − R: mean tokens/doc |
|---|---:|---:|---:|---|---|---|---:|
| `raw_diversity_oriented` | 0.0067 | 0.0069 | 0.00009 | 0.844 → 0.844 → 0.844 | 0.854 → 0.854 → 0.854 | 1702 → 1658 → 1656 | 30658 |
| `raw_disagreement_aware` | 0.0102 | 0.0106 | 0.00016 | 0.837 → 0.837 → 0.837 | 0.872 → 0.872 → 0.872 | 1785 → 1725 → 1725 | 35089 |
| `raw_random` | 0.0112 | 0.0117 | 0.00019 | 0.479 → 0.479 → 0.479 | 0.489 → 0.488 → 0.488 | 943 → 900 → 900 | 15499 |
| `raw_rewire_inspired` | 0.0834 | 0.0833 | 0.00647 | 0.479 → 0.629 → 0.629 | 0.489 → 0.582 → 0.582 | 943 → 1290 → 1291 | 713 |

### T6. Published corpora (local staging copy of the Hub files; sha256 listed in manifest.json)

| setting | rows | TRAIN tokens (pool tokens-llama2+1) | anchor rows | strategy rows | doc set = anchor ∪ S | duplicates | S digest (sorted orig_doc_id) | file-order digest |
|---|---:|---:|---:|---:|---|---:|---|---|
| `raw_diversity_oriented` | 7,138,615 | 10,000,002,885 | 4,120,164 | 3,018,451 | True | 0 | `4cdbeb0fb93a4331…` | `22843235ff43fb00…` |
| `raw_disagreement_aware` | 7,018,928 | 10,000,005,069 | 4,120,164 | 2,898,764 | True | 0 | `08ae25c57439792e…` | `678ceca45644212e…` |
| `raw_random` | 9,674,689 | 10,000,003,751 | 4,120,164 | 5,554,525 | True | 0 | `75b91f4d6b96f7b9…` | `c4749e064980e4dd…` |
| `raw_rewire_inspired` | 7,993,150 | 10,000,003,131 | 4,120,164 | 3,872,986 | True | 0 | `c0693ae04e5ddad7…` | `5a89bd5c7ef20fec…` |

### T7. Quality-First input vs the Quality-Base 5B block

| set | docs | TRAIN tokens | mean fastText pct | min fastText pct |
|---|---:|---:|---:|---:|
| Quality-First rewriting input (next 10B by fastText after the anchor) | 6,136,187 | 10,000,000,529 | 0.9280 | 0.8973 |
| unique sources of the final Quality-First rewritten half | 6,122,949 | 9,509,596,850 | 0.9281 | 0.8973 |
| Quality-Base non-anchor 5B block | 3,133,023 | 5,000,000,805 | 0.9431 | 0.9274 |

Quality-Base block ⊆ Quality-First input: True; block covered by R: 99.87% of docs (95.61% of tokens); R covered by the block: 51.10% (50.27%); block ∩ anchor = 0. `wytro/Know-Your-Sources` revision `9e5ff24149c2957c30f0c8fdd051a8eb3b75baad`.

### T8. Reconstruction checks

- shared-top-5B reproduced from `04_select/select_10b.py` rules equals the published anchor: **True** (4,120,164 docs, 5,000,002,332 TRAIN tokens; sorted orig_doc_id sha256 `d8a6d22307c2e7c325fdc20e52039ace950a6808dcc21f9b5e5b5687d2c15de4`, sorted doc_id sha256 `c80e0e7f8327f840d40028c7d8f9bc239d94d5e3a2c1408314a60cebda41658b`).
- λ=0.5 disagreement-aware input reproduced from `05_select_s5_variants/select_s5.py` rules equals bit 2 of `06_lambda_grid/lambda_grid.npz`: **True** (U = 14,982,068 docs, Q30(q) = 0.6956847310, Q90(v) = 0.1036491916, survivors 10,435,667).
- input `wrap`: 10,604,458 docs, 10,000,000,190 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `a46a74b5a46c8c13…`
- input `rewrite`: 21,214,299 docs, 20,000,000,679 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `9fc9f5bffccfb827…`
- input `diversity`: 5,876,747 docs, 10,000,077,737 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `566649d9e0e4b018…`
- input `lambda05`: 5,602,476 docs, 10,000,002,827 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `eb8f872a6ed03e3a…`
