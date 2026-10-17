# 3B rewrite-600m grid: design record (prepared 2026-10-17, NOT launched)

Three settings (quality-first, diversity-oriented, disagreement-aware) trained on the validated 600M-based
corpora at 3B scale. The grid uses seed 42, one shared init, and a WSD trunk with three decay branches.
Nothing here changes the 1.5B baseline grid.
- Code: worktree `projects/nanotron-kys-3b600m`, branch `rewrite3b-600m-train`.
- Base: **bc131392**, the exact revision the active 1.5B baselines run from (`projects/nanotron-kys`, branch
  `skipjack-1p5b-baselines`, left untouched).
- Every 3B file is new, under `configs/3B-rewrite-600m/` and `deploy/skipjack_3b_600m/`.
- Values come from `deploy/skipjack_3b_600m/registry_3b.py`.

Labels used below: **P** = stated in the paper (`rewrite-3B/00_paper/Know_Your_Sources.pdf`, sha256 `a7aed70a…`,
"Under review as a conference paper at ICLR 2027", PDF created 2026-09-26). **C** = configured in code / rendered
configs. **V** = verified at runtime or by measurement. **E** = engineering estimate.

## Method reference (P)
- **Trunk and branches.** App. E, PDF p. 26: "Each run uses a shared constant-learning-rate trunk with a 500-step linear warmup.
  The three reported epoch checkpoints branch from this trunk and linearly decay to zero over the final 10% of their
  respective horizons, ending at steps 4,768, 9,537, and 14,305."
- **Schedule.** Table 11: "constant-rate trunk, one decay branch per epoch"; "final 10% of each branch, linear to 0".
  §5.3 (p. 8) names a warmup-stable-decay schedule (Hu et al., 2024), following SmolLM2.
- **No feedback from branches.** The paper does not say this in words. It is implied by a constant-rate trunk that
  branches fork from. In code (C): the trunk segments write `trunk/`, each branch resumes from a fixed trunk step
  directory and writes its own `epN/` folder, and nothing reads `epN/` back.

## Hyperparameters: 1.5B (as run) vs 3B (proposed)

| | 1.5B | 3B proposed |
|---|---|---|
| architecture | Llama-2, 28 L, d 2048, 16 heads MHA, FFN 5632 SwiGLU, RoPE θ 1e4, RMSNorm 1e-5, tied (P, C) | 36 L, d 2560, 20 heads (head dim 128) MHA, FFN 6912 SwiGLU, otherwise identical (C) |
| parameters | 1,504,299,008 / 1,438,763,008 non-emb (P; V by count) | **2,936,855,040** / 2,854,935,040 non-emb (V by count, transformers LlamaForCausalLM on meta) |
| tokens/param | 30.0B / 1.50B = 20.0 (P) | 60.0B / 2.94B = 20.4 |
| vocab / tokenizer | 32000, Llama-2 (P); tokenizer.json sha256 81bb383c… (V) | same file (V: the 600M token counts used the same sha) |
| init | N(0, 0.02), row-parallel ×1/√(2L), shared per seed, weights-only into trunk1 (P, C) | same rule with L=36 (C: `scaling/parametrization.py`). One seed-42 init created once, sha-pinned, copied into all 3 trunks (C, tested) |
| context / packing | 2048; documents concatenated with EOS, windows cross document boundaries (P). Code: non-overlapping 2049-token windows, one window permutation per pass with data seed 42, repeated identically (C) | same |
| peak / min LR | 5e-4 / 0 (P, C) | **4e-4** / 0 |
| warmup | 500 steps linear (P, C) | **1000** steps linear (2.1B tokens) |
| stable + anneal | constant trunk; branches decay linearly to 0 over the last 10% (P). Trunk 4292/8583/12875 → branches end 4768/9537/14305 (P end steps, C branch points) | branch points **8583 / 17166 / 25749** → ends **9537 / 19073 / 28610**; decays 954 / 1907 / 2861. Same rule (registry_3b docstring) |
| AdamW | β (0.9, 0.95), wd 0.1, clip 1.0 (P); ε 1e-8 (C, not in paper) | same |
| precision | bf16 params, fp32 grads + optimizer state (P, C) | same |
| activation ckpt | not in paper. Original grid: none. Skipjack baselines: full layer recompute (C, V) | full layer recompute |
| optimizer sharding | ZeRO-0 (C) | **layout A: ZeRO-1** over DP=4 (fallback layout B: ZeRO-0), chosen by the smoke test |
| micro-batch × accum × DP | not in paper. Released grid: mbs 32 in 51/54 checkpoints (`tools/assert_invariants.py`). Skipjack: 32 × 8 × 4 (C, V) | A: **16 × 16 × 4**; B: 8 × 32 × 4. Identical for all 3 settings |
| global batch | 1024 seq × 2048 = 2,097,152 tokens/step (P, C, V) | same (1024 seq, 2,097,152 tokens) |
| horizons | ~10/20/30B; 14,305 steps ≈ 30.0B = 29,999,759,360 trained tokens (P) | 20,000,538,624 / 39,998,980,096 / **59,999,518,720** (plan.md decision 5 and the 05_final manifests: 20/40/60B) |
| checkpoint / resume | trunk every 1500 steps plus segment ends; branches save only their final; requeue-safe (C) | trunk every 1000 steps plus 6 segment ends; branch finals; same requeue logic |
| hardware / layout | 4 × H100 DP 4 (P). Released checkpoint configs show B300 (configs/know-your-sources-7b/README.md). Skipjack: 4 × H100 80GB (V) | exactly 4 × H100 80GB, DP 4, TP 1, PP 1, one node per task |
| throughput / memory | 16.1-16.2 s/it, 32.5k tok/s/GPU, 339 model-TFLOPs/GPU, peak 60.6 GiB (V, smoke 1115839) | ~31 s/it (E: FLOPs/token ×1.92 at equal MFU). Peak ~50 GiB in A, ~66 GiB in B (E, to be measured) |
| evaluation | LightEval, 6 commonsense tasks + MMLU, acc_norm, at each epoch horizon (P); kys-eval (C) | same protocol at the 9 finals (3 settings × ep1-3) |

## Learning rate: 4e-4 (recommended default, not proven optimal)
- **Anchors** (primary sources, read 2026-10-17):
  - This paper at 1.5B: 5e-4 at a 2.1M-token batch.
  - SmolLM2-1.7B: 5e-4 at 2M.
  - SmolLM3-3B: 2e-4 at 2.36M, but trained for 11.2T tokens with GQA; long horizons favour lower LR.
  - Llama-2 7B: 3e-4 at 4M.
  - OLMo 1B / 7B: 4e-4 / 3e-4 at ~4M.
  - Pythia-2.8B: 1.6e-4 at 2M (cosine, wd 0.01; conservative).
- **Relative scaling from the measured 1.5B recipe** (absolute fits don't transfer across setups):
  - DeepSeek LLM: η_opt ∝ C^-0.125. Compute grows 3.90× (C 2.96e20 → 1.16e21 with M = 72·L·d² + 12·L·d·s), so ×0.84 → 4.2e-4.
  - Porian et al. 2024: LR ∝ N^-0.36 at ~20 tokens/param. Non-embedding N grows 1.98×, so ×0.78 → 3.9e-4.
  - Standard-parameterisation width heuristic: 2048/2560 = 0.8 → 4.0e-4.
  - All three point to ~4e-4. DeepSeek's absolute formula would give 7.3e-4, but it would also give 8.6e-4 for the 1.5B
    run, which used 5e-4, so only its ratio is used.
- **Batch.** Kept at 2.1M tokens. DeepSeek's B_opt at this compute is ≈ 2.26M tokens, and the step-per-token
  granularity stays the same as at 1.5B.
- **Uncertainty.** A plausible range is 3e-4 to 5e-4, with no sweep and a single seed. The smoke test checks only
  stability, not optimality.
- **Fallback.** If the smoke test or the first trunk segment shows loss spikes or divergence, switch all three settings
  to 3e-4 before the first trunk advances past the problem: re-render, then restart all three trunk1 runs. The LR must be
  identical across settings.

## Unequal corpus sizes (accepted DIV shortfall)
- **Corpora** (corpus tokens include one EOS per document):

  | setting | corpus tokens | shortfall vs 20e9 |
  |---|---:|---:|
  | QF | 20,000,000,270 | — |
  | DIV | 19,834,314,433 | 165,685,567 |
  | DA | 20,000,000,227 | — |

  The rewritten part of DIV is 165,685,753 below 10e9. Four topic quotas (Literature, Religion, Adult, History) used
  every status-2 rewrite they had. The owner accepted this; nothing is backfilled or padded.
- **What the paper says.**
  - It states fixed horizons: "All six settings train for 14,305 steps … The training horizon is therefore identical
    across all settings" (p. 26), and "matched pretraining compute" (§4.1, p. 5).
  - It describes mixtures as "approximately 10B" and "around 5B" strategy tokens (§5.1, p. 7; Table 2), and defines
    topic quotas as a fixed share p_c·B_s (Eq. 6, p. 6).
  - It does **not** explicitly discuss quota shortfalls, backfill or padding, or the 1.5B DIV size (9.890B).
  - On consumption it says only "shuffled once at the document level … read in a fixed order, which repeats exactly in
    the second and third epochs" (p. 26).
- **What the implementation and plan say** (not the paper): plan.md §0 invariant 1, §1.9 and decision 5; the 05_final
  manifests (`passes_to_horizon` for DIV is 1.008 / 2.017 / 3.025); the Stage 04 "policy A". Together: shortfalls are
  allowed, there is no cross-topic backfill and no padding, horizons are matched in training tokens, and a smaller corpus
  is simply seen slightly more often.
- **How nanotron consumes a corpus** (C: `data/nanoset.py`, datatrove `DatatroveFileDataset`):
  - Horizons are fixed by optimizer steps, which means fixed processed tokens.
  - Each `.ds` file is read as non-overlapping 2049-token windows (2048 inputs + 1 shifted label). The file's tail
    under 2049 tokens is never read.
  - All windows of a pass are permuted once with data seed 42, and that permutation is cycled unchanged. There is no
    reshuffle between passes.
  - The last pass is a prefix of the permutation, and training stops at `train_steps`.
- **Recommendation.** Keep the paper's convention: identical optimizer steps for all three settings, so identical
  trained tokens at 20,000,538,624 / 39,998,980,096 / 59,999,518,720. Per pass, at most 9,760,859 windows for QF/DA and
  9,679,997 for DIV, so DIV runs about 3.026 passes by 28,610 steps versus 3.001 for QF/DA. `epoch_stats_3b.py` reports
  the exact numbers after tokenization.
- **Rejected: fixed epochs.** That would train DIV on 0.8% fewer tokens and break matched compute.

## Resources (E unless marked)
- **Per setting.** 25,749 trunk steps plus 5,722 branch steps = 31,471 steps. At 31 s/it that is ~271 h of 4-GPU time
  (~1,086 H100-h). At 35 s/it it is ~306 h.
- **Grid.** ~3,260 H100-h (3,670 at 35 s/it), plus ~54 h of segment overhead, the smoke test (~6 H100-h) and 9 evaluations.
- **Makespan.** 5 lanes × 4 GPUs, peak 20 GPUs (V by the plan simulation): ≈ 250 h (10.4 d) at 31 s/it, 282 h at 35 s/it.
  The longest segment is 4292 steps, which fits the 48 h job limit up to ~38 s/it; above that the submitter refuses and
  the trunk must be split further.
- **Storage.** One checkpoint ≈ 41 GB (bf16 weights + fp32 master + Adam). Trunks hold 31 checkpoints per setting,
  ≈ 3.8 TB in total, plus 0.37 TB of branch finals and ~120 GB of tokenized data. /weka/projects has 93 TB free (V).
