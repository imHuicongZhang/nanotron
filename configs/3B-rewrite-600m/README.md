# 3B rewrite-600m grid: design record (revised 2026-10-17, NOT launched)

> Historical Skipjack design record. The released grid has FOUR settings (quality-base added 2026-10-30); its
> scientific values are now defined once in `deploy/portable_3b/science_3b.py` (imported by `registry_3b.py`, which
> keeps only Skipjack paths/site). Running it elsewhere: `deploy/portable_3b/README.md`.

Three settings (quality-first, diversity-oriented, disagreement-aware) trained on the validated 600M-based
corpora at 3B scale. The grid uses seed 42, one shared init, and a WSD trunk with three decay branches.
- Code: worktree `projects/nanotron-kys-3b600m`, branch `rewrite3b-600m-train`.
- Base: **bc131392**, the revision the active 1.5B baselines run from (`projects/nanotron-kys`, left untouched).
- Every value lives in `deploy/skipjack_3b_600m/registry_3b.py`.

**Architecture rule (owner decision, 2026-10-17):** the 3B model is the 1.5B LLaMA-2-style model with **only
`num_hidden_layers` changed**, and it uses the exact 1.5B tokenizer files. `assert_invariants_3b.py` re-reads the 1.5B
config (read-only) and refuses any difference in the `model` block other than the layer count. That includes width,
heads, FFN, attention type, vocab, tied embeddings, RoPE, norm, dtype and the init settings.

Labels: **P** = stated in the paper. **C** = set in code or configs. **V** = verified by measurement or count.
**E** = engineering estimate.

## Architecture, 1.5B vs 3B
Sources for the 1.5B column: `configs/1.5B-baseline-seed42/filled/*` (C), the published checkpoints'
`model_config.json` under `checkpoints/rewrite-1p5b/` (V), and Table 11 of the paper (P).

| field | 1.5B | 3B |
|---|---|---|
| family | LlamaForCausalLM (nanotron `is_llama_config`) | same |
| num_hidden_layers | 28 | **57** |
| hidden_size | 2048 | 2048 |
| attention | 16 heads, head dim 128, MHA (16 KV heads), no bias | same |
| FFN | 5632, SwiGLU (silu), no bias | same |
| vocab / tokenizer | 32000, Llama-2 tokenizer | same files (below) |
| tied embeddings | yes | yes |
| positions | RoPE θ = 10000, not interleaved, max 2048 | same |
| norm | RMSNorm ε = 1e-5 (pre-norm) | same |
| bos / eos / pad | 1 / 2 / none | same |
| dtype | bfloat16 | same |
| init | N(0, 0.02); `o_proj` and `down_proj` N(0, 0.02/√(2L)) with L = 28 → **0.0026726**; RMSNorm weights 1 | same rule at L = 57 → **0.0018732** |
| total parameters | 1,504,299,008 (P 1.504B; V) | **2,994,444,288** (V) |
| non-embedding parameters | 1,438,763,008 | 2,928,908,288 |
| tokens per parameter at the last horizon | 29,999,759,360 / 1.504B = 19.94 | 59,999,518,720 / 2.994B = 20.04 |

**Parameter counts** (V: transformers LlamaForCausalLM on the meta device, matching the closed form):
- Per layer: 4·2048² + 3·2048·5632 + 2·2048 = **51,384,320**.
- Fixed part: tied 32000×2048 embedding + final norm = **65,538,048**.

| layers | total | distance from 3e9 |
|---|---:|---:|
| 54 | 2,840,291,328 | −159.7 M |
| 55 | 2,891,675,648 | −108.3 M |
| 56 (= 2×28) | 2,943,059,968 | −56.9 M |
| **57** | **2,994,444,288** | **−5.6 M** |
| 58 | 3,045,828,608 | +45.8 M |
| 59 | 3,097,212,928 | +97.2 M |
| 60 | 3,148,597,248 | +148.6 M |

Doubling the layers gives 2.943B, because the 65.5M fixed part does not scale with depth.

**Init (C: `scaling/parametrization.py`, StandardParametrizator):**
- `init_method.scaling_method` defaults to `NUM_LAYERS`, and the configs set only `std: 0.02`, exactly as at 1.5B.
- Column-parallel weights (q, k, v, gate/up) and the tied embedding are N(0, 0.02).
- Row-parallel outputs (`o_proj`, `down_proj`), which write into the residual stream, are N(0, 0.02/√(2L)). L is read
  from `num_hidden_layers`, so at 57 layers the factor is 1/√114. This keeps the residual-stream variance added at init
  comparable with depth, as the 1.5B rule intends.
- RMSNorm weights are 1. No other depth-dependent factor exists in this code path.

**Tokenizer** (V, sha256): the 3B tokenizer is a copy of the 1.5B data's tokenizer directory
`data/1p5_baseline/pretraining/raw/tokenizer`, pinned by the KYS-Pre-Rewritten `manifest.json`:

| file | sha256 |
|---|---|
| `tokenizer.json` | `81bb383c…` |
| `tokenizer_config.json` | `2bffa414…` |
| `tokenizer_report.json` | `4dcd1212…` |

- The same `tokenizer.json` was used for the 600M token counts and is in the published 1.5B HF exports.
- The HF exports' `tokenizer_config.json` differs only in `is_local` / `local_files_only`.
- `tokenize_3b.sbatch` refuses any tokenizer directory whose file set or hashes differ.

## Training hyperparameters
| | 1.5B (as run) | 3B (proposed) |
|---|---|---|
| peak / min LR | 5e-4 / 0 (P, C) | **4e-4** / 0 (unchanged recommendation; see below) |
| warmup | 500 updates (P, C) | 1000 updates (2.10B tokens; the same 10.49% of the first horizon) |
| schedule | constant trunk; branches 4292/8583/12875 → 4768/9537/14305, linear to 0 over the final 10% (P, C) | branches 8583/17166/25749 → 9537/19073/28610 (same rule) |
| AdamW / wd / clip | (0.9, 0.95), ε 1e-8, 0.1, 1.0 (P, C) | same |
| precision | bf16 params, fp32 grads + optimizer state (P, C) | same |
| global batch | 1024 × 2048 = 2,097,152 tokens/update (P, C, V) | same |
| micro-batch × accum × DP | 32 × 8 × 4, ZeRO-0, full recompute (C, V) | A: 16 × 16 × 4, ZeRO-1 (default). B: 8 × 32 × 4, ZeRO-0. C: 32 × 8 × 4, ZeRO-1 (optional; the 1.5B micro-batch). All with full recompute; the smoke test chooses |
| horizons | 9,999,220,736 / 20,000,538,624 / 29,999,759,360 | 20,000,538,624 / 39,998,980,096 / 59,999,518,720 |

**Resume fix (2026-10-17, this worktree only).** nanotron's resume path (`load_state_dict` + `_initial_step`) shifted
the scheduler index by +1 per resume, cumulatively (with one requeue: ep1 starts decaying at update 8584 instead of
8585, the trunk dips below peak before 25749, ep3 spends its last 7 updates at LR 0). `load_lr_scheduler` now sets the
next update's LR from the restored index without stepping, so every resumed or branched update n uses λ(n−1), exactly
as one uninterrupted run. The decay rule itself (`lr_scheduler_builder`) is unchanged. Test:
`deploy/skipjack_3b_600m/test_lr_resume_3b.py`; report: `training/reports/lr_resume_fix_20261017/`. The 1.5B
baselines still run the unfixed code, so their branches end with 1/2/3 updates at LR 0 (plus 1 per requeue).

## Learning rate for the deeper same-width model: keep 4e-4 (not proven optimal)
The previous width-based argument no longer applies, because the width is unchanged. Re-derived relative to the
measured 1.5B recipe (5e-4 at 2.1M tokens/update):
- **Compute scaling (DeepSeek LLM, η_opt ∝ C^-0.125).** M = 72·L·d² + 12·L·d·s: 9.865e9 → 20.08e9 FLOPs/token.
  C goes from 2.96e20 to 1.205e21 (×4.07), giving ×0.839 → **4.2e-4**.
- **Size scaling (Porian et al. 2024, LR ∝ N^-0.36 at ~20 tokens/param).** Non-embedding N ×2.036 gives ×0.774 →
  **3.9e-4**.
- **Width heuristic** (standard parameterization, LR ∝ 1/width for hidden matrices): ×1.0 → **5e-4**.
- **Depth** (engineering judgment): the model doubles in depth at fixed width, with no depth-wise residual multipliers
  beyond the 1/√(2L) init. Two effects:
  - More residual branches add updates to the same stream, so per-step function change grows with depth at a fixed LR.
  - Deep, narrow models are generally the more instability-prone direction.
  - This argues against raising the LR above the fits.
- **Recommendation: 4e-4.** It sits inside the 3.9-4.2e-4 band of the two size/compute fits and below the width-only
  5e-4. Plausible range 3e-4 to 5e-4. Fallback: 3e-4 for all three settings if the smoke test or the first trunk segment
  shows loss spikes or divergence.
- **Alternative scientific choice: 5e-4.** This holds every training hyperparameter fixed and changes only depth.
  It's a cleaner "depth-only" scale comparison, at somewhat higher instability risk.
- A smoke test doesn't establish the optimal LR or long-run stability.

**Global batch.** Unchanged. DeepSeek's B_opt at C = 1.2e21 is ≈ 2.3M tokens, close to 2.1M.

## Memory on 4 × H100 80GB (E; the smoke test measures them)
- **Static memory** per GPU for 2.994B parameters: ZeRO-0 at ~20 B/param ≈ 55.8 GiB; ZeRO-1 (master + Adam sharded
  over 4) ≈ 30.7 GiB. Add ~2.5 GiB of CUDA/NCCL.
- **Activations with full recompute.** The 1.5B smoke measured 60.6 GiB at mbs 32, about 0.94 GiB/sequence. At 57
  layers the stored layer inputs double (0.22 → 0.45 GiB/sequence), and the per-layer working set and logits are
  unchanged: ≈ 1.17 GiB/sequence.

| layout | estimated peak (budget 74.5 GiB) |
|---|---:|
| A | ≈ 52 GiB |
| B | ≈ 68 GiB |
| C | ≈ 70.5 GiB (tight) |
| ZeRO-0 at mbs 16 | ≈ 77 GiB (does not fit) |

## Throughput and runtime (E)
- **Per-token compute.** FLOPs/token incl. vocab grow 10.26e9 → 20.48e9 (×1.996). The layer matrix shapes are
  identical to the measured 1.5B run (16.1 s/it, 32.5k tokens/s/GPU), so equal efficiency is the natural assumption:
  ≈ **32.2 s/it**. Expected range 32-35, since ZeRO-1 adds a reduce-scatter/all-gather per update and mbs 16 runs
  slightly smaller kernels than mbs 32.
- **Per setting.** 31,471 updates (25,749 trunk + 954 + 1,907 + 2,861 branch) ≈ 281 h of 4-GPU time ≈ 1,126 H100-h
  (306 h / 1,224 H100-h at 35 s/it).
- **Grid.** ≈ 3,380 H100-h (3,670 at 35 s/it), plus ~54 H100-h of segment overhead, the smoke test (~6-8 H100-h) and
  9 evaluations.
- **Makespan.** 5 lanes × 4 GPUs, peak 20 (V, plan simulation): ≈ 259 h (10.8 d) at 32.2 s/it, 282 h at 35 s/it.
- **Job limit.** The longest segment (4292 updates) needs 42.7 h at 32.2 s/it and 46.4 h at 35 s/it, within the 48 h
  job limit. The submitter refuses above ~38 s/it.
- **Storage.** One checkpoint ≈ 41.9 GB. Trunks ≈ 3.9 TB, branch finals 0.38 TB, tokenized data ~120 GB.
  /weka/projects has 93 TB free.

## Unequal corpus sizes (accepted DIV shortfall)
Unchanged from the previous record:
- Identical optimizer steps, i.e. identical processed tokens, for all three settings.
- No backfill and no padding.
- DIV makes ≈ 3.026 physical passes by update 28,610, versus ≈ 3.001 for QF/DA. These are upper-bound window counts;
  `epoch_stats_3b.py` gives exact values after tokenization.
- The paper states fixed horizons (App. E, p. 26) but doesn't discuss quota shortfalls. That policy comes from the plan
  and implementation (plan.md decision 5, 05_final manifests).
