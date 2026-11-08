# Training-protocol sub-audit (configs and nanotron internals): findings as reported

These are the findings of the training sub-audit. The main session saved them, with the check of F1 recorded at the
end.

**Scripts in this folder:**
- `cmp.py`: template and v2 config diff;
- `flat.py`: diff across the v2 configs;
- `lrsim.py`: LR-schedule simulation;
- `samp.py`: sampler enumeration;
- `dl.py`: v2 and init metadata download.

**Data in this folder:**
- `v2_configs/`: the 54 v2 `config.yaml` at `wytro/KYS-1.5B-Rewritten-v2@b22c8733`;
- `init_meta/`: init metadata at `wytro/Know-Your-Sources-init@87e1335`.

## Template comparison (Part C)

**Method:**
- All 54 templates were diffed within each segment kind. Only `general.seed` and `data_stages[0].data.seed` differ;
  both equal the chain's seed.
- The shipped placeholder copies match their templates on every key.
- The rendered ep configs were parsed with this repo's `get_config_from_file` and diffed against the matching v2
  config (diversity_oriented or disagreement_aware, 3 seeds × 3 epochs).
- With run names and paths removed, only the runtime counters `general.step` and `general.consumed_train_samples`
  differ. The dp8 render additionally differs only in dp 8 / accum 4.
- Across the 54 v2 configs, the only variation outside the schedule is quality_base seed 42 at mbs 16 / accum 16.

| field | Table 11 | templates (54, rendered) | v2 | verdict |
|---|---|---|---|---|
| layers / hidden / FFN | 28 / 2048 / 5632 SwiGLU | 28 / 2048 / 5632, silu | same | match |
| heads / kv heads | 16, MHA | 16 / 16 | same | match |
| vocab / tied embeddings | 32000 / yes | 32000 / true | same | match |
| RoPE θ / RMSNorm ε | 1e4 / 1e-5 | 10000.0 / 1e-5 | same | match |
| AdamW betas / eps | (0.9, 0.95) | 0.9, 0.95 / 1e-8, fused | same | match |
| weight decay / clip | 0.1 / 1.0 | 0.1 (no exclusions) / 1.0 | same | match |
| peak LR / warmup | 5e-4 / 500 linear | 5e-4 / 500 linear | same | match |
| trunk ends | 4292 / 8583 / 12875 | train_steps 4292 / 8583 / 12875; trunk lr_decay_starting_step 12875 | not published | match (v2 side unverified) |
| branch decay | 4292→4768, 8583→9537, 12875→14305, linear to 0 | starts 4292 / 8583 / 12875; steps 476 / 954 / 1430; linear; min 0.0 | same | match |
| branch resume = decay start | – | `<trunk>/<lr_decay_starting_step>` (`render_config.py:320-322`; checked at `assert_invariants.py:394-399`) | `.../trunk/4292` etc. | match |
| seq / global batch | 2048 / 1024 | 32×8×4 (dp8: 32×4×8) = 2,097,152 tokens/step | 32×8×4 (QB s42: 16×16×4) | match |
| precision | bf16 params, fp32 grads and optimizer state | bfloat16, accumulate_grad_in_fp32 true | same | match |
| zero stage | – | 0 | 0 | match |
| recompute_layer | – | set by the cluster entry | false | does not change the math |
| init | per seed, shared across settings | `plan_submit.py:121-135` copies `_init_1.5B_seed<S>/0` into every setting's trunk | – | match by construction |
| data seed / shuffle | none | seed = S, but `shuffle_files` is false (`config.py:165`); the seed is used only for file shuffling (`tokenized_bytes.py:252-256`) | same | has no effect |
| checkpoint interval | – | trunk 1500 / branch 100000 | branch 100000 | match |
| load_optimizer / load_lr_scheduler | – | trunk1 false/false; all others true/true | ep: true/true | match; F1 |

## Findings

| ID | file:lines | requirement | actual | severity | evidence | fix |
|---|---|---|---|---|---|---|
| F1 | `trainer.py:231-259, 1046-1058`; `serialize/main.py:216-228`; `generate_configs.py:175-176`; `assert_invariants.py:378-391`; `kys_segment.sbatch:35,50` | A requeued trunk loses at most 1500 steps (WORKFLOW §9, RUNBOOK step 7) | trunk1 restarted after its 1500 or 3000 save loads those weights, but with load flags false it starts fresh metadata at step 0, a fresh AdamW, warmup again and data from sample 0. It writes a "4292" checkpoint that has seen up to about 7000 steps. `--check-resume` passes. | High | code | trunk1 fails closed when latest.txt ≠ 0, or loads state when latest.txt > 0; fix the docs |
| F2 | v2 `seed42/quality_base/ep{1,2,3}` config.yaml | same mbs as the other comparators | mbs 16 / accum 16 at dp 4 (the other 51 use 32/8). Only cells paired with QB seed 42 are affected; none of the rebuttal comparisons. Documented in WORKFLOW §13 and RUNBOOK step 3. | Note (documented) | artifact | none for the rebuttal; the Table 11 footnote "optimization therefore identical" is inaccurate (Low) |
| F3 | `plan_submit.py:64`; `registry.py:55-58`; `fill_placeholders.py:44` | only the 3 settings are queued | Omitting `--settings` defaults to all 8 settings (48 jobs per seed). It fails safe if the other filled configs are missing, and submits them if they exist. The WORKFLOW §9 and RUNBOOK step 6 examples omit `--settings`. | Medium | code + documentation | require `--settings`; fix the examples |
| F4 | `serialize/main.py:229-235`; `trainer.py:1058-1063`; `assert_invariants.py:357-408`; `kys_segment.sbatch:21,50,57` | a wrong resume path must not random-init silently | Guarded before torchrun, under `set -euo pipefail`. The remaining gap is F1. | Note | code | via F1 |
| F5 | `llama.py:747,773,781`; dropout 0 at `:298,574` | recompute does not change the math | reentrant checkpoint, RNG preserved, no dropout | Note | code | none |
| F6 | `serialize/optimizer.py:360-370`; torch `lr_scheduler.py:143-147`; `helpers.py:97-161`; `trainer.py:726-731` | branch decay starts at the resume step | A resumed branch uses λ(t): ep1 step 4293 = 475/476 of peak, step 4768 = 0. Same code path as v2. | Note | code + simulation | none |
| F7 | `kys_segment.sbatch:25` | the audited code is what runs | Nothing asserts that the imported nanotron is under `$REPO`; the local env imports the other clone. `repo_dir` is checked on marc-cluster. | Low | code | assert `nanotron.__file__` is under `$REPO` |
| F8 | `fill_placeholders.py:67` vs `render_config.py:175-185` | mbs is uniform across clusters | The cross-cluster mbs check only ever sees one cluster; uniformity rests on assert_invariants pinning mbs 32. | Note | code | keep mbs 32 in the dp8 entry |
| F9 | – | trunk fields match v2 | v2 publishes only ep configs | Unverified | – | – |
| F10 | `plan_submit.py:122-135` | init = v2 init | Checks only that there are 180 files; no hash check | Low | code | optionally hash-check |

## dp 8 / mbs 32 / accum 4 vs dp 4 / mbs 32 / accum 8

They produce identical micro-batches and identical per-token weighting.
- `MegatronPretrainingSampler` (`samplers.py:188-224`) gives rank d the slice [d·32, (d+1)·32) of each block of
  mbs·dp consecutive samples. Both layouts therefore produce the same 32 contiguous 32-sample micro-batches per step.
  This was verified by enumeration at consumed = 0, 4292·1024 and 8583·1024.
- The loss is `masked_mean` per micro-batch (`llama.py:984-1006`), divided by accum (`engine.py:53-55`).
- Gradients are summed in fp32 (`gradient_accumulator.py:215-225, 343-390`), then AVG all-reduced over dp
  (`helpers.py:459-461`).
- Every micro-batch gets weight 1/32 in both layouts. Only the fp summation order differs.

## Init metadata (`wytro/Know-Your-Sources-init@87e1335`)

- 180 files per seed.
- `model_config.json` is identical across seeds 42, 43 and 44 (md5 d3105d2e…).
- dp 4, last_train_step 0, NUM_LAYERS scaling, std 0.02, bf16.
- The optimizer state is empty.
- The stage name differs by seed (seed 42 `S0_top10B`, seeds 43/44 `S0_init`). This is harmless: trunk1 loads no
  metadata.

## Not verified

- v2 trunk configs (not published).
- The v2 code and torch versions.
- That the HF init weights equal the v2 init: metadata only, no weight hash.
- GPU types.
- Behavior on a checkpoint with missing weight files.
- SLURM requeue and afterok behavior on the collaborator's cluster.
- Tokenizer identity beyond vocab size (`run_train.py:195-197`).

## Main-session check of F1 (2026-11-07)

- **Code, confirmed:** `generate_configs.py:175-176`; `trainer.py:240-259`, where metadata is loaded only if
  `load_lr_scheduler`; `trainer.py:1046-1058`, where weights load from `init_checkpoint_path`;
  `assert_invariants.py:378-391`.
- **Prior record:** the same latent risk was flagged on skipjack on 2026-10-27 and left unfixed.
- **Severity raised to a launch blocker (B1):** WORKFLOW §7 allows `slurm.time` ≥ 12 h, while trunk1 takes 27.2 h, and
  WORKFLOW §9 calls resubmission safe. A restart of trunk1 is therefore likely rather than exceptional.
