# Know Your Sources - 1.5B rebuttal set - Marc execution workflow

For Marc and the agent operating Marc's training infrastructure. This document covers the ICLR 2027 rebuttal experiments: **three settings, three seeds, nine training chains, 27 annealed final models**, trained and evaluated under the exact protocol of the published 1.5B grid so that every result can be paired within seed against the published v2 checkpoints.

The repository documents `configs/1.5B-baseline/RUNBOOK.md` and `configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md` are the authoritative references for exact commands and flags. This document states what is fixed, what Marc fills in, the order of work, and the completion criteria. Where the two disagree, RUNBOOK and WORKFLOW win; report the disagreement.

Marc supplies hardware, allocation, filesystem paths, scheduler details and credentials. No GPU count is prescribed, but the schedule in §7 assumes 32 GPUs for wave 1.

## 1. Objective, ownership, and fixed release

Huicong has published the validated inputs, training code and evaluation code. Marc's agent downloads them, fills the cluster profile, tokenizes the two raw-text settings, trains the nine chains, converts and uploads the 27 finals, scores them with the pinned evaluation harness, and returns the per-checkpoint result files. Keep progress durable across session disconnects.

| Artifact | Exact source |
|---|---|
| Training code | https://github.com/imHuicongZhang/nanotron, branch `huicong-dev` |
| **Training commit** | **`c25f7a01` or later on `huicong-dev`.** Training code is identical since `6fe10b9b130a22338bfc1c2d199761131b711a88`; every later commit changes documentation and audit records only, confirmed by audit. Record the SHA used. |
| Evaluation code | https://github.com/imHuicongZhang/kys-eval, branch `main` |
| **Evaluation commit** | **`6da264bf` or later** (current `00bbca34`). Scoring code is unchanged since `216c1ba`; `6da264bf` adds the full-precision v2 reference, the `quality_base` pairing and the Appendix F sign convention, all of which the rebuttal needs. |
| Data repository | https://huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten, public |
| **Dataset revision** | **`afd1f177`**. Use that pin, not `main`. |
| Initial weights | https://huggingface.co/wytro/Know-Your-Sources-init at **`87e1335647a3105471928e0fd261d92846021d1a`**, public |
| Comparator checkpoints | https://huggingface.co/wytro/KYS-1.5B-Rewritten-v2 at `b22c87333783945e1c655ae46886b7e842ee4203`, public. Used only by the evaluation reference check. |
| Model output repository | https://huggingface.co/blab-jhu/KYS-1.5B-Raw-Selected-Baselines, **gated**. Requires a token with write access to `blab-jhu`. |
| Operator guide | `configs/1.5B-baseline/RUNBOOK.md` |
| Full guideline and execution plan | `configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md`, especially §1b |
| Prelaunch audits | `docs/prelaunch_audit_2026-11-07.md`, `docs/prelaunch_audit_2026-11-08.md` |

Do not change scientific settings to ease deployment. Record any local deployment patch as a separate commit and report it.

## 2. Scientific contract

Train exactly these three settings at seeds **42, 43, 44**. The five other raw settings in the repository are not part of this set and must not be queued.

| Setting | What it is | Published form | TRAIN tokens | Shards after tokenization |
|---|---|---|---:|---:|
| `raw_diversity_oriented` | Raw control for `diversity_oriented`: shared 5B fastText anchor plus 5B raw tokens sampled at the document level from that strategy's source documents | `raw_text/raw_diversity_oriented/`, 16 parquet files, consumer-tokenized | 10,000,002,885 | 16 |
| `raw_disagreement_aware` | Raw control for `disagreement_aware`, same construction | `raw_text/raw_disagreement_aware/`, 16 parquet files, consumer-tokenized | 10,000,005,069 | 16 |
| `signal-disagreement-lambda0` | Disagreement-Aware with λ = 0: same candidate domain, quality floor, variance cap, anchor and rewriting procedure as the published λ = 0.5 arm; only the ranking changes | `signal-disagreement-lambda0/tokenized/`, 26 datatrove shards, already tokenized | 10,000,002,556 | 26 |

Each corpus already contains its shared anchor and is already shuffled. Do not append another base, re-select, re-shuffle or re-tokenize the λ = 0 data. The λ = 0 total is 2 tokens below its 10,000,002,558 assembly budget because two empty rewrites tokenize to nothing; this is expected.

Model and optimization, identical to the published grid (paper Table 11):

- Llama 2 architecture, 28 layers, hidden 2048, FFN 5632 SwiGLU, 16 heads, RoPE θ = 10⁴, RMSNorm, vocabulary 32000, tied embeddings, 1.504B parameters.
- Published Llama 2 tokenizer from the dataset's `tokenizer/` folder. Tokenization convention: text tokens plus one `</s>` per document, no `<s>`.
- AdamW β = (0.9, 0.95), weight decay 0.1, gradient clip 1.0, peak LR 5e-4, 500-step linear warmup once.
- Warmup-stable-decay. Constant-rate trunk with segment ends at **4292, 8583, 12875**. Three decay branches from fixed trunk checkpoints, each decaying linearly to zero over the last 10% of its horizon, ending at **4768, 9537, 14305**.
- Sequence length 2048, global batch 1024 sequences, **2,097,152 tokens per optimizer update**. bf16 parameters, fp32 gradients and optimizer state.
- **dp 4, micro-batch 32, accumulation 8**, four GPUs per chain, one node. The only permitted alternative is wave 2's λ = 0 seed 44 at dp 8, micro-batch 32, accumulation 4 on a single 8-GPU node (§7). Micro-batch must stay 32 in both; it is part of the loss weighting.
- Layer recomputation enabled, as in the published grid.
- Every chain starts from the published per-seed initialization in `wytro/Know-Your-Sources-init`, identical across settings for a given seed. Never start from an earlier partial run or a fresh random init.

## 3. What Marc fills in

Add a cluster entry to `deploy/clusters.yaml`, named `marc-cluster`, and fill every null placeholder. If an 8-GPU node is available for wave 2, add `marc-cluster-8gpu` as well; WORKFLOW §1b shows both.

| Group | Operator inputs |
|---|---|
| Paths | `DATA_ROOT` (the downloaded dataset root), `CHECKPOINT_ROOT`, `LOG_ROOT`, `INIT_ROOT`, W&B directory. All on shared storage visible from the compute nodes; `assert_invariants.py` rejects node-local paths such as `/tmp`. |
| Environment | Absolute path to the training interpreter |
| Deployment | `RECOMPUTE_LAYER` (true, matching the published grid), dp 4 for the main profile, dp 8 for the optional 8-GPU profile |
| Scheduler | Slurm account, partition, QoS, time limit, CPUs and memory per segment |
| Credentials | A Hugging Face token that can read and write `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`. Keep it in the environment, never in configs, logs or commits. |

Use literal absolute paths without whitespace. Do not fill in publisher paths from Skipjack. Do not change global batch, LR, horizons or micro-batch to fit an allocation.

## 4. Install the pinned environments

Two environments are needed: the Nanotron training environment from the repository's requirements, and the kys-eval environment with its pinned LightEval fork. Follow RUNBOOK for the training environment and the kys-eval README for the evaluation environment. Tokenization needs datatrove 0.5.0, as RUNBOOK specifies.

```bash
git clone --branch huicong-dev https://github.com/imHuicongZhang/nanotron.git <<CODE_ROOT>>
cd <<CODE_ROOT>> && git log -1 --format=%H    # record the SHA
git clone https://github.com/imHuicongZhang/kys-eval.git <<EVAL_ROOT>>
cd <<EVAL_ROOT>> && git checkout 6da264bf      # or the later revision Huicong names
```

Run the kys-eval self-test after installation. It must pass before any scoring.

## 5. Download, verify, tokenize, render

Download the dataset at the pinned revision into `DATA_ROOT`, including `manifest.json`, `tokenizer/`, `raw_text/raw_diversity_oriented/`, `raw_text/raw_disagreement_aware/` and `signal-disagreement-lambda0/`. The other five `raw_text/` settings are not needed. Download the three per-seed init checkpoints from `wytro/Know-Your-Sources-init` at `87e1335` into `INIT_ROOT` and verify them:

```bash
python tools/hash_init_checkpoint.py <<INIT_ROOT>>/_init_1.5B_seed<<S>>/0 --check <<INIT_ROOT>>/init_1.5B_seed<<S>>.hash.json   # for S in 42 43 44
```

Verify the raw-text parquet files against the sha256 values in `manifest.json`.

For λ = 0, **run the checksum check before `fix_ds_metadata.py`**, because that script rewrites metadata files:

```bash
cd <<DATA_ROOT>>/signal-disagreement-lambda0/tokenized && sha256sum -c SHA256SUMS
python tools/fix_ds_metadata.py --tokenizer-dir <<DATA_ROOT>>/tokenizer <<DATA_ROOT>>/signal-disagreement-lambda0/tokenized
```

For the two raw settings, tokenize with the repository script, which uses 16 tasks, one `</s>` per document, no BOS and no shuffling, then fix metadata the same way:

```bash
bash tools/kys_raw/tokenize_raw_text.sh raw_diversity_oriented <<DATA_ROOT>>
bash tools/kys_raw/tokenize_raw_text.sh raw_disagreement_aware <<DATA_ROOT>>
```

The exact flags are in RUNBOOK. After tokenization the totals must be exactly 10,000,002,885 and 10,000,005,069 tokens in 16 shards each, and λ = 0 must be exactly 10,000,002,556 in 26 shards.

Render and check the configs for the three settings only. `--settings` is mandatory on both tools; omitting it is an error, and `--settings all` would plan five settings that are not in this set.

```bash
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --settings raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0 --seeds 42,43,44
python tools/assert_invariants.py <<rendered config dir>>                 # expect 54/54 pass
python tools/assert_invariants.py --check-resume <<rendered config dir>>  # expect 54/54 pass
```

Keep the rendered configs, the invariant reports and the tokenization logs with the run records.

## 6. Dependency graph

Each chain is six Slurm segments:

```
trunk1 (0 → 4292) → trunk2 (→ 8583) → trunk3 (→ 12875)
ep1 branch: from trunk checkpoint 4292 → 4768
ep2 branch: from trunk checkpoint 8583 → 9537
ep3 branch: from trunk checkpoint 12875 → 14305
```

Branches continue from the fixed trunk checkpoint directory at the branch point, not from a moving `latest.txt`, and write to their own directories. Their final checkpoints are the 27 models to evaluate. Preserve the three branch-point trunk checkpoints of every chain until its branches and their verification are complete.

Trunk warmup happens once at step 0. A trunk segment that restarts from a checkpoint past step 0 must load weights, optimizer state, LR scheduler state and data position. `deploy/slurm/kys_segment.sbatch` does this through `tools/kys_raw/trunk_restart_config.py` when `latest.txt` is past 0, and `assert_invariants.py --check-resume` refuses the unsafe case. **If any trunk1 log shows training restarting at iteration 1 after a requeue, stop that chain and report it; the result would be invalid.**

## 7. Submit in two waves, monitor durably, recover precisely

The execution plan is WORKFLOW §1b. With 32 GPUs:

**Wave 1, eight concurrent chains, dp 4 / mbs 32 / accum 8, four GPUs each:** `raw_diversity_oriented` and `raw_disagreement_aware` at seeds 42, 43, 44, plus `signal-disagreement-lambda0` at seeds 42 and 43. Seeds run concurrently in this set; this overrides the one-seed-at-a-time rule used for the full grid.

**Wave 2, after wave 1 frees GPUs:** `signal-disagreement-lambda0` seed 44. On one 8-GPU node use dp 8 / mbs 32 / accum 4 through the `marc-cluster-8gpu` profile; otherwise dp 4 on four GPUs. Tokens per step are identical either way.

Submit with `plan_submit.py`, always with explicit `--settings`:

```bash
# wave 1
python tools/kys_raw/plan_submit.py --cluster marc-cluster --settings raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0 --seeds 42,43
python tools/kys_raw/plan_submit.py --cluster marc-cluster --settings raw_diversity_oriented,raw_disagreement_aware --seeds 44
# wave 2, after wave 1 frees GPUs
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --settings signal-disagreement-lambda0 --seeds 44
```

Never pass `--settings all`. If wave 2 runs at dp 8, report that as a deviation.

Each segment's sbatch runs `assert_invariants.py` before training and refuses to start on a failed check. Segments chain with `afterok` dependencies. Save every job ID as it is issued. Maintain a single submitter; after a disconnect, reconcile queue state against checkpoints and submit only genuinely missing segments. Do not rerun a submission script wholesale.

Before the first full submission, run the 20-step smoke run described in WORKFLOW §1b on one chain to measure seconds per step, confirm finite loss, the expected LR at step 1, consumed-token accounting and checkpoint writes. Report the measured s/it; Huicong's cluster measured about 100 hours per chain on four H100s with this configuration, roughly 400 H100-hours per chain.

On failure, read the log, resume from validated state, and never alter LR, horizon or micro-batch to recover.

## 8. ETA and completion criteria

Nine chains at about 100 hours each, 897 node-hours on 4-GPU nodes. With wave 1 fully concurrent, wave 1 finishes about 4.2 days after launch and wave 2 about 2 to 4 days later depending on dp. **The rebuttal discussion closes November 18. Results must be scored and returned by November 15 to be usable. If wave 2 cannot finish by then, wave 1 alone is still a complete deliverable and Huicong will report λ = 0 at two seeds.**

Training completion means all 27 final checkpoint directories exist in the layout RUNBOOK specifies under `CHECKPOINT_ROOT`, `rewrite-1p5b/seed<S>/<setting>/ep<N>/...`, each with model tensors, config, metadata, optimizer and LR state, the correct final step (4768, 9537 or 14305) and a clean final log. A folder name or an empty queue is not completion.

## 9. Export and publish the 27 models

Convert each final with the repository's CPU converter and confirm the export loads with 28 layers, 1.504B parameters, tied embeddings and the published tokenizer:

```bash
python tools/kys_eval/convert_to_hf.py --checkpoint_path <<final checkpoint dir>> --save_path <<EXPORT_ROOT>>/rewrite-1p5b/seed<<S>>/<<setting>>/ep<<N>>/hf --tokenizer <<DATA_ROOT>>/tokenizer --dtype bfloat16
```

Upload to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` with the authorized token, in exactly this layout, which is what kys-eval reads:

```
rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/
rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/      (native final, optional but preferred)
rewrite-1p5b/seed<S>/<setting>/logs/
```

Record the Hub revision after each upload. **Never re-upload a checkpoint to the same path without telling Huicong**: the evaluation cache cannot detect a same-path re-upload and must then be run with `--force`. Pin the final revision in the report.

## 10. Evaluation

Use kys-eval at the pinned revision and score on **H100 80GB HBM3 (SXM)**, the GPU the v2 reference numbers were produced on; the harness accepts any GPU whose name contains "H100", so this is on the operator. First run the reference check, which re-scores a published v2 checkpoint and must print PASS before any new model is scored:

```bash
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203
```

Then score the 27 rebuttal cells only. The SLURM array refuses to run without `KYS_GRID_ARGS`; use the 27-cell filter and array range given in WORKFLOW §14 (`--array=0-26`). `KYS_GRID_ARGS=all` is the 72-cell full grid and is not wanted here. Protocol, fixed by the harness: six commonsense tasks (ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA), MMLU over 57 subjects, zero-shot, full splits, `acc_norm`.

Run `aggregate.py` for a first look, but the deliverable is the **27 per-checkpoint `results.json` files**, unmodified, plus the reference-check output. Upload them to the model repository under one folder, `rewrite-1p5b/eval_results/rebuttal_27/`, keeping kys-eval's own directory structure inside it, and also send them to Huicong directly. Huicong pairs them within seed against the v2 results with the repository's paired-analysis script. Do not average across seeds or tasks before sending.

## 11. Final report to Huicong

Provide the 3 × 3 × 3 completion table with final steps, the code and data SHAs actually used, the init hash check results, the filled cluster profile without secrets, measured s/it and total GPU-hours, every requeue and how it was recovered, the invariant and resume-check reports, the Hub revisions of all uploads, the reference-check output, and the 27 `results.json` files. State separately: training complete, export and upload complete, evaluation complete.

Known deviations already documented in WORKFLOW and RUNBOOK; do not resolve them, just preserve them: the published `quality_base` seed 42 ran at micro-batch 16 while all new chains use 32; the raw controls are equal-token-budget controls, not identical-document controls; λ = 0 seed 44 may run at dp 8; the new chains train on different hardware from v2; the published rewrites used greedy decoding without a sampling seed; seeds run concurrently in this set. Never shorten a horizon, drop a seed silently, or add a setting to meet the deadline.
