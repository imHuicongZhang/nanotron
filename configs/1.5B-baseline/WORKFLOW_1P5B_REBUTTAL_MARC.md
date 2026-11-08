# Know Your Sources - 1.5B rebuttal set - Marc execution workflow

For Marc and the agent operating Marc's training infrastructure. This document covers the ICLR 2027 rebuttal experiments: **three settings, three seeds, nine training chains, 27 annealed final models**, trained and evaluated under the exact protocol of the published 1.5B grid so that every result can be paired within seed against the published v2 checkpoints.

The repository documents `configs/1.5B-baseline/RUNBOOK.md` and `configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md` are the authoritative references for exact commands and flags. This document states what is fixed, what Marc fills in, the order of work, and the completion criteria. Where the two disagree, RUNBOOK and WORKFLOW win; report the disagreement.

Marc supplies hardware, allocation, filesystem paths, scheduler details and credentials. No GPU count is prescribed, but the schedule in §7 assumes 32 GPUs for wave 1.

## 1. Objective, ownership, and fixed release

Huicong has published the validated inputs, training code and evaluation code. Marc's agent downloads them, fills the cluster profile, tokenizes the two raw-text settings, trains the nine chains, converts and uploads the 27 finals, scores them with the pinned evaluation harness, and returns the per-checkpoint result files. Keep progress durable across session disconnects.

| Artifact | Exact source |
|---|---|
| Training code | https://github.com/imHuicongZhang/nanotron, branch `huicong-dev` |
| **Training commit** | **`PIN_SET_IN_NEXT_COMMIT`** (checked out explicitly in §4). Training code (everything outside `docs/` and the `configs/1.5B-baseline/*.md` documents) is identical since `6fe10b9b130a22338bfc1c2d199761131b711a88`; every later commit changes documentation and audit records only, confirmed by audit. Record the SHA used. |
| Evaluation code | https://github.com/imHuicongZhang/kys-eval, branch `main` |
| **Evaluation commit** | **`00bbca3483b35e932e20f75e6a49c18a57a225e4`** (checked out explicitly in §4). Scoring code is unchanged since `216c1ba`; `6da264bf` added the full-precision v2 reference, the `quality_base` pairing and the Appendix F sign convention, all of which the rebuttal needs; `00bbca34` changes only the README. |
| Data repository | https://huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten, public |
| **Dataset revision** | **`afd1f177`**. Use that pin, not `main`. |
| Initial weights | https://huggingface.co/wytro/Know-Your-Sources-init at **`87e1335647a3105471928e0fd261d92846021d1a`**, public with gating disabled (no token needed; verified 2026-11-08 by an anonymous download) |
| Comparator checkpoints | https://huggingface.co/wytro/KYS-1.5B-Rewritten-v2 at `b22c87333783945e1c655ae46886b7e842ee4203`, public. Used only by the evaluation reference check. |
| Model output repository | https://huggingface.co/blab-jhu/KYS-1.5B-Raw-Selected-Baselines, **gated**. Nodes that upload need a Hugging Face token with write access to `blab-jhu`; nodes that score need a token with read access to this gated repository. Keep tokens in the environment only, never in configs, logs or commits. |
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

Each corpus already contains its shared anchor and is already shuffled. Do not append another base, re-select, re-shuffle or re-tokenize the λ = 0 data. The λ = 0 total is 2 tokens below its 10,000,002,558 assembly budget because two empty rewrites tokenize to nothing; this is expected. The λ = 0 rewrites were generated in a separate run; byte identity of the historical rewriting prompts with the λ = 0.5 run could not be archived and is inferred from output agreement: documents rewritten in both runs are identical in 51.9% (wikipedia-style) and 45.9% (distill) of cases, against 55.7% and 49.9% for two runs known to share the prompt and 0% for a different prompt (audit 2026-11-08, §12).

Model and optimization, identical to the published grid (paper Table 11):

- Llama 2 architecture, 28 layers, hidden 2048, FFN 5632 SwiGLU, 16 heads, RoPE θ = 10⁴, RMSNorm, vocabulary 32000, tied embeddings, 1.504B parameters.
- Published Llama 2 tokenizer from the dataset's `tokenizer/` folder. Tokenization convention: text tokens plus one `</s>` per document, no `<s>`.
- AdamW β = (0.9, 0.95), weight decay 0.1, gradient clip 1.0, peak LR 5e-4, 500-step linear warmup once.
- Warmup-stable-decay. Constant-rate trunk with segment ends at **4292, 8583, 12875**. Three decay branches from fixed trunk checkpoints, each decaying linearly to zero over the last 10% of its horizon, ending at **4768, 9537, 14305**.
- Sequence length 2048, global batch 1024 sequences, **2,097,152 tokens per optimizer update**. bf16 parameters, fp32 gradients and optimizer state.
- **dp 4, micro-batch 32, accumulation 8**, four GPUs per chain, one node. The only permitted alternative is wave 2's λ = 0 seed 44 at dp 8, micro-batch 32, accumulation 4 on a single 8-GPU node (§7). Micro-batch must stay 32 in both; it is part of the loss weighting.
- Layer recomputation (`recompute_layer: true`) on 80 GB GPUs, which mbs 32 needs. The published v2 runs had it off; recomputation changes speed, not the math.
- Every chain starts from the published per-seed initialization in `wytro/Know-Your-Sources-init`, identical across settings for a given seed. Never start from an earlier partial run or a fresh random init.

## 3. What Marc fills in

**Operator values.** Define every site-specific value once. Save this block as `$HOME/kys_rebuttal_env.sh` and run `source "$HOME/kys_rebuttal_env.sh"` at the start of every shell session; every command below uses these variables. The paths must be literal absolute paths without whitespace, on shared storage visible from the compute nodes.

```bash
CODE_ROOT="<<absolute path for the nanotron checkout; also repo_dir in deploy/clusters.yaml>>"
EVAL_ROOT="<<absolute path for the kys-eval checkout>>"
DATA_ROOT="<<absolute path for the KYS-Pre-Rewritten download; also data_root>>"
INIT_ROOT="<<absolute path for the Know-Your-Sources-init download; also init_root>>"
CKPT_ROOT="<<absolute checkpoint root; also ckpt_root>>"
EXPORT_ROOT="<<absolute path for the HF exports of the 27 finals>>"
SMOKE_DIR="<<absolute scratch directory on shared storage for the 20-step smoke run>>"
S_PER_IT="<<seconds per step measured by the dp-4 smoke run, e.g. 22.8>>"
S_PER_IT_DP8="<<seconds per step measured by the dp-8 smoke run; only if wave 2 runs at dp 8>>"
export CODE_ROOT EVAL_ROOT DATA_ROOT INIT_ROOT CKPT_ROOT EXPORT_ROOT SMOKE_DIR S_PER_IT S_PER_IT_DP8
```

Fill the existing `marc-cluster` template entry in `deploy/clusters.yaml` with the same values: every field is `null` on purpose and `fill_placeholders.py` refuses while a required one is still null. If an 8-GPU node is available for wave 2, add `marc-cluster-8gpu` as a copy of `marc-cluster` with `gpus_per_node: 8`, `dp: 8`, `slurm.gres` giving 8 GPUs and `cpus_per_task` scaled; keep `micro_batch_size: 32`.

| Group | Operator inputs |
|---|---|
| Paths | `data_root` (= `$DATA_ROOT`), `tokenizer_path` (= `$DATA_ROOT/tokenizer`), `ckpt_root` (= `$CKPT_ROOT`), `init_root` (= `$INIT_ROOT`), `repo_dir` (= `$CODE_ROOT`; `plan_submit.py` refuses if it is not the clone it runs from), `wandb.dir`. All on shared storage visible from the compute nodes; `assert_invariants.py` rejects a node-local `wandb.dir` (`/tmp`, `/var/tmp`, `/dev/shm`, `/scratch/local`). SLURM logs go next to `wandb.dir` (`<parent>/slurm_logs`). |
| Environment | `env_activate`: the file sourced to activate the training environment (e.g. `/path/env/bin/activate`); optional `python_include` only if compute nodes lack Python headers |
| Deployment | `arch` (e.g. `sm_90`), `hbm_gib`, `gpus_per_node`, `dp` (4; 8 in `marc-cluster-8gpu`), `tp: 1`, `pp: 1`, `micro_batch_size: 32`, `recompute_layer: true` (needed for mbs 32 on 80 GB cards), `zero_stage: 0`; `wandb.mode`, `wandb.project` (`zhc-1p5b-10b-wsd`), `wandb.entity` (online only) |
| Scheduler | `slurm.partition`, `slurm.account` (optional), `slurm.qos` (optional), `slurm.gres` (dp GPUs on one node), `slurm.cpus_per_task`, `slurm.time` (must exceed ~9.5 h at 22.8 s/it, the longest stretch without a checkpoint; `plan_submit.py --s-per-it` refuses a shorter limit) |
| Credentials | Upload nodes (§9, §10): a Hugging Face token with write access to `blab-jhu`. Scoring nodes (§10): a token with read access to the gated `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`. Provide it through `HF_TOKEN` in the environment or `hf auth login`; never in `deploy/clusters.yaml`, a rendered config, a log or a commit. |

Do not fill in publisher paths from Skipjack. Do not change global batch, LR, horizons or micro-batch to fit an allocation.

## 4. Install the pinned environments

Two environments are needed: the nanotron training environment (`INSTALL.md` §2 in the nanotron checkout, plus wandb; it also holds datatrove 0.5.0 for tokenization and `huggingface_hub` 0.x, whose `hf` command does the model uploads in §9) and the kys-eval environment with its pinned, patched LightEval (`install.sh`; `huggingface_hub` 1.12.0, whose `hf` command does the result upload in §10; `huggingface-cli` no longer works there, so use `hf` in both environments).

```bash
source "$HOME/kys_rebuttal_env.sh"
NANOTRON_SHA="PIN_SET_IN_NEXT_COMMIT"
KYS_EVAL_SHA="00bbca3483b35e932e20f75e6a49c18a57a225e4"
git clone --branch huicong-dev https://github.com/imHuicongZhang/nanotron.git "$CODE_ROOT"
git -C "$CODE_ROOT" checkout "$NANOTRON_SHA"
git -C "$CODE_ROOT" log -1 --format=%H        # must print $NANOTRON_SHA; record it
git clone https://github.com/imHuicongZhang/kys-eval.git "$EVAL_ROOT"
git -C "$EVAL_ROOT" checkout "$KYS_EVAL_SHA"
git -C "$EVAL_ROOT" log -1 --format=%H        # must print $KYS_EVAL_SHA; record it

# training environment (INSTALL.md §2; Python 3.11; order matters)
conda create -n kys python=3.11 -y && conda activate kys
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install "datatrove[io]==0.5.0" "numpy==2.0.2" "huggingface_hub<1.0" \
            "transformers==4.46.3" "tokenizers==0.20.3" numba==0.60.0 pybind11
pip install -e "$CODE_ROOT"
mkdir -p "$PWD/.tmp"
TMPDIR="$PWD/.tmp" TORCH_CUDA_ARCH_LIST=9.0 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/fanshiqing/grouped_gemm@efe8c40eaf4c8ef57191e0ea9aa4117aa5b1a8f2"
pip install "wandb==0.30.0"                   # INSTALL.md does not install wandb; nanotron logs to it
python -c "import wandb; print('wandb', wandb.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"   # 0.x
hf --help > /dev/null && echo "hf CLI available"

# evaluation environment (in a new shell, after source "$HOME/kys_rebuttal_env.sh")
cd "$EVAL_ROOT"
./install.sh                                  # venv + pinned packages + patched LightEval; ends with check_install
source .venv/bin/activate
export HF_HOME="$EVAL_ROOT/hf_cache"
python -m kys_eval.check_install              # must end "OK: environment matches the pinned v2 setup"
python -m kys_eval.selftest                   # must print "selftest passed"
python -m kys_eval.prefetch                   # evaluation datasets at pinned revisions
hf --help > /dev/null && echo "hf CLI available"
```

The kys-eval self-test must pass before any scoring. On an architecture other than `sm_90`, follow INSTALL.md Appendix A.

## 5. Download, verify, tokenize, render

Download the dataset at the pinned revision into `$DATA_ROOT`, including `manifest.json`, `tokenizer/`, `raw_text/raw_diversity_oriented/`, `raw_text/raw_disagreement_aware/` and `signal-disagreement-lambda0/`. The other five `raw_text/` settings are not needed. Download the three per-seed init checkpoints from `wytro/Know-Your-Sources-init` at `87e1335` into `$INIT_ROOT` and verify them. Run everything below from `$CODE_ROOT` in the training environment.

```bash
cd "$CODE_ROOT"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset",
                  revision="afd1f177aa08f5bab9f5695048cc7b1be428d6fa", local_dir=os.environ["DATA_ROOT"],
                  allow_patterns=["manifest.json", "tokenizer/*", "raw_text/raw_diversity_oriented/*",
                                  "raw_text/raw_disagreement_aware/*", "signal-disagreement-lambda0/tokenized/*"])
snapshot_download("wytro/Know-Your-Sources-init", revision="87e1335647a3105471928e0fd261d92846021d1a",
                  local_dir=os.environ["INIT_ROOT"])
PY
for S in 42 43 44; do
  python tools/hash_init_checkpoint.py "$INIT_ROOT/_init_1.5B_seed$S/0" --check "$INIT_ROOT/init_1.5B_seed$S.hash.json"
  find "$INIT_ROOT/_init_1.5B_seed$S/0" -type f | wc -l     # must print 180
done
```

The raw-text parquet files are checked against the sha256 values in `manifest.json` by `tokenize_raw_text.sh` itself, before it tokenizes.

For λ = 0, **run the checksum check before `fix_ds_metadata.py`**, because that script rewrites metadata files:

```bash
cd "$CODE_ROOT"
(cd "$DATA_ROOT/signal-disagreement-lambda0/tokenized" && sha256sum -c SHA256SUMS)     # 78 files OK
python tools/fix_ds_metadata.py --output-folder "$DATA_ROOT/signal-disagreement-lambda0/tokenized" \
    --tokenizer-dir "$DATA_ROOT/tokenizer"
```

For the two raw settings, tokenize with the repository script (arguments: data root, then setting). It checks the tokenizer and the 16 parquet files against `manifest.json`, tokenizes with 16 tasks, one `</s>` per document, no BOS and no shuffling, runs `fix_ds_metadata.py` itself, and fails unless it gets 16 shards and the exact expected total:

```bash
cd "$CODE_ROOT"
bash tools/kys_raw/tokenize_raw_text.sh "$DATA_ROOT" raw_diversity_oriented
bash tools/kys_raw/tokenize_raw_text.sh "$DATA_ROOT" raw_disagreement_aware
# 16 tasks at once need ~165 GiB RAM; on a smaller node prefix KYS_TOKENIZE_WORKERS=n (n x 10.5 GiB must fit)
```

After tokenization the totals must be exactly 10,000,002,885 and 10,000,005,069 tokens in 16 shards each, and λ = 0 must be exactly 10,000,002,556 in 26 shards.

Render and check the configs for the three settings only. `--settings` is mandatory on both tools; omitting it is an error, and `--settings all` would plan all eight settings, including five that are not in this set. Both tools take one `--seed` per call; settings are comma-separated.

```bash
cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42 --settings "$W1"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 43 --settings "$W1"
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings raw_diversity_oriented,raw_disagreement_aware
# wave 2, λ = 0 seed 44, ONE of the two:
#   at dp 8, into a separate folder so seed 44's dp-4 configs stay as they are
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster-8gpu --seed 44 \
    --settings signal-disagreement-lambda0 --out-dir configs/1.5B-baseline-seed44/filled-dp8
#   or at dp 4: python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings signal-disagreement-lambda0
```

Then run the full invariant check of every rendered config, with its `.env` sourced. The block is a subshell whose exit status is nonzero if any config fails or if the count is not exactly 54 (18 per seed). **A failed invariant blocks launch:** do not plan or submit anything until this block exits 0, and report any failure with its `.invariants.log`.

```bash
cd "$CODE_ROOT"
(
  n_ok=0; n_fail=0
  for c in configs/1.5B-baseline-seed4{2,3,4}/filled/*_seed4?_*.yaml configs/1.5B-baseline-seed44/filled-dp8/*.yaml; do
    [ -f "$c" ] || continue
    if ( set -a; source "${c%.yaml}.env"; python tools/assert_invariants.py --config "$c" ) > "${c%.yaml}.invariants.log" 2>&1; then
      n_ok=$((n_ok + 1)); echo "OK   $c"
    else
      n_fail=$((n_fail + 1)); echo "FAIL $c  (see ${c%.yaml}.invariants.log)"
    fi
  done
  echo "invariants: $n_ok OK, $n_fail FAIL (expected 54 OK, 0 FAIL)"
  if [ "$n_fail" -ne 0 ] || [ "$n_ok" -ne 54 ]; then echo "INVARIANT CHECK FAILED: launch is blocked"; exit 1; fi
  echo "invariant check passed"
)
```

`fill_placeholders.py` already runs `assert_invariants.py --skip-corpus --skip-env` on each config it writes; the block above adds the corpus and environment checks. `assert_invariants.py --check-resume` is run by `deploy/slurm/kys_segment.sbatch` before every segment, against that segment's checkpoints; it is not a pre-submission step (before the trunk directories are seeded it has nothing to check).

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

**Trunk restarts and the trunk1 requeue rule.** Trunk warmup happens once, at the start of trunk1. All three trunk segments of a chain share one directory, `$CKPT_ROOT/seed<S>/<setting>/trunk/<setting>/seed<S>/`, and resume through its `latest.txt`; the submission script seeds it with the init as step 0 and `latest.txt` = 0, and trunk1 saves every 1500 steps (1500, 3000) before its final 4292. trunk1 is rendered weights-only (`load_optimizer: false`, `load_lr_scheduler: false`) because the init has no optimizer state. Every time a segment starts, `deploy/slurm/kys_segment.sbatch` does the following:

1. If the segment's final step directory already exists, it exits 0 without training.
2. It runs `tools/kys_raw/trunk_restart_config.py` on the rendered config:
   - if `latest.txt` = 0, or the config already loads optimizer and LR state, it returns the rendered config unchanged;
   - if `latest.txt` > 0 and the config is weights-only, it writes `<trunk dir>.restart/<run>_from<step>.yaml`, identical except `load_optimizer: true` and `load_lr_scheduler: true`, and returns that copy;
   - it exits nonzero if `latest.txt` is missing, the step has no checkpoint, or the step is not below `train_steps`.
3. It prints `[kys] running config: <path>`.
4. It runs `assert_invariants.py --config <that path> --check-resume`, which fails the job before `torchrun` if `latest.txt` > 0 while either load flag is false.

Two kinds of trunk1 start are **legitimate**:

- the first start, or any requeue before the step-1500 save, so that `latest.txt` = 0;
- a restart with `latest.txt` > 0.

For the first kind, `running config:` is the rendered trunk1 config, the first log line is `iteration: 1 / 4292`, and training starts again from the init, which is correct. For the second kind, `running config:` is the `.restart/..._from1500.yaml` or `_from3000.yaml` copy, and the first log line is `iteration: 1501 / 4292` or `iteration: 3001 / 4292`. trunk2 and trunk3 always load full state and first log `latest.txt` + 1.

The **blocker** is a trunk start that logs `iteration: 1` again while `latest.txt` > 0, for example because the rendered config was launched by hand outside `kys_segment.sbatch`. On every trunk1 requeue, check that the step in the `.restart` file name, the value of `latest.txt` and the first logged iteration agree. If they do not, stop that chain and report it, because the result would be invalid. A job that fails in `--check-resume` is the guard working, not a silent error: report it and do not bypass it.

## 7. Submit in two waves, monitor durably, recover precisely

The execution plan is WORKFLOW §1b. With 32 GPUs:

**Wave 1, eight concurrent chains, dp 4 / mbs 32 / accum 8, four GPUs each:** `raw_diversity_oriented` and `raw_disagreement_aware` at seeds 42, 43, 44, plus `signal-disagreement-lambda0` at seeds 42 and 43. Seeds run concurrently in this set; this overrides the one-seed-at-a-time rule used for the full grid.

**Wave 2, after wave 1 frees GPUs:** `signal-disagreement-lambda0` seed 44. On one 8-GPU node use dp 8 / mbs 32 / accum 4 through the `marc-cluster-8gpu` profile; otherwise dp 4 on four GPUs. Tokens per step are identical either way.

Before the first full submission, run the 20-step smoke run on one chain. `tools/kys_raw/smoke_20steps.py` derives a config from the filled trunk1 config that runs 20 optimizer steps from a scratch copy of the init and **writes no checkpoints**, then prints the torchrun command:

```bash
cd "$CODE_ROOT"
python tools/kys_raw/smoke_20steps.py \
    --config configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_trunk1.yaml \
    --init "$INIT_ROOT/_init_1.5B_seed42/0" --workdir "$SMOKE_DIR"
# then run the torchrun command it prints, on one node with dp GPUs
# (for dp 8: --config configs/1.5B-baseline-seed44/filled-dp8/signal-disagreement-lambda0_seed44_trunk1.yaml)
```

Smoke success criteria: finite loss; `lm_loss` at steps 1, 10 and 20 close to the reference 10.8 / 8.83 / 8.19 (4 × H100, dp 4, mbs 32 + recompute, rewritten `diversity_oriented`; a raw corpus differs slightly); the expected LR at step 1 (linear warmup from 5e-4 over 500 steps); 2,097,152 consumed tokens per step; steady-state `time_per_iteration_ms` (reference 22.8 s/it) and peak GPU memory (reference 59.4 GiB). Set `S_PER_IT` (and `S_PER_IT_DP8`) to the measured s/it. Checkpoint persistence is not tested by the smoke run: it is verified by the first real segment (trunk1's step-1500 save, then `latest.txt` = 1500) and, after any requeue, by the restart rule in §6. Report the measured s/it; Huicong's cluster measured about 100 hours per chain on four H100s with this configuration, roughly 400 H100-hours per chain.

Submit with `plan_submit.py`, always with explicit `--settings` and one `--seed` per call, in **three separate steps per wave**:

1. **Plan** without `--submit`. It writes and prints `configs/1.5B-baseline-seed<S>/filled/submit_seed<S>.sh` and the job table and submits nothing. Inspect each script: six segments per setting; the job names `kys_<setting>_seed<S>_<segment>`; the `afterok` chain; the per-job time limit; only the intended settings.
2. **Validate** with `--test-only`. SLURM checks partition, account, QoS, gres and time for every segment without queueing anything.
3. **Submit** with `--submit`, only after steps 1 and 2 have both succeeded for every line of the wave.

```bash
# ---- wave 1 ----
cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
# step 1: plan and inspect
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1"
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1"
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware
# step 2: validate with SLURM, nothing queued
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1" --test-only
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1" --test-only
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware --test-only
```

```bash
# step 3, wave 1: only after steps 1 and 2 succeeded for all three lines
cd "$CODE_ROOT"
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it "$S_PER_IT" --settings "$W1" --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it "$S_PER_IT" --settings "$W1" --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it "$S_PER_IT" --settings raw_diversity_oriented,raw_disagreement_aware --submit
```

```bash
# ---- wave 2, after wave 1 frees GPUs, at dp 8 with the filled-dp8 configs ----
cd "$CODE_ROOT"
# step 1: plan and inspect
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8"
# step 2: validate with SLURM, nothing queued
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8" --test-only
```

```bash
# step 3, wave 2: only after steps 1 and 2 succeeded
cd "$CODE_ROOT"
python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
    --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it "$S_PER_IT_DP8" --submit
```

If wave 2 runs at dp 4 instead, use the same three steps with `--cluster marc-cluster --s-per-it "$S_PER_IT"` and without `--configs` (the dp-4 λ = 0 seed 44 configs are in `filled/`). That plan rewrites `filled/submit_seed44.sh`, which wave 1 already ran; the wave-1 configs are kept.

The generated script refuses to submit anything if one of its job names is already queued for `$USER`, and seeds each trunk directory with the init checkpoint (step 0 and `latest.txt`).

Never pass `--settings all`. If wave 2 runs at dp 8, report that as a deviation.

Each segment's sbatch runs `assert_invariants.py` before training and refuses to start on a failed check. Segments chain with `afterok` dependencies. Save every job ID as it is issued. Maintain a single submitter; after a disconnect, reconcile queue state against checkpoints and submit only genuinely missing segments. Do not rerun a submission script wholesale.

On failure, read the log, resume from validated state, and never alter LR, horizon or micro-batch to recover.

## 8. ETA and completion criteria

Nine chains at about 100 hours each, 897 node-hours on 4-GPU nodes. With wave 1 fully concurrent, wave 1 finishes about 4.2 days after launch and wave 2 about 2 to 4 days later depending on dp. **The rebuttal discussion closes November 18. Results must be scored and returned by November 15 to be usable.**

**Partial deliverable.** If wave 2 cannot finish by November 15, wave 1 alone is a partial deliverable that Huicong has pre-approved for the rebuttal. λ = 0 results from seeds 42 and 43 only must then be labelled **n = 2** everywhere and never presented as a three-seed estimate. **Notify Huicong immediately** as soon as November 15 is at risk; do not wait for the deadline.

**Storage** (WORKFLOW_RAW_BASELINES.md §1b and §12, from measured sizes):
- **Peak local checkpoint storage** is at most **2.7 TB**: 9 chains × 303.9 GB with nothing pruned, where each chain has 14 full checkpoints × 21.06 GB plus the 9.03 GB init copy.
- **Retained** after pruning: each chain keeps the branch points 4292, 8583 and 12875 and its three finals, 6 × 21.06 GB = 126.4 GB per chain.
- **Pruning:** the trunk restart checkpoints (1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000) may be deleted once the next trunk segment has completed. Never delete a branch point before its branch has finished.
- **Inputs** add roughly 0.15 TB once:
  - two raw-text settings at 16.6–16.7 GB each;
  - their tokenized output, about 19–20 GB each (WORKFLOW §12: 159 GB tokenized over eight settings);
  - λ = 0 tokenized, 19 GB;
  - the three init checkpoints, 27.1 GB.
- **Uploaded bytes**, a separate quantity, total **0.65 TB**: 27 finals × (21.06 GB nanotron step directory + 3.01 GB HF export). The loss CSVs and evaluation results are negligible.

Training completion means all 27 final checkpoint directories exist at `$CKPT_ROOT/seed<S>/<setting>/ep<N>/<setting>/seed<S>/<step>/` with `<step>` = 4768, 9537 or 14305 (trunk segments write to `$CKPT_ROOT/seed<S>/<setting>/trunk/<setting>/seed<S>/<step>/`), each with model tensors, config, metadata, optimizer and LR state, the correct final step and a clean final log. A folder name or an empty queue is not completion.

## 9. Export and publish the 27 models

Convert each final with the repository's CPU converter and confirm the export loads with 28 layers, 1.504B parameters, tied embeddings and the published tokenizer. Uploads in this section run in the **training environment**, whose `huggingface_hub` 0.x provides `hf`, on a node with a token that has write access to `blab-jhu`. The upload layout below is exactly what kys-eval reads:

```
rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/
rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/      (native final step directory as written, ~21 GB incl. optimizer state; required)
rewrite-1p5b/seed<S>/<setting>/logs/<setting>_seed<S>_<segment>.csv    (step, lm_loss, lr, tokens/s)
```

```bash
cd "$CODE_ROOT"
for S in 42 43 44; do
  for setting in raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0; do
    for N in 1 2 3; do
      step=$(case $N in 1) echo 4768;; 2) echo 9537;; 3) echo 14305;; esac)
      final="$CKPT_ROOT/seed$S/$setting/ep$N/$setting/seed$S/$step"
      hfdir="$EXPORT_ROOT/rewrite-1p5b/seed$S/$setting/ep$N/hf"
      [ -f "$final/model_config.json" ] || { echo "missing final $final"; continue; }
      python tools/kys_eval/convert_to_hf.py --checkpoint_path "$final" --save_path "$hfdir" \
          --tokenizer "$DATA_ROOT/tokenizer" --dtype bfloat16
      hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "$hfdir" "rewrite-1p5b/seed$S/$setting/ep$N/hf"
      hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "$final" "rewrite-1p5b/seed$S/$setting/ep$N/nanotron"
    done
  done
done
# Hub revision after the uploads: record it, it is the HF_REVISION of §10
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('blab-jhu/KYS-1.5B-Raw-Selected-Baselines').sha)"
```

Record the Hub revision after each upload batch and the final one after all 27. **Never re-upload a checkpoint to the same path without telling Huicong**: the evaluation cache cannot detect a same-path re-upload, so the affected cells must then be re-scored with `--force` at the new revision (§10). Pin the final revision in the report.

## 10. Evaluation

Use kys-eval at the pinned revision, in the **evaluation environment**, on nodes whose token can read the gated output repository, and score on **H100 80GB HBM3 (SXM)**, the GPU the v2 reference numbers were produced on; the harness accepts any GPU whose name contains "H100", so this is on the operator. First run the reference check, which re-scores a published v2 checkpoint and must print PASS before any new model is scored:

```bash
cd "$EVAL_ROOT" && source .venv/bin/activate && export HF_HOME="$EVAL_ROOT/hf_cache"
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203
```

Then score the 27 rebuttal cells only, all at one pinned Hub revision of the output repository: the revision recorded after the last of the 27 uploads (§9), passed with `--hf-revision` to every cell through `KYS_GRID_ARGS`. In kys-eval the settings are **space-separated** (unlike the comma-separated nanotron tools). Replace the `<PARTITION>`, `<ACCOUNT>`, `<GPU_TYPE>` and `<TIME>` placeholders in `slurm/eval_grid_array.sbatch` first (`01:00:00` per checkpoint is safe). The array refuses to run without `KYS_GRID_ARGS` and refuses an `--array` range that is not `0..N-1` for the selected cells. `KYS_GRID_ARGS=all` is the 72-cell full grid and is not wanted here.

```bash
cd "$EVAL_ROOT" && source .venv/bin/activate && export HF_HOME="$EVAL_ROOT/hf_cache"
HF_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('blab-jhu/KYS-1.5B-Raw-Selected-Baselines').sha)")
echo "scoring at $HF_REVISION"                  # must equal the revision recorded in §9; record it
R="--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0 --hf-revision $HF_REVISION"
python -m kys_eval.run_grid $R --list --check-hub      # "27 cells; SLURM: --array=0-26", all "present"
mkdir -p logs && sbatch --array=0-26%8 --export=ALL,KYS_EVAL_ROOT="$EVAL_ROOT",KYS_GRID_ARGS="$R" slurm/eval_grid_array.sbatch
python -m kys_eval.run_grid $R --list                  # all 27 "done"
python -m kys_eval.aggregate                           # reports/raw_selected_report.md, raw_vs_rewritten.md, CSVs
```

Indices 0–8 are seed 42, 9–17 seed 43, 18–26 seed 44. To re-run single indices, also export `KYS_ARRAY_PARTIAL=1`. **After any re-upload**, record the new revision, set `HF_REVISION` to it and re-score the affected cells with `--force` added to `KYS_GRID_ARGS`. Protocol, fixed by the harness: six commonsense tasks (ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA), MMLU over 57 subjects, zero-shot, full splits, `acc_norm`.

Run `aggregate.py` for a first look, but the deliverable is the **27 per-checkpoint `results.json` files**, unmodified, plus the reference-check output. Once all 27 cells are scored, upload kys-eval's `results/` and `reports/` per seed to `rewrite-1p5b/seed<S>/eval/` (WORKFLOW §14), keeping kys-eval's own directory structure, and also send them to Huicong directly. This upload runs in the **evaluation environment** (`huggingface_hub` 1.12.0: use `hf`; `huggingface-cli` no longer works there) on a node with a write token:

```bash
cd "$EVAL_ROOT" && source .venv/bin/activate
for S in 42 43 44; do
  hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines "results/raw_selected/seed$S" \
      "rewrite-1p5b/seed$S/eval/results/raw_selected/seed$S"
  hf upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines reports "rewrite-1p5b/seed$S/eval/reports"
done
```

The reference-check output goes with the results to Huicong. Huicong pairs them within seed against the v2 results with the repository's paired-analysis script. Do not average across seeds or tasks before sending.

## 11. Final report to Huicong

Provide the 3 × 3 × 3 completion table with final steps, the code and data SHAs actually used, the init hash check results, the filled cluster profile without secrets, measured s/it and total GPU-hours, every requeue and how it was recovered, the invariant and resume-check reports, the Hub revisions of all uploads and the `HF_REVISION` used for scoring, the reference-check output, and the 27 `results.json` files. State separately: training complete, export and upload complete, evaluation complete.

Known deviations already documented in WORKFLOW and RUNBOOK; do not resolve them, just preserve them: the published `quality_base` seed 42 ran at micro-batch 16 while all new chains use 32; the raw controls are equal-token-budget controls, not identical-document controls; λ = 0 seed 44 may run at dp 8; the new chains train on different hardware from v2; the published rewrites used greedy decoding without a sampling seed; the λ = 0 rewrites come from a separate run whose prompt identity with λ = 0.5 is inferred from output agreement (51.9% / 45.9% identical outputs vs 55.7% / 49.9% for a known same-prompt pair and 0% for a different prompt), because the historical prompt files were not archived; seeds run concurrently in this set. Never shorten a horizon, drop a seed silently, or add a setting to meet the deadline.
