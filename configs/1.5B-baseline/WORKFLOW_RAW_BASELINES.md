# Know Your Sources — Raw-Selected Baselines — Execution Workflow

**For:** Marc Marone's cluster, and the agent operating it.
**Repo:** `github.com/imHuicongZhang/nanotron`, branch `huicong-dev`, at or after commit `ad78783`.
**Data:** `huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten` (raw text and tokenizer).
**Evaluation:** `github.com/imHuicongZhang/kys-eval`, branch `main`; v2 comparator checkpoints at `huggingface.co/wytro/KYS-1.5B-Rewritten-v2`.

This document is the single entry point. `configs/1.5B-baseline/RUNBOOK.md`, `INSTALL.md`
and `SOP.md` in the repo are the detailed references; this file says what to do, in what
order, and who does it. It follows the 1.5B `WORKFLOW.md` step for step; what changed is
that the data arrives as raw text and you tokenize it, the grid is 4 settings instead of 6,
seeds run one after another, and the submit script now exists in the repo.

---

## 0. Division of labour

| | Human (Marc) | Agent (Claude Code / Codex) |
|---|---|---|
| **Step 1** Clone repo | — | yes |
| **Step 2** Build env | — | yes |
| **Step 3** Download raw text, tokenizer, init checkpoints | — | yes |
| **Step 4** Tokenize the four corpora (CPU) | — | yes |
| **Step 5** Fill `marc-cluster` in `clusters.yaml` | **yes — only human step** | — |
| **Step 6** Smoke checks incl. the 20-step run | — | yes |
| **Step 7** Fill configs for seed 42 | — | yes |
| **Step 8** Plan and submit seed 42 | review the plan | yes |
| **Step 9** Convert, upload, report | — | yes |
| **Step 10** Repeat 7–9 for seeds 43, 44 | — | yes |

Everything except step 5 is mechanical and verifiable. Step 5 needs facts only you have.

Before any of this can start, Huicong provides: the four `raw_text/` folders plus
`manifest.json` on the HF dataset (the tokenizer ships in the same dataset under `tokenizer/`),
read access to `wytro/Know-Your-Sources-init`, and write access to the model repo
`blab-jhu/KYS-1.5B-Raw-Selected-Baselines` for uploads. Ask if any of these is missing.

---

## 0a. Everything Marc has to supply by hand — the complete list

Nothing else in either repository needs a human value. If the agent asks for anything not on
this list, stop and ask Huicong.

### A. `deploy/clusters.yaml`, entry `marc-cluster` (nanotron repo)

Paths on your cluster. Absolute paths; the agent creates the directories.

| key | value | notes |
|---|---|---|
| `data_root` | e.g. `/scratch/<you>/kys_raw` | where `snapshot_download("blab-jhu/KYS-Pre-Rewritten")` lands; ~80 GB raw text, +~80 GB after tokenizing |
| `tokenizer_path` | `<data_root>/tokenizer` | the directory, not the `.json`; it ships inside the same download |
| `ckpt_root` | e.g. `/scratch/<you>/kys_ckpt` | reserve 3.5 TB per seed (1.2 TB after pruning) |
| `init_root` | e.g. `/scratch/<you>/kys_init` | where `snapshot_download("wytro/Know-Your-Sources-init")` lands; ~27 GB |
| `repo_dir` | e.g. `/home/<you>/nanotron` | this checkout, branch `huicong-dev` |
| `env_activate` | e.g. `/home/<you>/envs/kys/bin/activate` | the training env from `INSTALL.md` |
| `python_include` | leave `null` unless compute nodes lack `Python.h` | then the Python 3.11 include dir |

Hardware. Read from your nodes; do not guess.

| key | value | notes |
|---|---|---|
| `arch` | `sm_90` for H100/H200, `sm_100` for B200/B300 | `nvidia-smi --query-gpu=compute_cap --format=csv` |
| `hbm_gib` | `74.5` for 80 GB H100, `131.3` for H200, `268.0` for B300 | usable memory, drives the fit check |
| `gpus_per_node` | GPUs on one node | |
| `dp` | equal to `gpus_per_node` | one node per chain |
| `tp` | `1` | fixed |
| `pp` | `1` | fixed |
| `micro_batch_size` | `32` | fixed unless it cannot fit; see §13 |
| `recompute_layer` | `true` on 80 GB cards, otherwise per the renderer's fit output | speed only |
| `zero_stage` | `0` | fixed |

Logging. Offline is fine; CSV export is the deliverable.

| key | value |
|---|---|
| `wandb.mode` | `offline` (or `online` with your own entity) |
| `wandb.project` | `zhc-1p5b-10b-wsd` |
| `wandb.entity` | your entity if online, else leave `null` |
| `wandb.dir` | shared storage, not `/tmp` |

SLURM. From `sinfo -o "%P %a %l %D %G"` and your allocation.

| key | value |
|---|---|
| `slurm.partition` | a partition whose nodes have `gpus_per_node` GPUs |
| `slurm.account` | your allocation, or delete if none |
| `slurm.qos` | if your site needs one, else delete |
| `slurm.nodes` | `1` |
| `slurm.gres` | e.g. `gpu:4` or `gpu:h100:4`; must equal `dp × tp × pp` |
| `slurm.cpus_per_task` | what the node offers, e.g. `32` |
| `slurm.time` | ≥ 3 h so `ep3` (1430 steps) fits in one job; trunks requeue regardless |

### B. `slurm/eval_grid_array.sbatch` and `slurm/reference_check.sbatch` (kys-eval repo)

| placeholder | value |
|---|---|
| `<PARTITION>` | a partition with H100s |
| `<ACCOUNT>` | your allocation, or delete the line |
| `<GPU_TYPE>` | gres name for H100, e.g. `h100`; or `--gres=gpu:1` plus a `--constraint` |
| `<TIME>` | `01:00:00` per checkpoint |
| `KYS_REFERENCE_MODEL` | `hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf` |

### C. Accounts and access

| item | needed for |
|---|---|
| HF read token (or public access) | downloading the two datasets and the init checkpoints |
| HF write token with access to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` | uploading checkpoints, logs and eval results |
| GitHub push access to branch `marc-runs` on `imHuicongZhang/nanotron` | committing the filled configs; or send them as a patch |

Huicong sets up the HF model repo and the branch; tell him which HF username and GitHub
username to grant.

### D. Things to report back before the first training job

GPU model and count per node, how many nodes you can hold at once, the partition wall limit,
and the smoke-test numbers from §6.

---

## 1. Clone the code

```bash
git clone -b huicong-dev https://github.com/imHuicongZhang/nanotron.git
cd nanotron
git log --oneline -1        # expect ad78783 or later
```

Read `INSTALL.md` before installing anything. The fork carries 9 source patches already
applied on this branch; `pip install` of upstream nanotron will not work.

---

## 2. Environment

Follow `INSTALL.md` §2 exactly. The pins are interlocking, not preferences:

```bash
conda create -n kys python=3.11 -y && conda activate kys
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install "datatrove[io]==0.5.0" "numpy==2.0.2" "huggingface_hub<1.0" \
            "transformers==4.46.3" "tokenizers==0.20.3" numba==0.60.0 pybind11
pip install -e .
# grouped_gemm at the commit pinned in INSTALL.md — a runtime import, every run dies without it
```

H100 and H200 are `sm_90`; the prebuilt torch and flash-attn wheels carry `sm_90` SASS, so no
source build is needed. B200/B300 are `sm_100`; see `INSTALL.md` Appendix A before using them.

If compute nodes lack Python development headers, set `python_include` in the cluster entry
(step 5). Triton and nanotron's dataset helper compile against `Python.h` at run time.

---

## 3. Download — raw text, tokenizer, init checkpoints

```python
from huggingface_hub import snapshot_download

data_root = snapshot_download(                      # ~80 GB raw text, 4 settings
    repo_id="blab-jhu/KYS-Pre-Rewritten",
    repo_type="dataset",
    local_dir="/YOUR/PATH/kys_raw",
)
init_root = snapshot_download(                      # ~27 GB, model repo — no repo_type
    repo_id="wytro/Know-Your-Sources-init",
    local_dir="/YOUR/PATH/init",
)
```

`local_dir` **is** the root. Nothing is moved or renamed afterwards.

```
<data_root>/manifest.json                          token counts, sha256, tokenizer revision, code commit
<data_root>/raw_text/<setting>/part-000NN.parquet  16 files per setting, columns orig_doc_id, source, text
<data_root>/tokenizer/                             Llama-2 tokenizer, vocab 32000, BOS 1, EOS 2 (same dataset repo)
<init_root>/_init_1.5B_seed<S>/0/                  the checkpoint itself, S in 42 43 44
<init_root>/init_1.5B_seed<S>.hash.json            its manifest
```

Do not download the rewritten arms (`wytro/Know-Your-Sources`, `wytro/Know-Your-Sources-tokenized`).
They are not needed to train.

Verify before going on:

```bash
python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed42/0 \
    --check <init_root>/init_1.5B_seed42.hash.json          # repeat for 43, 44
sha256sum <data_root>/tokenizer/tokenizer.json              # must equal tokenizer.sha256 in manifest.json
```

---

## 4. Tokenize the four corpora — CPU only, once

The parquet files are the **final corpus in training order**: the 5B anchor is already merged
into each setting, documents are already shuffled with seed 42, rows are in order. Your job is
to turn text into `.ds` shards without changing that order.

```bash
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_diversity_oriented <data_root>/tokenizer
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_disagreement_aware <data_root>/tokenizer
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_random             <data_root>/tokenizer
tools/kys_raw/tokenize_raw_text.sh <data_root> raw_rewire_inspired    <data_root>/tokenizer
```

For each setting the script:

1. checks the 16 parquet files against the sha256 in `manifest.json`;
2. runs `tools/preprocess_data_parquet.py` with `--n-tasks 16` and no shuffle, so datatrove
   task i reads exactly file i and shards `00000…00015` concatenate to the file order;
3. runs `tools/fix_ds_metadata.py --tokenizer-dir <data_root>/tokenizer` (see below);
4. asserts 16 shards and that the summed `.ds.metadata` token count equals
   `settings.<setting>.expected_total_tokens` in `manifest.json` **exactly**.

Output lands at `<data_root>/<setting>/tokenized/000NN_unshuffled.ds{,.index,.metadata}`.
That is what the templates reference as `{{DATA_ROOT}}/<setting>/tokenized`.

Rules that protect data order. Do not pass `--shuffle`, do not change `--n-tasks`, do not
re-split or concatenate the parquet files, do not use any other tokenizer file or revision.
Any of these changes the data order or the token stream and the run is no longer comparable.

Budget: a few CPU-hours per setting at 16 cores. Run the four in parallel if you can.

### Why `fix_ds_metadata.py` runs

Every `*.ds.metadata` file has one line, `<tokenizer path>|<bytes per token>`. datatrove
must be given the `tokenizer.json` **file**; nanotron's config asserts the training
`tokenizer_path` equals that string exactly and then calls `AutoTokenizer.from_pretrained`
on it, which needs a **directory**. The tool rewrites the left side to the directory you pass.
Pass the **same** `--tokenizer-dir` for all four settings; nanotron also asserts every metadata
file across every folder carries an identical tokenizer string. The script already does this;
`assert_invariants.py` catches it if it was skipped and prints the exact command.

---

## 5. THE ONLY HUMAN STEP — fill `marc-cluster` in `deploy/clusters.yaml`

Every field of the `marc-cluster` entry is `null` on purpose and commented in place. Fill all
of them; the tooling refuses to run past a null.

| group | keys | what to put |
|---|---|---|
| paths | `data_root` | the `local_dir` from step 3 |
| | `tokenizer_path` | `<data_root>/tokenizer` — the directory, same string used in step 4 |
| | `ckpt_root` | your choice — **reserve 3.5 TB per seed**, see §11 |
| | `init_root` | the init `local_dir` from step 3 |
| | `repo_dir` | absolute path of this checkout |
| | `env_activate` | file to `source` for the env from step 2 |
| | `python_include` | optional, only if compute nodes lack `Python.h` |
| hardware | `arch` | `sm_90` for H100/H200, `sm_100` for B200/B300 |
| | `hbm_gib` | usable GPU memory, e.g. `74.5` for 80 GB H100, `131.3` for H200 |
| | `gpus_per_node` | GPUs on one node |
| | `dp`, `tp`, `pp` | `dp` = GPUs on one node, `tp: 1`, `pp: 1` |
| | `micro_batch_size` | **32** — see §13 |
| | `recompute_layer` | `true` on 80 GB cards; on 140 GB+ check the renderer's fit output |
| | `zero_stage` | `0` |
| logging | `wandb.mode/project/entity/dir` | `offline` is fine; `project: zhc-1p5b-10b-wsd`; `dir` on shared storage, not `/tmp` |
| SLURM | `partition`, `account`, `qos`, `gres`, `cpus_per_task`, `time` | see below |

**`partition` / `gres`.** Find them with `sinfo -o "%P %a %l %D %G"`. `gres` must provide
`dp × tp × pp` GPUs on one node, e.g. `gpu:4` or `gpu:h100:4`.

**`time`.** Per-job wall clock limit. The longest segment is a trunk at 4292 steps; at about
4 s/step on 4 × H100 that is about 5 hours. Trunks checkpoint every 1500 steps and requeue, so
a short limit costs at most 1500 steps of rework. Branches write **only** their final state:
a killed `ep3` reruns all 1430 steps (about 1.6 h). Size `time` so the branches are covered
comfortably; trunks tolerate anything.

`dp` may be any power of two such that `mbs × dp` divides 1024; `accum` is derived. If your
nodes are not 4 GPUs, set `dp` to what one node has and tell Huicong which GPU type it is.

---

## 6. Smoke checks

Per `INSTALL.md` §3, four of them. All must pass before submitting.

| check | what it proves |
|---|---|
| §3.1 GPU + flash-attn | torch sees the GPUs, flash-attn imports for your `arch` |
| §3.2 tokenizer | `AutoTokenizer.from_pretrained(<data_root>/tokenizer)` works and `len == 32000` |
| §3.3 init checkpoint | `hash_init_checkpoint.py --check` matches the manifest, all three seeds |
| §3.4 20-step run | real data, real init, filled config, global batch, on one node |

§3.4 needs step 7 done first for seed 42. Then:

```bash
python tools/kys_raw/smoke_20steps.py \
    --config configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_trunk1.yaml \
    --init <init_root>/_init_1.5B_seed42/0 --workdir <scratch dir>
# then run the torchrun command it prints, on one node with dp × tp × pp GPUs
```

It must reach step 20 and exit 0. Report `lm_loss` at steps 1, 10, 20 and the steady-state
`time_per_iteration_ms` to Huicong. Reference values from our 4 × H100 run are in
`INSTALL.md` §3.4 once the follow-up commit lands; agreement to about 0.01 confirms the stack
reproduces ours.

---

## 7. Fill the configs — per seed

```bash
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42
```

This re-renders the 24 templates in `configs/1.5B-baseline-seed42/templates/` into
`configs/1.5B-baseline-seed42/filled/` with your `marc-cluster` values, derives
`accum = 1024 / (mbs × dp)`, checks the memory fit against `hbm_gib`, and runs the
placeholder check on all 24. Each render emits `<name>.yaml` and `<name>.env`; both are
needed. **Never edit the 24 rendered configs by hand.**

If `mbs` is not 32, pass `--expected-mbs <N>` here and in step 8, and report it (§13).

Commit the filled `clusters.yaml` entry and the `filled/` directories to a branch named
`marc-runs`. Do not push to `huicong-dev`.

---

## 8. Plan and submit — per seed

```bash
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured>
# writes configs/1.5B-baseline-seed42/filled/submit_seed42.sh and prints the plan
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured> --submit
```

Without `--submit` nothing is submitted; the script is written and printed. Review it, then
submit. The script:

1. seeds each of the four trunk directories with the seed's init checkpoint as step `0` plus
   `latest.txt` (skipped when `latest.txt` already exists; never overwrites a trunk that has
   progressed);
2. submits the 24 segments of `deploy/slurm/kys_segment.sbatch` with `--dependency=afterok`
   encoding §9 exactly and `--requeue`.

Inside every job, on the compute node, the sbatch script sources the `.env`, runs
`assert_invariants.py --config <cfg> --check-resume`, and only then `torchrun`. A segment
whose final checkpoint already exists exits 0 before `torchrun`, so re-running
`submit_seed<S>.sh` at any time is safe and never double-trains.

`--settings a,b` restricts to a subset, so one chain can be validated first.

---

## 9. Training order and dependency graph

**Seeds run one after another.** Seed 43 is not started until all 12 ep finals of seed 42
exist on disk and are uploaded; seed 44 likewise after seed 43. Within a seed the four settings
are independent chains and should run in parallel.

Within a chain:

```
trunk1 (1→4292) ──→ trunk2 (4293→8583) ──→ trunk3 (8584→12875)
   │                    │                      │
   └→ ep1 (→4768)       └→ ep2 (→9537)         └→ ep3 (→14305)
```

| run | steps | resumes from | writes to | LR at start → end |
|---|---|---|---|---|
| `trunk1` | 1 → 4292 | `<trunk>` via `latest.txt` (step 0) | `<trunk>` | warmup 0 → 5e-4 over 500, then flat |
| `trunk2` | 4293 → 8583 | `<trunk>` via `latest.txt` (4292) | `<trunk>` | flat 5e-4 |
| `trunk3` | 8584 → 12875 | `<trunk>` via `latest.txt` (8583) | `<trunk>` | flat 5e-4 |
| `ep1` | 4293 → 4768 | `<trunk>/4292` **direct** | `<ep1>` | 5e-4 → 0 over 476 |
| `ep2` | 8584 → 9537 | `<trunk>/8583` **direct** | `<ep2>` | 5e-4 → 0 over 954 |
| `ep3` | 12876 → 14305 | `<trunk>/12875` **direct** | `<ep3>` | 5e-4 → 0 over 1430 |

Branches resume from a **step directory**, not the trunk folder, so they ignore `latest.txt`
and always fork from their own branch point.

**Why order matters.** If `<trunk>/4292` does not exist when `ep1` starts, nanotron logs "No
previous checkpoint found" at INFO level, trains 476 steps **from random initialisation**, then
exits 0 with a plausible-looking checkpoint and a plausible annealing curve. Nothing says it
failed. `afterok` makes this unschedulable rather than silent; the `--check-resume` preflight
inside the job is the second guard.

---

## 10. Setting ↔ corpus ↔ model

| setting | rewritten counterpart | corpus dir | what the strategy 5B is |
|---|---|---|---|
| `raw_diversity_oriented` | `diversity_oriented` | `raw_diversity_oriented/tokenized` | source docs selected by Diversity Oriented, unrewritten |
| `raw_disagreement_aware` | `disagreement_aware` | `raw_disagreement_aware/tokenized` | source docs selected by Disagreement Aware, unrewritten |
| `raw_random` | `wrap_inspired` | `raw_random/tokenized` | the random 10B sample WRAP rewrote, unrewritten; also the no-selection reference |
| `raw_rewire_inspired` | `rewire_inspired` | `raw_rewire_inspired/tokenized` | source docs of the 5B kept by REWIRE's post-rewrite filter, unrewritten |

Every corpus is the shared 5B anchor (4,120,164 docs, 5,000,002,332 tokens, identical across
the four) plus the setting's own 5B of raw source text, subsampled at the document level with
seed 42 from about 10B raw source tokens, merged, shuffled with seed 42. Exact doc and token
counts per setting are in `manifest.json`; `assert_invariants.py` reads them from there.

**Model:** identical across all four and identical to the rewritten arms. 1.5B Llama,
1,504,299,008 parameters — 28 layers, hidden 2048, FFN 5632, 16 heads (no GQA), vocab 32000,
tied embeddings, RoPE θ=10000, seq len 2048, bf16, AdamW β 0.9/0.95, peak LR 5e-4, weight
decay 0.1, clip 1.0, `zero_stage 0`, fp32 grad accumulation.

**Grid:** 4 settings × 3 seeds × 6 segments = **72 jobs**, producing **48 logical runs**
(12 trunks + 36 branches) and **36 annealed finals**.

---

## 11. Outputs — what lands where

### Checkpoints, under `<ckpt_root>`

```
<ckpt_root>/seed<S>/<setting>/trunk/<setting>/seed<S>/
    0/                            the seeded init
    1500/ 3000/ 4292/ ...         restart insurance + the three branch points
    latest.txt
<ckpt_root>/seed<S>/<setting>/ep1/<setting>/seed<S>/4768/     the 1-epoch annealed model
<ckpt_root>/seed<S>/<setting>/ep2/<setting>/seed<S>/9537/     the 2-epoch annealed model
<ckpt_root>/seed<S>/<setting>/ep3/<setting>/seed<S>/14305/    the 3-epoch annealed model
```

This nested layout matches the existing rewritten-arm checkpoints, so the two grids can be
handled by the same scripts later.

| | per chain | per seed (×4) | all seeds (×12) |
|---|---:|---:|---:|
| checkpoints, peak | 295 GB | 1.2 TB | 3.5 TB |
| after pruning trunk restart points | 126 GB | 0.5 TB | 1.5 TB |

14 checkpoints per chain at 21.06 GB (optimizer 18.05 + model 3.01). **Six must be kept** until
Huicong confirms the uploads: the three branch points `4292 / 8583 / 12875` and the three
annealed finals `4768 / 9537 / 14305`. The other eight are trunk restart insurance and may be
deleted once their segment completes.

**Do not prune 4292 / 8583 / 12875 until their branch has finished.** Deleting a branch point
early makes that branch train from random init and exit 0.

### wandb

Your own project (or `mode: offline` with `wandb.dir` on shared storage). Run names are
`raw_<setting>_seed<S>_<segment>`; the rendered `.env` sets `WANDB_RUN_GROUP` and tags. wandb
is not the deliverable; the CSV export below is. If the `wandb` package is missing, training
runs fine and logs nothing.

### Deliverables per (setting, seed) — as soon as its three ep finals exist

1. **HF export** of each ep final (CPU only):
   ```bash
   python tools/kys_eval/convert_to_hf.py \
       --checkpoint_path <ckpt_root>/seed<S>/<setting>/ep<N>/<setting>/seed<S>/<step> \
       --save_path <out>/hf --tokenizer <data_root>/tokenizer
   ```
2. **Upload** to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`:
   ```
   rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/    the step directory as is (21 GB)
   rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/          the export (3 GB)
   rewrite-1p5b/seed<S>/<setting>/logs/raw_<setting>_seed<S>_<segment>.csv   step, loss, lr, tokens/s
   ```
3. **Note** per seed on branch `marc-runs`: GPU type, `dp`, `mbs`, `recompute`, measured s/it,
   any requeue or preemption and where, any deviation from this document.

---

## 12. Evaluation

You run it, with the `kys-eval` repository, after each setting's three ep finals have been
converted to HF format (section 11). It pins the exact LightEval commit, patch, task file,
dataset revisions and metric used for the 54 rewritten checkpoints ("v2"), so the raw scores
are directly comparable.

| | |
|---|---|
| repo | `github.com/imHuicongZhang/kys-eval`, branch `main` |
| what it scores | 0-shot `acc_norm` (token-normalized log-likelihood) on ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA, and the 57 MMLU subjects; reports Mean6, MMLU macro-average and the four MMLU categories |
| hardware | one H100 per checkpoint, about 9 minutes. Other GPU architectures move bf16 near-ties by about 7e-4 on Mean6; if you have no H100, say so before scoring and pass `--allow-any-gpu` |
| comparator | v2 results of the 54 rewritten checkpoints ship inside the repo under `reference/` |

Order of work:

1. Install per the kys-eval README (`install.sh`, then `python -m kys_eval.check_install` must
   end with `OK`). It builds a separate environment from the training one; do not mix them.
2. **Reference check, once per cluster, before scoring anything new:**
   ```bash
   python -m kys_eval.reference_check \
       --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf
   ```
   It re-scores a known v2 checkpoint and compares against recorded scores. Expected verdict on
   an H100 is `PASS (exact match)`. If it fails, stop and report; nothing scored on a failing
   pipeline is comparable.
3. Score each raw ep final from its HF export:
   ```bash
   python -m kys_eval.eval_checkpoint \
       --model hf://blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed<S>/<setting>/ep<N>/hf \
       --label raw_<setting>_seed<S>_ep<N>
   ```
   or the whole seed with `python -m kys_eval.run_grid` and `slurm/eval_grid_array.sbatch`
   (fill its placeholders). Checkpoints with existing results are skipped.
4. After each seed, `python -m kys_eval.aggregate` writes the markdown reports; upload
   `results/` and `reports/` next to the checkpoints under `rewrite-1p5b/seed<S>/eval/`.

The 54 v2 rewritten checkpoints themselves are public at `wytro/KYS-1.5B-Rewritten-v2`, same
layout, HF export plus nanotron weights without optimizer state.

---

## 13. Do not change these

### `micro_batch_size` — pinned for numerical comparability, not tuned for speed

The rewritten arms trained at `mbs 32, dp 4, accum 8`. `Loss.forward` computes
`masked_mean(loss, label_mask)` per micro-batch; its denominator is the number of unmasked
tokens *in that micro-batch*, which varies by a few tokens. Regrouping the same 1024 sequences
into a different `mbs` changes the effective per-token weights by order 1e-3 relative, four
orders of magnitude above fp32 rounding. **It changes the objective, not the speed.**

On 80 GB cards `mbs 32` needs `recompute_layer: true` (measured 49.6 GiB peak on one H100;
about 191 GiB without). Recomputation costs about 23% time and changes no math. If you truly
cannot run `mbs 32`, use the largest that fits, pass `--expected-mbs`, and report it; the runs
stay internally consistent but carry a documented small offset against the rewritten arms.

`dp` **is** free. Scale `dp`, never `mbs`.

### `tokenizers` — floor is 0.20

`0.19.1` cannot parse this `tokenizer.json` and raises
`data did not match any variant of untagged enum ModelWrapper`. That reads like corruption; it
is not. `transformers 4.44.x` pulls in `0.19.1`, so reusing an old environment reproduces it.
Build fresh per §2.

### The global batch — 1024 sequences, invariant

`mbs × accum × dp = 1024 sequences × 2048 = 2,097,152 tokens/step`. The renderer derives
`accum` and refuses any non-integer combination. Never set `accum` by hand.

### Data order

No reshuffle, no re-sharding, no filtering, no other tokenizer or revision, no `--n-tasks`
other than 16. The parquet order **is** the experiment.

### Launch order

`trunk1 → trunk2 → trunk3`; branches only after their own trunk segment; seeds strictly
serial. See §9.

### Files

The 72 templates, `tools/kys_raw/`, `tools/generate_configs.py`, `tools/render_config.py`,
`tools/assert_invariants.py`, `deploy/slurm/kys_segment.sbatch`. The only thing you own is the
`marc-cluster` entry and how many chains run in parallel.

---

## 14. Things that look broken and are not

**Every `.ds` is named `000NN_unshuffled.ds`.** That suffix is a datatrove filename default.
The shuffle happened upstream at the parquet stage, seed 42, identically to the rewritten
arms. The directory is the authority, not the filename.

**A shard may be 0 bytes.** With 16 tasks a task can receive no input if the parquet split is
uneven. `assert_invariants.py` checks the shard count and the total; the loader steps past an
empty shard. Keep it.

**`.cache/huggingface/` appears inside `data_root`.** `snapshot_download`'s resume ledger. Inert.
Deleting it costs a full re-download.

**Checkpoints appear at steps 1500 / 3000 / 4500 … but never at 4292.** Trunk
`checkpoint_interval` is 1500, chosen so it never coincides with a branch point. Branch points
are produced by a segment *ending* there. Branches use `checkpoint_interval: 100000`, which
never fires; they write exactly one checkpoint via `save_final_state`.

**A segment exits 0 in seconds.** Its final checkpoint already existed; the sbatch guard skipped
`torchrun`. Expected on any resubmission.

**Loss steps slightly at about step 4768 and 9537 inside trunk2 and trunk3.** The data wraps
there (epoch 2 and 3 begin). A small step is expected; a spike or a rise that persists is not.

---

## 15. If something fails

| symptom | likely cause |
|---|---|
| `ModuleNotFoundError: grouped_gemm` | not installed at the pinned commit — §2 |
| `ModelWrapper` enum error | `tokenizers < 0.20` — §13 |
| `Tokenizer passed in config … does not match dataset's` | `fix_ds_metadata.py` not run, or a different `--tokenizer-dir` — §4 |
| token total ≠ `manifest.json` after tokenizing | parquet files re-split, shuffled, or a different tokenizer — §4; stop and report |
| `Python.h: No such file` during Triton compile | set `python_include` — §2 |
| `AssertionError` on shard count | a `.ds` missing — §14 |
| `DependencyNeverSatisfied` | an upstream segment failed; find it in the job-id map `plan_submit.py` writes |
| Run completes fast with a clean anneal | **check it resumed from a real checkpoint** — §9 |
| NaN / inf loss, or loss rising for hundreds of steps | stop the chain and report; do not tune |
| wandb directory empty | package missing, or `wandb.dir` unset |

Anything involving a run that exits 0 but looks wrong: stop the chain and ask rather than
resubmitting. The failure modes this project guards against are the ones that do not crash.

---

## 16. Contacts and escalation

Stop and ask Huicong before:

- filling any value outside the `marc-cluster` entry
- using an `mbs` other than 32, or a `dp` that is not a power of two
- proceeding when a hash, token total, or invariant check fails
- proceeding after any run that exits 0 but whose curves look unexpected
- deleting anything under `<ckpt_root>` other than the eight prunable trunk restart points
- starting seed 43 before seed 42 is fully uploaded

Report to Huicong, without being asked:

- the §6 smoke results: `lm_loss` at steps 1/10/20 and `time_per_iteration_ms`
- the four tokenized totals from §4
- peak memory and s/it from the first `trunk1` that runs
- each setting's three ep finals uploaded
- each seed's completion
